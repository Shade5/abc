"""Batch-1 DiT inference latency on one sim observation (3 cameras).

Times DiTInferencePolicy.infer end to end (numpy observation in, unnormalized
action chunk out) with the sim eval defaults: 168x224 cameras, 10 Euler steps,
RTC prefix of 4 actions. Each mode runs in its own process:

    uv run scripts/bench_inference.py --mode eager    # --no-fast-inference (fp32)
    uv run scripts/bench_inference.py --mode stock    # bf16 + compiled stock sampler
    uv run scripts/bench_inference.py --mode fused    # bf16 + FusedDiTSampler, step compiled once
    uv run scripts/bench_inference.py --mode engine --engine-path engine.pt2  # AOT-compiled fused sampler

Actions use fixed noise and are saved under --out; every mode after the first
prints its max difference from the eager fp32 actions.

RTX 3090, torch 2.11.0+cu128, bottles_75k.pt, mean infer() latency per RTC chunk
(max |action diff| vs eager fp32 in action std units; compile + warmup time):

    eager  184.4 ms
    stock   70.4 ms  (0.020; 832 s)
    fused   49.0 ms  (0.019; 541 s)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import resolve_prompt
from abc_minimal.policy import DiTInferencePolicy

CACHE = Path(os.environ.get("ABC_CACHE", "cache"))


def sim_observation(config: SimEvalConfig, path: Path) -> dict:
    """One rendered reset observation, cached so every mode sees the same pixels."""
    if not path.exists():
        from abc_minimal.sim_env import SimTaskEnv

        env = SimTaskEnv(
            task=config.task,
            height=config.camera_height,
            width=config.camera_width,
            camera_keys=tuple(config.model.camera_keys),
            prompt=config.prompt,
        )
        obs = env.reset(seed=config.seed)
        env.close()
        np.savez(
            path,
            state=obs["state"],
            prompt=obs["prompt"],
            **{f"img_{cam}": img for cam, img in obs["images"].items()},
        )
    data = np.load(path)
    return {
        "state": data["state"],
        "prompt": str(data["prompt"]),
        "images": {cam: data[f"img_{cam}"] for cam in config.model.camera_keys},
    }


def timed(fn, iters: int) -> np.ndarray:
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return np.asarray(out)


def stats(ms: np.ndarray) -> dict:
    return {
        "mean": float(ms.mean()),
        "p50": float(np.median(ms)),
        "p90": float(np.percentile(ms, 90)),
        "std": float(ms.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["eager", "stock", "fused", "engine"], required=True)
    parser.add_argument(
        "--engine-path", help="engine mode: the .pt2 engine, built first when missing"
    )
    parser.add_argument("--checkpoint", default=str(CACHE / "bottles_75k.pt"))
    parser.add_argument("--task", default="put_plastic_bottles_in_bin")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--prefix-length", type=int, default=4)
    parser.add_argument("--out", default="outputs/bench_inference")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = SimEvalConfig(checkpoint=args.checkpoint, task=args.task)
    config.prompt = resolve_prompt(config)
    obs = sim_observation(config, out / f"obs_{args.task}.npz")
    print(f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} prompt={obs['prompt']!r}")

    t0 = time.perf_counter()
    if args.mode == "engine":
        if not args.engine_path:
            parser.error("--mode engine needs --engine-path")
        policy = DiTInferencePolicy.from_engine(
            Path(args.engine_path), Path(args.checkpoint), config, "cuda",
            model_config=config.model, compile_mode=config.fast_compile_mode,
        )
    else:
        policy = DiTInferencePolicy(Path(args.checkpoint), config, "cuda", model_config=config.model)
    print(f"policy load {time.perf_counter() - t0:.1f}s")
    m = config.model
    noise = np.random.default_rng(0).standard_normal((m.chunk_length, m.action_dim), dtype=np.float32)
    # A realistic RTC prefix: hold the current joint state for prefix_length actions.
    prefix = np.repeat(obs["state"][None, : m.action_dim], args.prefix_length, axis=0)

    t0 = time.perf_counter()
    if args.mode == "engine":
        for _ in range(3):
            policy.infer(obs, noise=noise)
            policy.infer(obs, noise=noise, action_prefix=prefix, prefix_length=args.prefix_length)
    elif args.mode != "eager":
        policy.fused_sampler = args.mode == "fused"
        policy.enable_fast_inference(
            compile_mode=config.fast_compile_mode,
            warmup_obs=obs,
            warmup_noise=noise,
            rtc_prefix_length=args.prefix_length,
        )
    else:
        for _ in range(3):
            policy.infer(obs, noise=noise)
            policy.infer(obs, noise=noise, action_prefix=prefix, prefix_length=args.prefix_length)
    print(f"setup + warmup {time.perf_counter() - t0:.1f}s")

    def plain():
        return policy.infer(obs, noise=noise)

    def rtc():
        return policy.infer(obs, noise=noise, action_prefix=prefix, prefix_length=args.prefix_length)

    iters = args.iters if args.mode != "eager" else min(args.iters, 30)
    result = {
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(),
        "rtc_ms": stats(timed(rtc, iters)),
        "plain_ms": stats(timed(plain, iters)),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    actions = {"rtc": rtc(), "plain": plain()}
    np.savez(out / f"actions_{args.mode}.npz", **actions)
    ref_path = out / "actions_eager.npz"
    if args.mode != "eager" and ref_path.exists():
        ref = np.load(ref_path)
        # Actions are unnormalized joint targets; also report it in the
        # normalized space the model samples in (std units).
        std = policy.norm_stats["actions"]["std"] + 1e-6
        for key in actions:
            diff = np.abs(actions[key] - ref[key])
            result[f"{key}_max_abs_diff"] = float(diff.max())
            result[f"{key}_max_norm_diff"] = float((diff / std).max())
    print(json.dumps(result, indent=2))
    (out / f"result_{args.mode}_{torch.cuda.get_device_name().replace(' ', '_')}.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
