"""Ahead-of-time compiled ABC-DiT sampler: a per-GPU engine file that loads in seconds.

``torch.compile`` caches still re-trace the model, re-capture CUDA graphs and need
the full fp32 checkpoint on every start (about 6 minutes on an RTX PRO 4000).
This module instead exports FusedDiTSampler's two compiled functions, ``prepare``
and ``step``, with AOTInductor into one ``.pt2`` package holding the compiled
kernels, the bf16 weights they read and the normalization stats. Loading it
needs neither the checkpoint nor the Python model.

The prefix length is a mask input rather than a constant, so one engine and one
CUDA graph serve both plain and RTC (prefix-conditioned) sampling: an all-false
mask leaves every value as the plain sampler computes it.

Engines are specific to the GPU model, the torch version, the checkpoint and the
sampler shape; ``engine_key`` records all of them and a mismatch means rebuild.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from abc_minimal.fast_dit import FusedDiTSampler

logger = logging.getLogger(__name__)

ENGINE_FORMAT = 1
METADATA_FILE = "abc_engine.json"


class EngineMismatch(Exception):
    """The engine file exists but was built for something else, or is unreadable."""


def engine_key(checkpoint: Path, model_config: Any, diffusion_steps: int, device: torch.device) -> dict:
    """Everything an engine is specific to. Engines whose key differs are rebuilt.

    The checkpoint is identified by name and size; when it isn't on this machine
    (an engine needs no checkpoint to load), by name alone.
    """
    identity = {"name": checkpoint.name}
    if checkpoint.exists():
        identity["size"] = checkpoint.stat().st_size
    props = torch.cuda.get_device_properties(device)
    return {
        "format": ENGINE_FORMAT,
        "torch": torch.__version__,
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "checkpoint": identity,
        "model": json.loads(json.dumps(dataclasses.asdict(model_config))),
        "diffusion_steps": diffusion_steps,
    }


class _Prepare(torch.nn.Module):
    """Tensor-only wrapper of FusedDiTSampler.prepare for torch.export."""

    def __init__(self, sampler: FusedDiTSampler, camera_keys: list[str], num_steps: int):
        super().__init__()
        self.sampler = sampler
        self.camera_keys = camera_keys
        self.num_steps = num_steps

    def forward(self, state, task_vec, noise, action_prefix, prefix_mask, *images):
        batch = {
            "state": state,
            "task_vec_clip": task_vec,
            "images": dict(zip(self.camera_keys, images)),
        }
        x_t, mods, k_all, v_all, _, action_prefix = self.sampler.prepare(
            batch, self.num_steps, noise, action_prefix, prefix_mask=prefix_mask
        )
        return x_t, mods, k_all, v_all, action_prefix


class _Step(torch.nn.Module):
    """Tensor-only wrapper of FusedDiTSampler.step for torch.export."""

    def __init__(self, sampler: FusedDiTSampler, num_steps: int):
        super().__init__()
        self.sampler = sampler
        self.dt = -1.0 / num_steps

    def forward(self, x_t, step_mod, zero_mod, k_all, v_all, prefix_mask, action_prefix):
        return self.sampler.step(
            x_t, step_mod, zero_mod, k_all, v_all, prefix_mask, action_prefix, self.dt
        )


def build_engine(
    path: Path,
    sampler: FusedDiTSampler,
    camera_keys: list[str],
    num_steps: int,
    example: dict[str, Any],
    metadata: dict[str, Any],
    compile_mode: str = "max-autotune",
) -> None:
    """Export, AOT-compile and package the sampler at ``path``.

    ``example`` holds batch-1 inputs in the dtypes ``DiTInferencePolicy.infer``
    passes: ``state``, ``task_vec``, ``noise``, ``images`` (dict by camera).
    The package is written beside the system temp dir first and then copied, so
    a failed build never leaves a file at ``path``.
    """
    from torch.export.pt2_archive._package import package_pt2

    device = example["state"].device
    chunk = example["noise"].shape[1]
    prefix = torch.zeros_like(example["noise"])
    mask = torch.zeros(1, chunk, 1, dtype=torch.bool, device=device)
    images = tuple(example["images"][cam] for cam in camera_keys)
    prepare_args = (example["state"], example["task_vec"], example["noise"], prefix, mask, *images)

    options: dict[str, Any] = {"aot_inductor.package": True}
    if compile_mode == "max-autotune":
        options["max_autotune"] = True
    prepare = _Prepare(sampler, camera_keys, num_steps)
    step = _Step(sampler, num_steps)
    with torch.no_grad():
        x_t, mods, k_all, v_all, action_prefix = prepare(*prepare_args)
        step_args = (x_t, mods[:, 1], mods[:, 0], k_all, v_all, mask, action_prefix)
        files = {}
        for name, module, args in (("prepare", prepare, prepare_args), ("step", step, step_args)):
            logger.warning("exporting and AOT-compiling the sampler's %s", name)
            exported = torch.export.export(module, args, strict=False)
            files[name] = torch._inductor.aot_compile(exported.module(), args, options=options)

    metadata = {
        **metadata,
        "camera_keys": list(camera_keys),
        "inputs": {
            "state": list(example["state"].shape),
            "task_vec": list(example["task_vec"].shape),
            "task_vec_dtype": str(example["task_vec"].dtype).removeprefix("torch."),
            "noise": list(example["noise"].shape),
            "image": list(images[0].shape),
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "engine.pt2"
        package_pt2(
            local, aoti_files=files, extra_files={METADATA_FILE: json.dumps(metadata)}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        # Copied, not renamed: global volumes don't support atomic rename, and a
        # partial copy fails load_engine's checks and is rebuilt.
        shutil.copyfile(local, path)
    logger.warning("saved compiled engine to %s (%.0f MB)", path, path.stat().st_size / 1e6)


class SamplerEngine:
    """A loaded engine: runs prepare + every Euler step as one CUDA graph replay."""

    def __init__(self, prepare, step, metadata: dict[str, Any], device: torch.device):
        self._prepare = prepare
        self._step = step
        self.metadata = metadata
        self.device = device
        self.num_steps = int(metadata["diffusion_steps"])
        self.camera_keys = list(metadata["camera_keys"])
        shapes = metadata["inputs"]
        task_dtype = getattr(torch, shapes["task_vec_dtype"])
        z = lambda shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype, device=device)  # noqa: E731
        self._state = z(shapes["state"])
        self._task_vec = z(shapes["task_vec"], task_dtype)
        self._noise = z(shapes["noise"])
        self._prefix = z(shapes["noise"])
        self._mask = z([1, shapes["noise"][1], 1], torch.bool)
        self._images = {cam: z(shapes["image"]) for cam in self.camera_keys}
        self._graph: torch.cuda.CUDAGraph | None = None
        self._out: torch.Tensor | None = None

    @property
    def norm_stats(self) -> dict[str, Any]:
        return self.metadata["norm_stats"]

    @property
    def trained_max_prefix(self) -> int:
        return int(self.metadata["trained_max_prefix"])

    def _run(self) -> torch.Tensor:
        images = [self._images[cam] for cam in self.camera_keys]
        x_t, mods, k_all, v_all, action_prefix = self._prepare(
            self._state, self._task_vec, self._noise, self._prefix, self._mask, *images
        )
        for i in range(self.num_steps):
            x_t = self._step(x_t, mods[:, i + 1], mods[:, 0], k_all, v_all, self._mask, action_prefix)
            if isinstance(x_t, (list, tuple)):
                x_t = x_t[0]
        return x_t

    def capture(self) -> None:
        """Warm up on a side stream, then capture the whole sampler in one CUDA graph."""
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):
                self._run()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            self._out = self._run()
        self._graph = graph
        torch.cuda.synchronize(self.device)

    def _check(self, name: str, tensor: torch.Tensor, static: torch.Tensor) -> None:
        if tensor.shape != static.shape:
            raise ValueError(
                f"the compiled engine takes {name} of shape {tuple(static.shape)}, got "
                f"{tuple(tensor.shape)}; the engine serves one observation at a time"
            )

    @torch.no_grad()
    def sample(self, batch, noise=None, action_prefix=None, prefix_length: int = 0) -> torch.Tensor:
        """Same contract as FusedDiTSampler.sample at batch size 1."""
        self._check("state", batch["state"], self._state)
        self._state.copy_(batch["state"])
        self._task_vec.copy_(batch["task_vec_clip"])
        for cam in self.camera_keys:
            image = batch["images"][cam]
            self._check(f"image {cam!r}", image, self._images[cam])
            self._images[cam].copy_(image)
        if noise is None:
            self._noise.normal_()
        else:
            self._noise.copy_(noise.reshape(self._noise.shape))
        if action_prefix is None:
            self._mask.zero_()
        else:
            self._prefix.copy_(action_prefix.reshape(self._prefix.shape))
            self._mask.copy_(
                (torch.arange(self._mask.shape[1], device=self.device) < prefix_length).view(self._mask.shape)
            )
        if self._graph is None:
            return self._run().clone()
        self._graph.replay()
        return self._out.clone()


def load_engine(path: Path, expected_key: dict[str, Any], device: torch.device) -> SamplerEngine:
    """Load and graph-capture an engine; EngineMismatch when it doesn't fit ``expected_key``."""
    from torch.export.pt2_archive._package import load_pt2

    try:
        # Single-threaded runners skip the cross-thread event bookkeeping that
        # CUDA graph capture rejects.
        contents = load_pt2(
            path,
            run_single_threaded=True,
            device_index=device.index if device.index is not None else -1,
        )
        metadata = json.loads(contents.extra_files[METADATA_FILE])
        prepare = contents.aoti_runners["prepare"]
        step = contents.aoti_runners["step"]
    except Exception as e:
        raise EngineMismatch(f"unreadable engine file ({e!r})") from e
    found = {k: metadata.get(k) for k in expected_key}
    if isinstance(found["checkpoint"], dict):
        found["checkpoint"] = {k: found["checkpoint"].get(k) for k in expected_key["checkpoint"]}
    if found != expected_key:
        diff = {k: (found[k], expected_key[k]) for k in expected_key if found[k] != expected_key[k]}
        raise EngineMismatch(f"engine built for something else (found, expected): {diff}")
    engine = SamplerEngine(prepare, step, metadata, device)
    engine.capture()
    return engine


def norm_stats_to_json(norm_stats: dict[str, Any]) -> dict[str, Any]:
    """Parsed norm stats (numpy arrays) as JSON lists; parse_norm_stats reads them back."""
    return {
        key: {k: np.asarray(v).tolist() for k, v in stats.items()}
        for key, stats in norm_stats.items()
    }
