# ABC-DiT fast inference

Low-latency inference for the [ABC](https://abc.bot) DiT policy: **184 ms → 49 ms per
action chunk on an RTX 3090** (3 cameras, 10 flow-matching steps, 30-action chunk),
with actions as close to the fp32 reference as the stock bf16 path.

## 1. Install

Linux with an NVIDIA GPU and a driver compatible with CUDA 12.8 (the pinned PyTorch
build), Python 3.12, and [uv](https://docs.astral.sh/uv/):

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# System libraries: ffmpeg for videos, EGL/GL for headless MuJoCo.
sudo apt-get install -y ffmpeg libegl1 libgl1

# Python 3.12 venv with every pinned dependency (torch 2.11.0+cu128, MuJoCo, MJWarp, ...)
git clone https://github.com/Shade5/abc.git
cd abc
uv python pin 3.12
uv sync
```

Download the bottles-in-bin checkpoint (~8.1 GB, includes its DINOv3 vision backbone
and normalization statistics) and the simulator assets:

```bash
uv run prepare.py --checkpoint   # cache/bottles_75k.pt
uv run prepare.py --sim          # sim scenes, robot and object meshes
```

Files land in `cache/`, or in `$ABC_CACHE` if set.

## 2. Optimize inference

Fast inference is on by default for sim evaluation and the viewer, and is enabled with
`--fast-inference` for deployment. It casts the policy to bf16 and compiles a fused
sampler (`abc_minimal/fast_dit.py`) with `torch.compile` (max-autotune, CUDA graphs).

At batch size 1 the sampler is limited by reading weights from GPU memory, and the
stock Euler loop re-reads two sets of them on every step although their inputs never
change across steps:

- **adaLN modulations** depend only on (state, task, t), and the t schedule is fixed,
  so all 32 blocks' modulations for all 10 steps (plus the t = 0 row that RTC prefix
  positions use) come out of one GEMM per inference.
- **Cross-attention K/V** depend only on the vision tokens, so all 32 blocks' K/V come
  out of one GEMM per inference.

Each step then runs only the self-attention, the cross-attention query/output
projections and the MLP: about 40% fewer weight reads. The math is unchanged (it
matches the stock sampler to fp32 rounding). One compiled step is reused for all 10
Euler steps, which also keeps compile time down. Set
`DiTInferencePolicy.fused_sampler = False` to compile the stock sampler instead.

The first fast call compiles and autotunes (about 9 minutes on a 3090); later calls
replay CUDA graphs.

### Results

RTX 3090, `bottles_75k.pt`, one rendered sim observation (3 cameras at 168x224),
RTC prefix of 4 actions, 10 steps. Mean `infer()` latency per action chunk, numpy
observation in and actions out:

| Configuration | Latency | Speedup | Max action diff vs fp32 | Compile + warmup |
| --- | ---: | ---: | ---: | ---: |
| Eager fp32 (`--no-fast-inference`) | 184.4 ms | 1.0x | — | — |
| Stock fast inference (bf16, compiled sampler) | 70.4 ms | 2.6x | 0.020 std | 832 s |
| **Fused fast inference (default)** | **49.0 ms** | **3.8x** | 0.019 std | 541 s |

Action differences are in units of the per-dimension action std. An NVIDIA L4 is about
2.5x slower than the 3090 here (408 ms eager, 184 ms stock fast), since it has about a
third of the memory bandwidth.

Reproduce on your GPU (each mode runs in its own process; run `eager` first, as the
others compare against its actions):

```bash
uv run scripts/bench_inference.py --mode eager   # fp32 reference
uv run scripts/bench_inference.py --mode stock   # bf16, stock sampler compiled
uv run scripts/bench_inference.py --mode fused   # bf16, fused sampler compiled
```

Next steps: the fused sampler reaches about half of the 3090's memory bandwidth, so
capturing the whole sampler in one CUDA graph should help, and int8 weight-only
quantization would roughly halve the weight reads again (it changes the numerics and
needs an accuracy check).

## 3. Run inference

### In simulation

Roll out the policy in the bottles-in-bin sim task with fast inference (the first
launch also compiles MJWarp's CUDA kernels, about a minute):

```bash
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --num-worlds 20 \
    --save-video --video-every-n-actions 15
```

Results go to `outputs/sim_eval_put_plastic_bottles_in_bin/` (`summary.json` with
the success rate, and `world_*.mp4` videos with `--save-video`). Pass
`--no-fast-inference` for the eager fp32 path.

To watch the policy live in a viser window at `localhost:8080`:

```bash
uv run viz_policy.py --sim.task put_plastic_bottles_in_bin --port 8080
```

### From Python

```python
from pathlib import Path

import numpy as np

from abc_minimal.config import SimEvalConfig
from abc_minimal.policy import DiTInferencePolicy

config = SimEvalConfig(
    checkpoint="cache/bottles_75k.pt",
    prompt="sim put the plastic bottles in the bin",
)
policy = DiTInferencePolicy(
    Path(config.checkpoint), config, "cuda", model_config=config.model
)
policy.enable_fast_inference(rtc_prefix_length=4)  # compile + warm up once

obs = {
    "state": state,  # float32 (14,) joint state
    "images": {  # uint8 (3, H, W) per camera
        "top": top_image,
        "left": left_wrist_image,
        "right": right_wrist_image,
    },
    "prompt": config.prompt,
}
actions = policy.infer(obs)  # float32 (30, 14), unnormalized joint targets

# RTC: condition on the next 4 not-yet-executed actions of the previous chunk.
actions = policy.infer(obs, action_prefix=previous[:4], prefix_length=4)
```

### Policy server for a real robot

```bash
uv sync --extra deploy
CUDA_VISIBLE_DEVICES=0 uv run deploy/serve_policy.py \
    --policy.checkpoint-path=cache/bottles_75k.pt \
    --policy.prompt='throw plastic bottles in bin' \
    --policy.fast-inference
```
