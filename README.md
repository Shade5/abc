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

### Adamo server for a robot streamed over Adamo

`deploy/adamo_server.py` is a FastAPI server that drives a robot streamed over
[Adamo](https://docs.adamohq.com) (for example the `abc-sim` bottles-in-bin sim). It
decodes the robot's camera tracks and reads its joint state, runs the fast
policy with RTC, and publishes 14-D joint targets at 30 Hz as JointState JSON on
`{robot}/control/joint_state`.

On a fresh node, after the install and checkpoint steps above:

```bash
# GStreamer, which the adamo SDK links against and decodes video with
# (appsrc ! h264parse ! avdec_h264). uv can't install it: GStreamer's PyPI
# wheels are macOS and Windows only.
sudo apt-get install -y --no-install-recommends \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-bad gstreamer1.0-libav

uv sync --extra adamo   # adamo==0.4.59, fastapi, uvicorn
export ADAMO_API_KEY=ak_...   # optional; /start can take the key instead
uv run deploy/adamo_server.py --policy.checkpoint-path=cache/bottles_75k.pt --port 8000
```

The server compiles the policy at startup, about 13 minutes on a 3090.
`--engine-path` keeps the compiled policy in one file, so later starts skip
the compile: when the file exists it is loaded first, otherwise it is written
after compiling. Engines are specific to the GPU model and the torch version,
so keep one per GPU, for example on a shared volume:

```bash
uv run deploy/adamo_server.py --policy.checkpoint-path=cache/bottles_75k.pt \
    --engine-path /workspace/abc/3090/engine.bin   # 541 MB; 2 min startup instead of 13
```

`GET /status` reports `"policy": "ready"` when it's done. Then:

```bash
curl -X POST localhost:8000/start -H 'content-type: application/json' \
    -d '{"robot_name": "abc-sim"}'
curl localhost:8000/status               # cameras, state, inference latency, control
curl -X POST localhost:8000/take_control  # reclaim after an operator took over
curl -X POST localhost:8000/stop          # release control and disconnect
```

`/start` takes `adamo_api_key` (defaults to `$ADAMO_API_KEY`), `robot_name`, and
optionally `prompt` and the camera track names `head`, `wrist_left`, `wrist_right`
(those are the defaults; `head` is a side-by-side stereo pair whose left view is the
policy's `top` camera, set `head_stereo: false` otherwise). It claims the robot as
an Adamo operator, sending `acquired`, then a heartbeat every second on
`{robot}/control/json/operator_control`. If another operator acquires the robot,
the policy stops sending actions until `/take_control`. `GET /topics` lists every
key the robot publishes on. If a camera sends no frame for `frame_timeout_s`
(default 30, counted from `/start` until its first frame; `null` disables it), the
run stops as with `/stop`, and `/status` reports why under `last_stop`.

The adamo SDK is pinned to 0.4.59 because 1.0 has no API to publish arbitrary
topics or receive video.

#### On RunPod

1. Deploy an RTX 3090 pod (Secure Cloud) from the console with the image
   `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` and at least 30 GB of
   container disk. To reuse a compiled engine across pods, attach a
   [global volume](https://docs.runpod.io/storage/globalvolume/overview): it mounts
   at `/workspace` and holds `abc/3090/engine.bin`. Global volumes can only be
   attached from the console for now, not from the API or `runpodctl`. Keep the
   default `8888/http` port, which RunPod serves at
   `https://<pod-id>-8888.proxy.runpod.net`. Adding a port later restarts the pod
   and wipes its container disk.

2. SSH in and install. The repo, venv and checkpoint live on the container disk,
   so a pod restart means running this again:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH
   apt-get update && apt-get install -y ffmpeg libegl1 libgl1
   apt-get install -y --no-install-recommends \
       gstreamer1.0-plugins-base gstreamer1.0-plugins-bad gstreamer1.0-libav
   git clone https://github.com/Shade5/abc.git && cd abc
   uv python pin 3.12 && uv sync --extra adamo
   ABC_CACHE=/root/cache uv run prepare.py --checkpoint
   ```

3. Serve on 8888 in place of Jupyter. The first run on a new GPU type compiles
   for about 13 minutes and writes the engine; later pods load it in about 2:

   ```bash
   pkill -f jupyter-lab
   setsid nohup uv run deploy/adamo_server.py \
       --policy.checkpoint-path=/root/cache/bottles_75k.pt \
       --engine-path /workspace/abc/3090/engine.bin --port 8888 \
       > /root/server.log 2>&1 < /dev/null &
   ```

4. From any machine, once `/status` says `ready`:

   ```bash
   curl https://<pod-id>-8888.proxy.runpod.net/status
   curl -X POST https://<pod-id>-8888.proxy.runpod.net/start \
       -H 'content-type: application/json' \
       -d '{"adamo_api_key": "ak_...", "robot_name": "abc-sim"}'
   curl -X POST https://<pod-id>-8888.proxy.runpod.net/stop
   ```

The proxy URL has no authentication, so anyone with it can start and stop runs.
They still need their own Adamo key to send actions, unless `ADAMO_API_KEY` is set
in the server's environment. To keep the server private, leave it on port 8000
and forward the port over the pod's direct SSH (the `ssh.runpod.io` proxy doesn't
forward ports):
`ssh -N -L 8000:localhost:8000 root@<pod-ip> -p <pod-ssh-port>`.
