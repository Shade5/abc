# **Scalable Behavior Cloning with Open Data, Training, and Evaluation**

<p align="center">
  <strong>
    <a href="https://abc.bot">Project Website</a> |
    <a href="https://abc.bot/abc.pdf">Paper</a> |
    <a href="https://huggingface.co/datasets/XDOF/ABC-130k">Raw Data</a>
  </strong>
</p>

![](assets/teaser.jpg)


Code for the ABC project.

## Release status

This release includes ABC-DiT and ABC-VLA training, pretrained and
task-finetuned checkpoints, simulation and evaluation tools, real-robot
deployment, and data conversion utilities. Use `prepare.py --sim-data-list`
to see the currently published simulation datasets and
`prepare.py --sim-bundle-list` to browse available evaluation bundles.

## Repo layout

This README covers setup, a short evaluation smoke test, and training. The package READMEs below hold the full reference for their areas.

| Package | What it holds |
| --- | --- |
| [`abc_minimal/`](abc_minimal/README.md) | ABC models, dataloader, training loop, policy inference, episode tools |
| [`abc_sim/`](abc_sim/README.md) | self-contained simulator: MuJoCo scenes, task catalogue, randomization, evaluators, Gym API, sim eval |
| [`deploy/`](deploy/README.md) | real-robot deployment: local/remote inference, RTC, teleop, DAgger, recording |
| [GELLO hardware](deploy/gello/README.md) | printable parts, bill of materials, and assembly guide |

`train.py`, `eval_policy.py`, `viz_episode.py`, `viz_policy.py`, and `prepare.py` at the root are the entrypoints; `scripts/` holds the data conversion utilities.

## Setup

The training and default GPU evaluation commands below target Linux with an
NVIDIA GPU and a driver compatible with CUDA 12.8 (the pinned PyTorch build).
Use Python 3.12. The reference DiT training run uses 8 H100/H200 GPUs with
80 GB VRAM each; reduce the GPU count and per-GPU `--batch-size` for smaller
machines. Minimum evaluation VRAM has not been established.

```bash
# Install uv if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Install ffmpeg.
sudo apt-get install -y ffmpeg     # on Linux
```

```bash
git clone https://github.com/amazon-far/abc.git
cd abc
# Pin Python and create the project venv. uv reads pyproject.toml here.
uv python pin 3.12
uv sync
```

## Quick evaluation smoke test

After setup, download the bottles checkpoint (~8.1 GB), preview data, and its
simulation assets, then run one short rollout on a single NVIDIA GPU:

```bash
uv run prepare.py --checkpoint
uv run eval_policy.py \
    --checkpoint "${ABC_CACHE:-cache}/bottles_75k.pt" \
    --num-worlds 1 --num-chunks 2 --no-fast-inference \
    --save-video --video-every-n-actions 15
```

The checkpoint includes its vision backbone and normalization statistics;
no separate DINO weight download is needed for this evaluation. The first
launch compiles MJWarp CUDA kernels (approximately one minute); download and
rollout times depend on your connection and GPU. This short run skips the
optional inference compilation.

Expect `summary.json` and a `world_*.mp4` video under
`outputs/sim_eval_put_plastic_bottles_in_bin/`. Completing the run checks
checkpoint loading, rendering, and policy inference. Two action chunks are
too short to measure task success; use the full [evaluation](#evaluation)
instructions for that.

## Training

First we need to download the requisite data (norm stats and either a sample or full data.)
```bash
uv run prepare.py            # preview (a few episodes of data, ~130MB)
uv run prepare.py --full     # all data for bottles in bin (~35GB)
uv run prepare.py --checkpoint  # add to also pull the pretrained 75k policy (~8.1GB)
uv run prepare.py --sim-data sim_spell_abc  # one sim task's episodes + its assets (--sim-data-list to browse)
```

This populates the cache dir (default `cache/`, or `ABC_CACHE` if set) with:

```
cache/
  norm_stats.json                       # state/action z-score stats
  train_real/episode_<uuid>/{states_actions.bin, combined_camera-images-rgb.mp4, episode_metadata.json}
  val_real/...
  train_sim/...
  val_sim/...
```

Set `ABC_CACHE=/path/to/cache` before running commands if you want the cache
outside the repository.

:warning: Note: `prepare.py` does not download DINO weights. Review and follow the DINO license terms, then download the weights from [Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) or [Hugging Face](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m). Save the file as `dinov3_vitb16_pretrain_lvd1689m.pth` in the cache dir. :warning:

### ABC-DiT Training

The command to run training is below. Note that this is for single node training with 8 GPUs, change `nproc-per-node` if you want.

```bash
uv run torchrun --standalone --nproc-per-node 8 train.py

# Resume from a local checkpoint, including optimizer/scheduler and data stream position.
uv run torchrun --standalone --nproc-per-node 8 train.py --resume-from cache/finetune_checkpoints/last.pt
```
The dataclass config is exposed as CLI flags; `uv run python train.py --help` shows training, optimizer, flow, CLIP asset, and model options. The default model config is the checkpoint-compatible ABC-DiT XL shape.

If you pulled with `--full` above, this checkpoint is expected to work for the bottles in bin task in both sim and real. The performance should be similar to [this](assets/bottles_real.mp4).

Training defaults in `abc_minimal/config.py` match the production reference finetune (lr 1e-4 with a 1k-step linear warmup, AdamW(0.9, 0.95), wd 0.01, grad clip 10, prefix conditioning max 8 with noise 0.05, 10% state masking, batch 90/GPU, 75k steps, hours-weighted 2-component mixture).

If you have fewer GPUs than 8 you need to reduce nproc per node or if you have less than 80Gb of VRAM you may need to reduce `--batch-size`.  The above training yields ~2.6-3 iterations / sec on H100/H200. It achieves a training loss of ~`0.048` after 75k steps.  The DiT policy also supports a CLIP ViT-B/16 vision backbone (in place of DINOv3) via `--model.vision-backbone clip`.

### ABC-VLA Training

A diffusion-only port of the Gemma 3 VLA: Gemma 3 4B with SigLIP at 224x224, one
selectable Gemma feature layer, QK-normalized learned-query pooling, a state token
plus optional direct state conditioning, and a small AdaLN DiT head. FAST tokens
and the alternative conditioners are not included. VLA training supports FSDP
with the restrictions described below.

Training from the Gemma base needs Google's `gemma_pytorch` checkpoint
(`google/gemma-3/pyTorch/gemma-3-4b-pt` on Kaggle, under the Gemma Terms of Use);
the Hugging Face format is not compatible. Select the policy with `--policy vla`:

```bash
uv run torchrun --standalone --nproc-per-node 8 train.py \
    --policy vla \
    --vla-model.backbone.checkpoint /path/to/gemma3_4b_pt.pt \
    --flow.num-diffusion-draws 4
```

The `--prompt.*` subtask and operator options apply to the VLA as to the DiT;
neither has been validated for it. `--fsdp` shards parameters, gradients, and
Adam state across the ranks instead of replicating them, roughly halving
per-GPU memory on two GPUs. Launch with `torchrun --nproc-per-node` greater
than 1; `--fsdp` is VLA-only and cannot be combined with `--compile-siglip`.

### Finetuning from a released checkpoint

To finetune from a released checkpoint instead of training from scratch, download the parent and pass `--load-pretrained` (fresh optimizer, step 0).

#### ABC-DiT Finetuning

```bash
# Pulls cache/abc_dit_xl_200k_model.pt (~8.1 GB)
uv run prepare.py --pretrained
uv run train.py --load-pretrained
```

`--pretrained-ckpt-name` picks the checkpoint file inside the cache dir (default
`abc_dit_xl_200k_model.pt`, which is what `--pretrained` downloads). Use
`--model.vision-backbone clip` when the parent is a CLIP-DiT checkpoint; DINOv3
is the default.

#### ABC-VLA Finetuning

The same weights-only path works for VLA checkpoints and needs no Gemma base:

```bash
uv run prepare.py --vla-pretrained

uv run train.py \
    --policy vla \
    --load-pretrained \
    --pretrained-ckpt-name vla_abc130k_200000_v2.pt
```

`--vla-pretrained` fetches the recommended `abc130k` step-200000 parent, verifies
its checksum, and installs the assets for its five sim tasks;
`--vla-pretrained-family {abc130k,200k}` and `--vla-pretrained-step
{50000,100000,200000}` select the others (the `200k` family trained on xdof only).
The v2 files embed norm stats and architecture metadata, which is checked against
the CLI config before strict loading; `--resume-from` is only for a stateful
continuation.

**Multi-node training, the episode data format, and prompt conditioning options
are documented in the [abc_minimal README](abc_minimal/README.md).**

## Viewing the Sim Data

To download and view particular tasks from the sim data, and visualise the episodes, use the following

```bash
uv run prepare.py --sim-data-list # list possible tasks
uv run prepare.py --sim-data conveyor_pick # download one from the list
uv run viz_episode.py --root cache/train_sim --port 8080 # visualise data
```

The replay modes (pose playback vs physics re-simulation) are described in the
[abc_minimal README](abc_minimal/README.md#visualizing-episodes-and-policies).

## Evaluation

`eval_policy.py` evaluates a checkpoint on the `abc_sim/` task catalogue: one
you trained yourself (`cache/finetune_checkpoints/last.pt`) or a released one:

```bash
# bottles_75k.pt: the 75k-step bottles-only policy, with norm_stats.json and the preview tar.
uv run prepare.py --checkpoint

# abc_dit_xl_200k_model.pt: the multi-task DiT parent.
uv run prepare.py --pretrained

# vla_abc130k_200000_v2.pt: the VLA parent, with its five tasks' assets.
uv run prepare.py --vla-pretrained

# A per-task finetune at its recommended step, sha-verified, with its eval command
# printed. --sim-checkpoint-list shows the catalogue with results;
# --sim-checkpoint-full-state fetches the ~24 GB training-state file instead.
uv run prepare.py --sim-checkpoint pour
```

Every download installs the sim assets its checkpoint needs and prints eval and
viewer commands. Both tools use the prompt each checkpoint trained under, so
`--sim.prompt` and `--sim.checkpoint` are only for overrides.

To watch a policy live in a viser window at `localhost:8080`, prepare its task
bundle once (the assets plus the recommended model-only checkpoint, no episode
data), then select the task in the viewer. Every task has a DiT bundle and a
`vla_` bundle; `--sim-bundle-list` shows all of them with their results, and
`--sim-force` refreshes a cached manifest:

```bash
uv run prepare.py --sim-bundle-list
uv run prepare.py --sim-bundle put_plastic_bottles_in_bin
uv run prepare.py --sim-bundle vla_lego_blocks_sorting
uv run viz_policy.py --sim.task put_plastic_bottles_in_bin --port 8080
```

![](assets/sim_eval.gif)

To run the evaluation, download the sim assets once (`uv run prepare.py --sim`;
the first launch also compiles MJWarp's CUDA kernels, ~1 min):

```bash
# 20 worlds, save a video of each rollout, log per-chunk progress.
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --num-worlds 20 \
    --save-video --video-every-n-actions 15 --log-every-chunk

# Output: $REPO/outputs/sim_eval_put_plastic_bottles_in_bin/
#   summary.json     — success_rate, num_success, mean_reward,
#                      mean_max_progress, mean_max_bottles_in_bin
#   world_*.mp4      — per-world rollout videos (with --save-video)

# 100 worlds stepped together in MJWarp physics (see the abc_sim README).
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --num-worlds 100 --parallel-worlds 100 \
    --randomization '{"bottle_count": 6, "randomize_variants": false, "randomize_scales": false}'
```

Any catalogue task is selected by name with `--task`. The task list, eval
flags, prompt defaults, and the released checkpoints' expected numbers are all
documented in the [abc_sim README](abc_sim/README.md#sim-eval).

## Fast DiT inference

With fast inference on (the default for `eval_policy.py` and `viz_policy.py`,
`--fast-inference` for deployment), the DiT policy casts to bf16 and compiles a
fused sampler (`abc_minimal/fast_dit.py`). At batch 1 the sampler is bound by
reading weights, and the stock Euler loop re-reads two sets of them on every
step although their inputs never change across steps:

- **adaLN modulations** depend only on (state, task, t), and the t schedule is
  fixed, so all 32 blocks' modulations for all 10 steps (plus the t = 0 row
  that RTC prefix positions use) come out of one GEMM per inference.
- **Cross-attention K/V** depend only on the vision tokens, so all 32 blocks'
  K/V come out of one GEMM per inference.

Each step then runs only the self-attention, the cross-attention query/output
projections and the MLP, about 40% fewer weight reads. The math is unchanged
(it matches the stock sampler to fp32 rounding), and one compiled step is
reused for every Euler step, so compiling is also shorter. Set
`DiTInferencePolicy.fused_sampler = False` to compile the stock sampler instead.

RTX 3090, `bottles_75k.pt`, one rendered sim observation (3 cameras at
168x224), RTC prefix of 4, 10 steps; mean `infer()` latency per action chunk,
numpy observation in and actions out:

| Configuration | Latency | Speedup | Max action diff vs fp32 | Compile + warmup |
| --- | ---: | ---: | ---: | ---: |
| Eager fp32 (`--no-fast-inference`) | 184.4 ms | 1.0x | — | — |
| Stock fast inference (bf16, compiled sampler) | 70.4 ms | 2.6x | 0.020 std | 832 s |
| **Fused fast inference (default)** | **49.0 ms** | **3.8x** | 0.019 std | 541 s |

Action differences are in units of the per-dimension action std; both bf16
paths are equally close to fp32. An NVIDIA L4 is about 2.5x slower than the
3090 here (408 ms eager, 184 ms stock fast), since it has about a third of the
memory bandwidth. Reproduce with:

```bash
uv run scripts/bench_inference.py --mode eager   # fp32 reference actions
uv run scripts/bench_inference.py --mode stock
uv run scripts/bench_inference.py --mode fused
```

The fused sampler reaches about half the 3090's memory bandwidth, so capturing
the whole sampler in one CUDA graph, and int8 weight-only quantization (which
changes the numerics and needs an accuracy check), are the next steps.

## Real-robot deployment

The deployment stack, including RTC, teleoperation, and recording, is
documented in [`deploy/README.md`](deploy/README.md). Install its optional
hardware dependencies with `uv sync --extra deploy`. Our five real-robot
[DAgger checkpoints](deploy/README.md#dagger-checkpoints) download with
`uv run prepare.py --dagger-checkpoint <name>`.

## Episode exports & training data format

The quickstart real-data download covers the bottles task; simulation data for
additional tasks is available through `prepare.py --sim-data-list`. To prepare
other real-data tasks, the ABC-130k MCAPs are hosted on Hugging Face at
[`XDOF/ABC-130k`](https://huggingface.co/datasets/XDOF/ABC-130k) (the dataset
is gated, so accept access on the dataset page and set `HF_TOKEN`). Download
all MCAPs for one task and convert them in place:

```bash
uv run scripts/export_hf_task.py --task organize_the_condiment_bottles
```

The episode format the trainer reads, the local MCAP converter, multi-node
sharded downloads, and subtask/operator conditioning are documented in the
[abc_minimal README](abc_minimal/README.md#training-data).

## Licenses

The project code is Apache-2.0 ([`LICENSE`](LICENSE)), with third-party
components covered by the licenses listed below. The published DiT checkpoints
`bottles_75k.pt`, `abc_dit_xl_200k_model.pt` and the DAgger checkpoints are
Apache-2.0. All of them embed a DINOv3-derived vision backbone, so the DINOv3 use
restrictions below apply to the weights as well as to the code that loads them.

The released VLA checkpoints (including `vla_abc130k_200000_v2.pt`, the
`abc130k` and `200k` families, and their task finetunes) contain Gemma-derived
weights and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms),
including its use and redistribution conditions. See [Gemma weights terms of
use](#gemma-weights-terms-of-use) below.

This repository includes and adapts code from the following third-party
projects. Original license files and copyright headers are retained in all
cases. Bundled license texts live under `assets/third_party/`.

| Project | License | License file | Inclusion | What we use/adapt |
| --- | --- | --- | --- | --- |
| [DINOv3](https://github.com/facebookresearch/dinov3) | DINOv3 License (Meta) | [`assets/third_party/dinov3/LICENSE.md`](assets/third_party/dinov3/LICENSE.md) | Adapted (`abc_minimal/dit.py`); pretrained weights downloaded by the user | ViT-B/16 vision backbone (`DinoRope`, `DinoAttention`, `DinoMlp`, etc.) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | MIT | [`assets/third_party/clip/LICENSE`](assets/third_party/clip/LICENSE) | Adapted (`abc_minimal/dit.py`); ViT-B/32 text weights + BPE vocab downloaded at runtime | CLIP text encoder + BPE tokenizer (`CLIPBPETokenizer`, `CLIPTextTower`, `CLIPTextEmbedder`) |
| [openpi](https://github.com/Physical-Intelligence/openpi) | Apache 2.0 | [`assets/third_party/openpi/LICENSE`](assets/third_party/openpi/LICENSE) | Adapted (`deploy/client/websocket_client_policy.py`, `deploy/client/msgpack_numpy.py`) | Websocket inference client skeleton + msgpack NumPy serialization |
| [msgpack-numpy](https://github.com/lebedov/msgpack-numpy) | BSD 3-Clause | [`assets/third_party/msgpack_numpy/LICENSE.md`](assets/third_party/msgpack_numpy/LICENSE.md) | Adapted (`deploy/client/msgpack_numpy.py`, via openpi) | NumPy serialization strategy for msgpack |
| [Gemma](https://ai.google.dev/gemma) | Apache 2.0 (code); Gemma Terms of Use (weights) | [`assets/third_party/gemma/LICENSE`](assets/third_party/gemma/LICENSE) | Adapted (`abc_minimal/gemma/`); base checkpoint supplied by the user | Gemma 3 model + SentencePiece tokenizer for ABC-VLA |
| [SigLIP](https://github.com/google-research/big_vision) | Apache 2.0 | [`assets/third_party/gemma/LICENSE`](assets/third_party/gemma/LICENSE) | Adapted (`abc_minimal/gemma/siglip_vision/`) | SigLIP vision encoder for ABC-VLA |

### Gemma weights terms of use

Gemma model code is Apache-2.0, but Gemma *weights* are additionally governed by
the Google Gemma Terms of Use (https://ai.google.dev/gemma/terms), including its
Prohibited Use Policy. Training from the base Gemma 3 checkpoint requires a
user-supplied file; the base checkpoint is not bundled in this repository.
The released VLA checkpoints downloaded by `prepare.py` contain Gemma-derived
weights, so those terms also apply when finetuning or evaluating them without
a separate base checkpoint. See `assets/third_party/gemma/NOTICE` and the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) for details.

### Simulator asset licenses

The simulator asset packages installed by `prepare.py --sim` (and the RoboCasa
packs fetched by `--sim-robocasa`) bundle meshes and textures from the sources
below. abc-side processing — recentering, rescaling, trimesh re-export, resized
textures, custom convex collision decompositions, and retuned MJCF wrappers —
does not change the upstream licenses. Per-object credits ship inside the
packages where noted.

| Source | License | Where it is used |
| --- | --- | --- |
| [RoboCasa](https://github.com/robocasa/robocasa) objaverse pack: objects curated from [Objaverse 1.0](https://objaverse.allenai.org/objaverse-1.0), originally by individual Sketchfab creators | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (RoboCasa asset release) | `mug`; the object libraries in `task_put_relative`, `task_grab_clutter`, and `task_conveyor_pick`; mug variants in `task_mug_flip` and `task_mug_tree`; plates in `task_dishrack`; bottles in `task_water_bottles`; objects in `task_multi_drawer_search` and `task_inhand_transfer`; the verbatim pack via `--sim-robocasa` |
| RoboCasa lightwheel pack: objects by [LightWheel AI](https://www.lightwheel.ai/) | CC BY 4.0 (RoboCasa asset release) | object variants in `task_put_relative`, `task_grab_clutter`, `task_conveyor_pick`, `task_inhand_transfer`, and `task_multi_drawer_search`; dish racks in `task_dishrack`; the verbatim pack via `--sim-robocasa` |
| RoboCasa aigen pack: AI-generated objects ([Luma.ai](https://lumalabs.ai/)) | CC BY 4.0 (RoboCasa asset release) | `bowl` |
| [Google Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects) | CC BY 4.0 | Two office objects in `task_multi_drawer_search`; per-object credits in the package's `OFFICE_ATTRIBUTIONS.md` |
| Sketchfab creators, individually credited | CC BY 4.0 | Six office objects in `task_multi_drawer_search` (see `OFFICE_ATTRIBUTIONS.md`); the box and crate in `task_bins` (see the package `README.md`); the Fujiya tin in `task_chess` by [Vision Fountain](https://sketchfab.com/visionfountain); [`bin`](https://sketchfab.com/3d-models/bin-8984db4f15284436ab704919327ca251) by [AlaPasta](https://sketchfab.com/alapasta); [`blocks`](https://sketchfab.com/3d-models/wooden-alphabet-blocks-5f8dfddbbc7d468784ca014378f7e5fe) by [Cherryvania](https://sketchfab.com/mikequeen123); [`dustpan`](https://sketchfab.com/3d-models/dustpan-d91eae20c02a4741aeb889246b417ae4) by [c_irby_paint](https://sketchfab.com/cirby2180); [`garbage_can`](https://sketchfab.com/3d-models/garbage-can-trashcan-bin-926826667ff04fb09a0907bbec54c766) by [BlackCube](https://sketchfab.com/blackcube4), including its copy in `task_water_bottles`; [`paper_ball`](https://sketchfab.com/3d-models/paper-ball-8afd2bfbe8c14fad937f768617d55f9e) by [ianshanewise](https://sketchfab.com/ianshanewise); [`tray`](https://sketchfab.com/3d-models/plastic-tray-e9b536258ae4499abec7b31ebd231daf) by [Aullwen](https://sketchfab.com/Aullwen) |
| [freepoly.org](https://freepoly.org/) | CC0 | The yellow tin in `task_chess` |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | MIT | The `short_cabinet` drawer fixture in `task_multi_drawer_search`; license copy in the package |
| [i2rt YAM](https://github.com/i2rt-robotics) | MIT | The `i2rt_yam` robot model package; license copy in the package |
| This project | Apache-2.0 | Everything else, modeled or scanned in-house: `ball_sorting_toy`, `brush`, `brush_flat`, `building_blocks`, `chess`, `cup_stacking`, `dishrack`, `drawer`, `flexible_gripper`, `hand_brush_smooth`, `jenga`, `letters`, `marker`, `mug_tree`, `new_brush`, `plate`, the generated `task_nuts_bolts` meshes, the conveyor fixture in `task_conveyor_pick`, the baked plate/rack originals in `task_dishrack`, `tin_2` in `task_chess`, and all task-tuned MJCF wrappers and collision meshes |

### DINOv3 use restrictions

The DINOv3 License prohibits use of the DINO Materials (including weights and
derivatives) for: military purposes; activities subject to ITAR or other
export-control regimes covering defense articles; nuclear applications;
espionage; and the development, manufacture, or use of weapons. Downstream
users who load DINOv3 weights through this codebase are bound by these
restrictions; see `assets/third_party/dinov3/LICENSE.md` for the full
license text.


## Citation

Please cite this work as

```
@misc{abc2026,
  title         = {Scalable Behavior Cloning with Open Data, Training, and Evaluation},
  author        = {Arthur Allshire and Himanshu Gaurav Singh and Ritvik Singh and Adam Rashid and Hongsuk Choi and David McAllister and Justin Yu and Yiyuan Chen and Huang Huang and Pieter Abbeel and Xi Chen and Rocky Duan and Phillip Isola and Jitendra Malik and Fred Shentu and Guanya Shi and Philipp Wu and Angjoo Kanazawa},
  year          = {2026},
  eprint        = {2606.27375},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2606.27375},
  url           = {https://arxiv.org/abs/2606.27375},
}
```
