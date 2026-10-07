"""Serve deploy/adamo_server.py on a Modal GPU, like the RunPod pod in the README.

One container serves the FastAPI app at a public ``*.modal.run`` URL with the
same ``/status``, ``/start``, ``/stop`` and ``/take_control`` routes. It scales
to zero: the first request starts it (about a minute, then ``/status`` reports
``ready``), and it stops SCALEDOWN_S after the last request once no run is active
and the policy has loaded.
The checkpoint and the per-GPU AOT engine live on the ``abc-cache`` Modal
Volume, so only the first start on a GPU type builds the engine (7-13 min);
later starts load it in under a minute.

    # Every adamo 1.0.x on PyPI is yanked: point ADAMO_WHEEL at a wheel built
    # with the adamo repo's scripts/build-python-wheel.sh.
    export ADAMO_WHEEL=~/workspace/adamo/target/wheels/adamo-1.0.6-cp38-abi3-manylinux_2_38_x86_64.whl
    modal run deploy/modal_app.py::prepare     # once: checkpoint -> abc-cache volume
    modal deploy deploy/modal_app.py           # serve; bills only while a container is up
    modal app stop abc-adamo                   # take the endpoint down

Environment, read at deploy time: ABC_GPU (default L40S), ABC_REGION (default
us; empty for any region), ABC_PROXY_AUTH=1 to require Modal proxy auth tokens on
every request (the keepalive below can't send them, so runs then end
SCALEDOWN_S after the last request).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import modal

APP_NAME = "abc-adamo"
GPU = os.environ.get("ABC_GPU", "L40S")
REGION = os.environ.get("ABC_REGION", "us") or None
PROXY_AUTH = os.environ.get("ABC_PROXY_AUTH") == "1"
PORT = 8888
SCALEDOWN_S = 180
KEEPALIVE_S = 30

REPO = "/root/abc"
VENV = "/opt/venv"
CACHE = "/cache"
CHECKPOINT = f"{CACHE}/bottles_75k.pt"
ENGINE = f"{CACHE}/engines/{GPU}/engine.pt2"

volume = modal.Volume.from_name("abc-cache", create_if_missing=True)


def _adamo_wheel() -> Path | None:
    wheel = os.environ.get("ADAMO_WHEEL")
    if not wheel:
        return None
    path = Path(wheel).expanduser()
    if not path.is_file():
        raise SystemExit(f"ADAMO_WHEEL={wheel} does not exist; build it with the adamo repo's scripts/build-python-wheel.sh")
    return path


repo_root = Path(__file__).resolve().parent.parent
# Ubuntu 24.04 for glibc >= 2.38 (adamo's manylinux_2_38 wheel); the devel
# image carries the CUDA headers and compiler AOTInductor builds the engine with.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "libegl1", "libgl1", "gstreamer1.0-plugins-base", "build-essential")
    .pip_install("uv")
    .env({"UV_PROJECT_ENVIRONMENT": VENV, "UV_LINK_MODE": "copy", "UV_FIND_LINKS": "/wheels"})
)
if modal.is_local():
    wheel = _adamo_wheel()
    if wheel is not None:
        image = image.add_local_file(wheel, f"/wheels/{wheel.name}", copy=True)
    else:
        image = image.run_commands("mkdir -p /wheels")
# Dependencies first, from pyproject alone, so source edits don't reinstall torch.
image = (
    image.add_local_file(repo_root / "pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file(repo_root / "README.md", "/build/README.md", copy=True)
    .run_commands(f"cd /build && uv sync --extra adamo --no-install-project --python /usr/local/bin/python")
    .env({"PYTHONPATH": REPO, "ABC_CACHE": CACHE})
    .add_local_dir(
        repo_root,
        REPO,
        ignore=[".git", ".venv", "cache", "**/__pycache__", "*.pt", "*.pt2", "*.bin"],
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(volumes={CACHE: volume}, cpu=4, memory=16384, timeout=3600)
def prepare() -> None:
    """Download the checkpoint into the abc-cache volume (skipped if present)."""
    if Path(CHECKPOINT).exists():
        print(f"{CHECKPOINT} already present ({Path(CHECKPOINT).stat().st_size / 1e9:.1f} GB)")
        return
    subprocess.run([f"{VENV}/bin/python", "prepare.py", "--checkpoint"], cwd=REPO, check=True)
    volume.commit()


@app.function(
    gpu=GPU,
    region=REGION,
    volumes={CACHE: volume},
    cpu=8,
    memory=32768,
    # One stateful server. Modal sees a robot run (Zenoh traffic, no HTTP) as
    # idle, so _keepalive below holds the container up while one is active.
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
    timeout=24 * 3600,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(PORT, startup_timeout=1800, requires_proxy_auth=PROXY_AUTH)
def serve() -> None:
    server = (
        f"exec {VENV}/bin/python deploy/adamo_server.py"
        f" --policy.checkpoint-path={CHECKPOINT} --engine-path={ENGINE} --port {PORT}"
    )
    if not Path(CHECKPOINT).exists():
        # Recover from a skipped `prepare`: download first (several minutes),
        # then serve. /status answers once the download is done.
        print(f"{CHECKPOINT} missing; downloading it before serving")
        server = f"{VENV}/bin/python prepare.py --checkpoint && {server}"
    Path(ENGINE).parent.mkdir(parents=True, exist_ok=True)
    # gVisor may not expose the cgroup CPU quota the server sizes torch threads
    # from, so set it: half the 8 CPUs, leaving the rest for video decoding.
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    subprocess.Popen(["bash", "-c", server], cwd=REPO, env=env)
    threading.Thread(target=_keepalive, daemon=True).start()


def _keepalive() -> None:
    """Request our own public /status while the policy loads or a run is active.

    Modal counts only HTTP requests through its proxy as activity, so without
    this the container scales down SCALEDOWN_S after /start, mid-run, or while
    a first engine build is still compiling.
    """
    url = None
    while True:
        time.sleep(KEEPALIVE_S)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/status", timeout=10) as r:
                status = json.load(r)
        except Exception:
            continue  # still starting: Modal doesn't scale down before the port is up
        if status["policy"] not in ("loading", "compiling") and status["run"] is None:
            continue
        try:
            url = url or serve.get_web_url()
            with urllib.request.urlopen(f"{url}/status", timeout=30):
                pass
        except Exception as e:
            print(f"keepalive request failed ({e!r}); the container may scale down mid-run")
