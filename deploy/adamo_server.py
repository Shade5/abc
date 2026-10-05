"""Run ABC-DiT against a robot streamed over Adamo (https://docs.adamohq.com).

A FastAPI server loads and compiles the policy once at startup. ``POST /start``
with an Adamo API key and robot name then:

- subscribes to the robot's camera tracks and decodes them (Adamo's native
  H.264 decoder) into a latest-frame slot per camera,
- subscribes to the robot's joint state: ``{robot}/state/joints`` from a robot
  on adamo 1.0, ``{robot}/proprioception/joints`` from an older one,
- claims the robot the way operate.adamohq.com does (``take_control``:
  acquired, then a heartbeat a second on ``{robot}/control/json/operator_control``),
- runs an RTC inference loop on the latest observation and publishes one
  action per control tick on the robot's control topic.

If another operator acquires the robot the policy stops sending actions until
``POST /take_control``. ``POST /stop`` releases the robot and disconnects;
``GET /status`` reports the cameras, state, inference and who holds control.

    uv run deploy/adamo_server.py --policy.checkpoint-path=cache/bottles_75k.pt
    curl -X POST localhost:8000/start -H 'content-type: application/json' \
        -d '{"adamo_api_key": "ak_...", "robot_name": "abc-sim"}'
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import adamo
import numpy as np
import torch
import tyro
import uvicorn
from adamo import JointState, Priority
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from abc_minimal.policy import InferenceConfig, shared_inference_fields
from deploy.policy import Policy, PolicyConfig

logger = logging.getLogger("adamo_server")

JOINT_NAMES = [
    *(f"left_joint_{i}" for i in range(6)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(6)),
    "right_gripper",
]
# A claim on the robot expires without a heartbeat for this long (Adamo's
# recommended lease); the policy heartbeats every HEARTBEAT_S.
CONTROL_LEASE_S = 3.0
HEARTBEAT_S = 1.0


@dataclass
class Args:
    policy: InferenceConfig = field(
        default_factory=lambda: InferenceConfig(
            checkpoint_path="cache/bottles_75k.pt",
            prompt="sim put the plastic bottles in the bin",
            fast_inference=True,
            rtc_prefix_length=4,
        )
    )
    host: str = "0.0.0.0"
    port: int = 8000
    engine_path: str | None = None
    """Compiled-policy cache (torch.compiler cache artifacts) for this GPU, e.g.
    /workspace-global/abc/3090/engine.bin. Loaded before compiling so startup
    skips the compile; written after a compile when it is missing."""


def cpu_quota() -> float | None:
    """CPUs this container may use (cgroup quota), or None when unlimited."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()  # cgroup v2
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def cap_torch_threads() -> int | None:
    """Size torch's CPU thread pool to half the container's CPU quota, leaving
    the rest for the camera decoders, unless OMP_NUM_THREADS says otherwise.

    torch sizes the pool from the host's core count: on a RunPod 4090 that is
    120 threads in a 12.75-CPU container, which throttles every thread, takes
    each inference from ~50 ms to ~2.6 s, and starves video decoding.
    """
    if "OMP_NUM_THREADS" in os.environ:
        return None
    quota = cpu_quota()
    if quota is None:
        return None
    threads = max(1, int(quota) // 2)
    torch.set_num_threads(threads)
    return threads


class StartRequest(BaseModel):
    adamo_api_key: str | None = None
    """Defaults to the ADAMO_API_KEY environment variable."""
    robot_name: str
    prompt: str | None = None
    """Overrides the server's default prompt for this run."""
    head: str = "head"
    """Adamo video track fed to the policy's ``top`` camera."""
    wrist_left: str = "wrist_left"
    """Adamo video track fed to the policy's ``left`` wrist camera."""
    wrist_right: str = "wrist_right"
    """Adamo video track fed to the policy's ``right`` wrist camera."""
    head_stereo: bool = True
    """The head track is a side-by-side stereo pair; the left view is used."""
    state_topic: str | None = None
    """Joint state key. None follows both ``{robot}/state/joints`` (JSON
    ``{"positions": [14]}``, robots on adamo 1.0) and
    ``{robot}/proprioception/joints`` (binary, older robots)."""
    control_topic: str = "{robot}/control/joint_state"
    control_hz: float = 30.0
    execute_steps: int = 16
    """Actions executed from each chunk before the next one takes over."""
    inference_lead_steps: int = 7
    """Start the next inference this many steps before ``execute_steps``."""
    prefix_length: int = 4
    """RTC prefix: not-yet-executed actions the next chunk is conditioned on."""
    frame_timeout_s: float | None = 30.0
    """Stop the run (as ``POST /stop``) when a camera has sent no frame for this
    long, counted from ``/start`` until its first frame. None disables it."""

    @property
    def cameras(self) -> dict[str, str]:
        """Policy camera key -> Adamo video track name on the robot."""
        return {"top": self.head, "left": self.wrist_left, "right": self.wrist_right}


class LatestSlot:
    """Thread-safe latest value with arrival time and count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.value = None
        self.stamp = 0.0
        self.count = 0

    def set(self, value) -> None:
        with self._lock:
            self.value = value
            self.stamp = time.monotonic()
            self.count += 1

    def get(self):
        with self._lock:
            return self.value, self.stamp


STATE_TOPICS = ("{robot}/state/joints", "{robot}/proprioception/joints")


def decode_joints(payload: bytes) -> np.ndarray:
    """JSON ``{"positions": [...]}`` (a robot's state store on adamo 1.0), or
    ``[ts_us: u64 BE][f32 BE] * N`` (the adamo 0.4 joints wire format)."""
    if payload[:1] == b"{":
        return np.asarray(json.loads(payload)["positions"], dtype=np.float32)
    n = (len(payload) - 8) // 4
    return np.asarray(struct.unpack_from(f">{n}f", payload, 8), dtype=np.float32)


class AdamoRun:
    """One robot connection: camera decoders, state, inference and control."""

    def __init__(self, req: StartRequest, policy: Policy, prompt: str) -> None:
        self.req = req
        self.policy = policy
        self.prompt = prompt
        self.robot = req.robot_name
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.started = time.monotonic()

        api_key = req.adamo_api_key or os.environ.get("ADAMO_API_KEY")
        if not api_key:
            raise ValueError("pass adamo_api_key or set ADAMO_API_KEY")
        self.session = adamo.connect(api_key=api_key)
        self.frames = {cam: LatestSlot() for cam in req.cameras}
        self.state = LatestSlot()
        self.state_key: str | None = None  # where the latest joint state came from
        self.decode_ms = {cam: 0.0 for cam in req.cameras}

        self._receivers = {
            cam: self.session.video_receiver(self.robot, track, max_queued_frames=1)
            for cam, track in req.cameras.items()
        }
        state_topics = (req.state_topic,) if req.state_topic else STATE_TOPICS
        self._state_subs = [
            self.session.subscribe(
                topic.format(robot=self.robot), callback=self._on_state
            )
            for topic in state_topics
        ]
        self._control_pub = self.session.publisher(
            req.control_topic.format(robot=self.robot),
            priority=Priority.REAL_TIME,
            express=True,
        )

        # Operator ownership, as operate.adamohq.com claims a robot: acquired,
        # then a heartbeat every second, then released, on
        # {robot}/control/json/operator_control.
        self.operator = {
            "userId": "abc-dit-policy",
            "sessionId": uuid.uuid4().hex,
            "displayName": "ABC-DiT policy",
        }
        self.in_control = False
        self.controller: dict | None = None  # the other operator holding the robot
        self._controller_seen = 0.0
        ownership_topic = f"{self.robot}/control/json/operator_control"
        self._ownership_pub = self.session.publisher(
            ownership_topic, priority=Priority.DATA_HIGH, reliable=True
        )
        self._ownership_sub = self.session.subscribe(
            ownership_topic, callback=self._on_ownership
        )

        # Shared between the inference and control threads. Ownership changes
        # take the lock too, so no action is published after control is lost.
        self._chunk_lock = threading.Lock()
        self._generation = 0  # bumped on every ownership change
        self._chunk: np.ndarray | None = None
        self._chunk_start = 0  # control step that chunk[0] is executed at
        self._step = 0  # control steps published so far
        self.infer_ms: list[float] = []
        self.published = 0

        self._threads = [
            threading.Thread(target=self._camera_loop, args=(cam,), daemon=True)
            for cam in req.cameras
        ]
        self._threads.append(threading.Thread(target=self._control_loop, daemon=True))
        self._threads.append(threading.Thread(target=self._heartbeat_loop, daemon=True))
        for t in self._threads:
            t.start()

    # -- cameras ---------------------------------------------------------------

    def _camera_loop(self, cam: str) -> None:
        while not self.stop_event.is_set():
            rx = self._receivers[cam]
            try:
                frame = rx.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception as e:
                # A decode error resets on its own, but a connection error ends
                # the receiver, so reopen it either way.
                logger.exception("camera %s receive failed; reopening", cam)
                self.error = f"camera {cam}: {e}"
                if self.stop_event.wait(0.5):
                    break
                self._reopen_receiver(cam)
                continue
            t0 = time.perf_counter()
            rgba = np.asarray(frame.rgba)
            if cam == "top" and self.req.head_stereo:
                rgba = rgba[:, : rgba.shape[1] // 2]
            # RGBA (H, W, 4) -> RGB (3, H, W), as the policy expects.
            rgb = rgba[..., :3].transpose(2, 0, 1).copy()
            self.frames[cam].set(rgb)
            self.decode_ms[cam] = (time.perf_counter() - t0) * 1e3

    def _on_state(self, sample) -> None:
        try:
            joints = decode_joints(bytes(sample.payload))
        except (ValueError, KeyError, struct.error) as e:
            self.error = f"joint state on {sample.key}: {e!r}"
            return
        self.state_key = sample.key
        self.state.set(joints)

    def _reopen_receiver(self, cam: str) -> None:
        try:
            self._receivers[cam].close()
        except Exception:
            logger.exception("camera %s close failed", cam)
        try:
            self._receivers[cam] = self.session.video_receiver(
                self.robot, self.req.cameras[cam], max_queued_frames=1
            )
        except Exception as e:
            logger.exception("camera %s reopen failed", cam)
            self.error = f"camera {cam}: reopen failed: {e}"

    def latest_obs(self) -> dict | None:
        state, _ = self.state.get()
        if state is None:
            return None
        images = {}
        for cam, slot in self.frames.items():
            img, _ = slot.get()
            if img is None:
                return None
            images[cam] = img
        return {"state": state, "images": images, "prompt": self.prompt}

    def frame_timeout(self) -> str | None:
        """Why the run timed out waiting for camera frames, or None."""
        timeout = self.req.frame_timeout_s
        if timeout is None:
            return None
        now = time.monotonic()
        for cam, slot in self.frames.items():
            age = now - (slot.stamp if slot.count else self.started)
            if age > timeout:
                return f"no frames from the {cam} camera for {age:.0f} s"
        return None

    # -- operator control --------------------------------------------------------

    def _publish_ownership(self, event: str, active: bool) -> None:
        msg = {
            "version": 1,
            "active": active,
            "event": event,
            "operator": self.operator,
            "timestampMs": self.session.fabric_now_us() / 1000,
        }
        self._ownership_pub.put(json.dumps(msg).encode())

    def _on_ownership(self, sample) -> None:
        try:
            msg = json.loads(sample.payload)
        except ValueError:
            return
        operator = msg.get("operator") or {}
        if operator.get("sessionId") == self.operator["sessionId"]:
            return
        if msg.get("active"):
            if self.in_control and msg.get("event") == "acquired":
                logger.warning("control taken by %s", operator.get("displayName"))
                self._set_in_control(False)
            self.controller, self._controller_seen = operator, time.monotonic()
        elif self.controller and operator.get("sessionId") == self.controller.get(
            "sessionId"
        ):
            self.controller = None

    def take_control(self) -> None:
        """Claim the robot and start sending actions from a fresh chunk."""
        self._publish_ownership("acquired", active=True)
        self._set_in_control(True)
        self.controller = None

    def release_control(self) -> None:
        """Stop sending actions and give up the claim."""
        if not self.in_control:
            return
        self._set_in_control(False)
        self._publish_ownership("released", active=False)

    def _set_in_control(self, in_control: bool) -> None:
        """Change ownership and drop the chunk; inference started under the
        previous generation is discarded even if control has come back since."""
        with self._chunk_lock:
            self.in_control = in_control
            self._chunk = None
            self._generation += 1

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(HEARTBEAT_S):
            if self.in_control:
                self._publish_ownership("heartbeat", active=True)
            if (
                self.controller
                and time.monotonic() - self._controller_seen > CONTROL_LEASE_S
            ):
                self.controller = None  # its lease lapsed without a release

    # -- inference (runs on the policy thread) ---------------------------------

    def inference_step(self) -> bool:
        """Run one inference if a new chunk is due. Returns False if idle."""
        req = self.req
        if not self.in_control:
            return False
        with self._chunk_lock:
            chunk, start, step = self._chunk, self._chunk_start, self._step
            generation = self._generation
        if (
            chunk is not None
            and step - start < req.execute_steps - req.inference_lead_steps
        ):
            return False
        obs = self.latest_obs()
        if obs is None:
            return False

        prefix = None
        if chunk is not None:
            offset = step - start
            prefix = chunk[offset : offset + req.prefix_length]
            if len(prefix) < req.prefix_length:
                prefix = None
        t0 = time.perf_counter()
        if prefix is None:
            actions = self.policy.infer(obs)["actions"]
        else:
            actions = self.policy.infer(
                obs, action_prefix=prefix, prefix_length=req.prefix_length
            )["actions"]
        self.infer_ms.append((time.perf_counter() - t0) * 1e3)
        self.infer_ms = self.infer_ms[-100:]

        with self._chunk_lock:
            if self._generation != generation or self._chunk is not chunk:
                return True  # control changed hands mid-inference; drop it
            # The new chunk starts at the step inference was launched from; any
            # steps executed meanwhile came from the prefix it was conditioned on.
            self._chunk = actions
            self._chunk_start = step if chunk is not None else self._step
        return True

    # -- control ---------------------------------------------------------------

    def _control_loop(self) -> None:
        period = 1.0 / self.req.control_hz
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            next_tick += period
            with self._chunk_lock:
                chunk, start = self._chunk, self._chunk_start
                if chunk is not None and self.in_control:
                    idx = min(self._step - start, len(chunk) - 1)
                    msg = JointState(names=JOINT_NAMES, positions=chunk[idx].tolist())
                    self._step += 1
                    # Under the lock, so a release can't land between the
                    # ownership check and the put.
                    self._control_pub.put(msg.to_json())
                    self.published += 1
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def status(self) -> dict:
        now = time.monotonic()
        state, state_t = self.state.get()
        return {
            "robot": self.robot,
            "error": self.error,
            "control": {
                "in_control": self.in_control,
                "operator": self.operator,
                "held_by": self.controller,
            },
            "cameras": {
                cam: {
                    "frames": slot.count,
                    "age_ms": round((now - slot.stamp) * 1e3) if slot.count else None,
                    "shape": None if slot.value is None else list(slot.value.shape),
                    "decode_ms": round(self.decode_ms[cam], 2),
                }
                for cam, slot in self.frames.items()
            },
            "state": {
                "key": self.state_key,
                "msgs": self.state.count,
                "age_ms": round((now - state_t) * 1e3) if self.state.count else None,
                "dim": None if state is None else len(state),
            },
            "inference": {
                "chunks": len(self.infer_ms),
                "last_ms": round(self.infer_ms[-1], 1) if self.infer_ms else None,
                "mean_ms": round(float(np.mean(self.infer_ms)), 1)
                if self.infer_ms
                else None,
            },
            "actions_published": self.published,
        }

    def close(self) -> None:
        try:
            self.release_control()
        except Exception:
            logger.exception("release failed")
        self.stop_event.set()
        for t in self._threads:
            t.join(timeout=2)
        for closable in (
            *self._state_subs,
            self._ownership_sub,
            self._control_pub,
            self._ownership_pub,
            self.session,
        ):
            try:
                closable.close()
            except Exception:
                logger.exception("close failed")


class PolicyWorker:
    """Owns the policy on one thread: load, compile, then serve the active run.

    torch.compile's CUDA graphs are recorded per thread, so warmup and every
    later inference happen on this same thread.
    """

    def __init__(self, config: PolicyConfig, engine_path: str | None = None) -> None:
        self.config = config
        self.engine_path = engine_path
        self.policy: Policy | None = None
        self.phase = "loading"
        self.error: str | None = None
        self.run: AdamoRun | None = None
        self.last_stop: str | None = None  # why the last run stopped on its own
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()

    def _main(self) -> None:
        cap_torch_threads()  # per thread: this one runs every inference
        try:
            t0 = time.perf_counter()
            self.phase = "compiling" if self.config.fast_inference else "loading"
            loaded = self._load_engine()
            self.policy = Policy(self.config)
            logger.warning("policy ready in %.0f s", time.perf_counter() - t0)
            if not loaded:
                self._save_engine()
            self.phase = "ready"
        except Exception as e:
            logger.exception("policy load failed")
            self.phase, self.error = "failed", repr(e)
            return
        while True:
            run = self.run
            if run is None or run.stop_event.is_set():
                time.sleep(0.005)
                continue
            reason = run.frame_timeout()
            if reason is not None:
                self._stop_timed_out(run, reason)
                continue
            try:
                if not run.inference_step():
                    time.sleep(0.002)
            except Exception as e:
                logger.exception("inference failed")
                run.error = f"inference: {e!r}"
                time.sleep(0.5)

    def _stop_timed_out(self, run: AdamoRun, reason: str) -> None:
        logger.warning("stopping %s: %s", run.robot, reason)
        with self._lock:
            if self.run is run:  # not already stopped or replaced by /start
                self.last_stop = reason
                self.stop()

    def _load_engine(self) -> bool:
        if not (self.engine_path and os.path.exists(self.engine_path)):
            return False
        with open(self.engine_path, "rb") as f:
            torch.compiler.load_cache_artifacts(f.read())
        logger.warning("loaded compiled engine from %s", self.engine_path)
        return True

    def _save_engine(self) -> None:
        if not (self.engine_path and self.config.fast_inference):
            return
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            logger.warning("no compile artifacts to save")
            return
        os.makedirs(os.path.dirname(self.engine_path) or ".", exist_ok=True)
        # Written in place: global volumes don't support atomic rename.
        with open(self.engine_path, "wb") as f:
            f.write(artifacts[0])
        logger.warning(
            "saved compiled engine to %s (%.0f MB)",
            self.engine_path,
            len(artifacts[0]) / 1e6,
        )

    def start(self, req: StartRequest) -> AdamoRun:
        if self.policy is None:
            raise HTTPException(503, f"policy not ready ({self.phase})")
        with self._lock:
            self.stop()
            self.last_stop = None
            self.run = AdamoRun(req, self.policy, req.prompt or self.config.prompt)
            self.run.take_control()
            return self.run

    def stop(self) -> None:
        with self._lock:
            run, self.run = self.run, None
            if run is not None:
                run.close()


def create_app(args: Args) -> FastAPI:
    config = PolicyConfig(**shared_inference_fields(args.policy))
    worker: PolicyWorker | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal worker
        worker = PolicyWorker(config, args.engine_path)
        yield
        worker.stop()

    app = FastAPI(title="ABC-DiT Adamo server", lifespan=lifespan)

    @app.post("/start")
    def start(req: StartRequest) -> dict:
        try:
            run = worker.start(req)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"could not start: {e!r}") from e
        return {"started": True, "robot": run.robot}

    @app.post("/take_control")
    def take_control() -> dict:
        """Reclaim the robot for the active run, e.g. after an operator took it."""
        run = worker.run
        if run is None:
            raise HTTPException(409, "no active run; POST /start first")
        run.take_control()
        return {"in_control": True, "robot": run.robot}

    @app.post("/stop")
    def stop() -> dict:
        """Release control and disconnect from the robot."""
        worker.stop()
        return {"stopped": True}

    @app.get("/status")
    def status() -> dict:
        """Policy phase, plus camera, state, inference and control counters."""
        run = worker.run
        return {
            "policy": worker.phase,
            "policy_error": worker.error,
            "last_stop": worker.last_stop,
            "run": None if run is None else run.status(),
        }

    @app.get("/topics")
    def topics(seconds: float = 3.0) -> dict:
        """Keys the active robot publishes on, sampled for ``seconds``."""
        run = worker.run
        if run is None:
            raise HTTPException(409, "no active run; POST /start first")
        seen: dict[str, dict] = {}

        def on_sample(s) -> None:
            entry = seen.setdefault(s.key, {"count": 0, "bytes": 0})
            entry["count"] += 1
            entry["bytes"] = len(s.payload)
            if "sample" not in entry:
                try:
                    entry["sample"] = json.loads(s.payload)
                except ValueError:
                    entry["sample"] = s.payload[:32].hex()

        sub = run.session.subscribe(f"{run.robot}/**", callback=on_sample)
        time.sleep(seconds)
        sub.close()
        return dict(sorted(seen.items()))

    return app


def main(args: Args) -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    threads = cap_torch_threads()
    if threads is not None:
        logger.warning(
            "torch CPU threads: %d (half of a %.2f-CPU quota)", threads, cpu_quota()
        )
    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
