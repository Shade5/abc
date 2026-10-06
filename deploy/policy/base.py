"""Shared adapter between an inference policy and the websocket server."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

MAX_ABS_ACTION = 6.3  # z-score normalized; past this is a decode failure


def resolve_device(requested: str) -> str:
    """Resolve ``"auto"`` to CUDA when it is available, otherwise CPU."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class DeployPolicy:
    """Expose an inference policy through the server contract.

    Subclasses set ``engine_cls`` to the abc_minimal.policy class they adapt.
    """

    engine_cls: type

    def __init__(self, config: Any, engine_path: Path | None = None):
        """``engine_path``: an AOT-compiled engine file for fast inference (see
        abc_minimal/aot_engine.py), loaded instead of compiling at startup."""
        self.config = config
        if config.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        checkpoint = Path(config.checkpoint_path).expanduser().resolve()
        device = resolve_device(config.device)
        if engine_path is not None and config.fast_inference:
            if not hasattr(self.engine_cls, "from_engine"):
                raise ValueError(
                    f"{self.engine_cls.__name__} has no compiled engine; drop --engine-path"
                )
            self._policy = self.engine_cls.from_engine(
                engine_path, checkpoint, config, device, compile_mode=config.fast_compile_mode
            )
        else:
            self._policy = self.engine_cls(checkpoint, config, device)
        if config.fast_inference and engine_path is None:
            self._policy.enable_fast_inference(
                config.fast_compile_mode,
                rtc_prefix_length=config.rtc_prefix_length,
            )
        self.chunk_len = config.model.chunk_length
        self.action_dim = config.model.action_dim

    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
    ) -> dict:
        if noise is None:
            noise = np.zeros((self.chunk_len, self.action_dim), dtype=np.float32)
        actions = self._policy.infer(
            obs,
            noise=noise,
            action_prefix=action_prefix,
            prefix_length=prefix_length,
        )
        max_action = float(np.abs(actions).max())
        if max_action > MAX_ABS_ACTION:
            warnings.warn(
                f"Action bounds exceeded: max |action|={max_action:.3f}",
                RuntimeWarning,
                stacklevel=2,
            )
        return {"actions": actions}
