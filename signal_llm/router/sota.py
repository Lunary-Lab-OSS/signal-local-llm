"""
SOTA Router Strategy

Implementation of the Single-Tower Scalar Regression router strategy.

Contract (remediation R01-R08):

- Checkpoint loading is strict (R03): a versioned schema distinguishes
  head-only and full state dicts; missing/unexpected/partial keys,
  non-finite tensors, and shape mismatches fail initialization and leave
  the strategy unloaded. A controller is published only after validation.
- Explicit checkpoint paths are honoured (R04): a configured relative path
  resolves against ``models_dir`` first; generated filename discovery
  applies only when no explicit path is configured.
- Devices parse through ``torch.device`` semantics (R05): ``cuda:1`` keeps
  its index instead of silently becoming CPU.
- INT4 quantization is decided before construction (R06): no post-hoc
  flag mutation, and a failed quantization leaves the backbone clean.
- Scores are validated at the boundary (R07): the controller returns a
  finite scalar in [0, 1] or raises.
"""

import logging
import os
from pathlib import Path
from typing import Any

import torch

from signal_llm.config import LLMConfig

from .base import RouterStrategy
from .model import SingleTowerStudent

logger = logging.getLogger(__name__)

_CHECKPOINT_SCHEMA_VERSION = 1


class SotaRouterController:
    """Controller for the SOTA router (RouteLLM-compatible surface)."""

    def __init__(self, model: SingleTowerStudent):
        self.model = model
        self.routers = {"sota": self}

    def calculate_strong_win_rate(self, prompt: str) -> float:
        """Return a validated difficulty score in [0, 1] (R07)."""
        self.model.eval()
        with torch.no_grad():
            score_logits = self.model([prompt])
            if score_logits.numel() != 1 and score_logits.shape != (1, 1):
                raise RuntimeError(
                    f"router produced unexpected output shape {tuple(score_logits.shape)}"
                )
            if not torch.isfinite(score_logits).all():
                raise RuntimeError("router produced non-finite logits")
            prob = torch.sigmoid(score_logits).reshape(-1)[0].item()
        if not (prob == prob) or prob in (float("inf"), float("-inf")):  # NaN/inf guard
            raise RuntimeError(f"router produced an invalid score: {prob!r}")
        if not 0.0 <= prob <= 1.0:  # pragma: no cover - sigmoid output range
            raise RuntimeError(f"router score outside [0, 1]: {prob!r}")
        return float(prob)


class SotaRouterStrategy(RouterStrategy):
    """SOTA router strategy implementation using SingleTowerStudent."""

    def __init__(self, config: LLMConfig, device: str, platform: str):
        super().__init__(config, device, platform)
        self.model: SingleTowerStudent | None = None
        self.models_dir: Path | None = None

    # ------------------------------------------------------------------ #
    # Device resolution (R05)
    # ------------------------------------------------------------------ #

    def _resolve_device(self) -> torch.device:
        requested = (self.device or "auto").strip().lower()
        if requested in ("", "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            parsed = torch.device(requested)
        except RuntimeError as exc:
            raise ValueError(f"invalid device {self.device!r}: {exc}") from exc
        if parsed.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(f"CUDA device {self.device!r} requested but unavailable")
            if parsed.index is not None and parsed.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {parsed.index} does not exist")
        if parsed.type == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            raise ValueError(f"MPS device {self.device!r} requested but unavailable")
        return parsed

    # ------------------------------------------------------------------ #
    # Checkpoint discovery (R04)
    # ------------------------------------------------------------------ #

    def _resolve_checkpoint_path(self, model_id: str, use_int4: bool) -> str | None:
        """Resolve the checkpoint location.

        An explicit configured path always wins and is resolved against
        ``models_dir`` when relative; generated filename discovery applies
        only when nothing was configured.
        """
        configured = self.config.routellm_checkpoint_path
        models_dir = Path(self.models_dir) if self.models_dir else None

        if configured:
            explicit = Path(configured)
            if not explicit.is_absolute() and models_dir is not None:
                candidate = models_dir / explicit
                if candidate.exists():
                    return str(candidate)
                raise FileNotFoundError(
                    f"configured routellm_checkpoint_path {configured!r} not found "
                    f"(resolved to {candidate})"
                )
            if explicit.exists():
                return str(explicit)
            raise FileNotFoundError(f"configured routellm_checkpoint_path {configured!r} not found")

        if models_dir is None:
            return None
        model_name = model_id.replace("/", "_").replace("-", "_")
        ordered = (
            [f"sota_router_{model_name}_int4.pt", f"sota_router_{model_name}_fp32.pt"]
            if use_int4
            else [f"sota_router_{model_name}_fp32.pt", f"sota_router_{model_name}_int4.pt"]
        )
        for filename in ordered:
            candidate = models_dir / filename
            if candidate.exists():
                return str(candidate)
        return None

    # ------------------------------------------------------------------ #
    # Checkpoint loading (R03)
    # ------------------------------------------------------------------ #

    def _load_checkpoint(self, checkpoint_path: str, device_obj: torch.device) -> None:
        assert self.model is not None
        artifact = torch.load(checkpoint_path, map_location=device_obj, weights_only=True)
        if not isinstance(artifact, dict) or set(artifact) != {
            "schema_version",
            "kind",
            "state_dict",
        }:
            raise RuntimeError("checkpoint requires schema_version, kind, and state_dict")
        if (
            type(artifact["schema_version"]) is not int
            or artifact["schema_version"] != _CHECKPOINT_SCHEMA_VERSION
        ):
            raise RuntimeError("unsupported checkpoint schema_version")
        kind = artifact["kind"]
        if kind not in ("head", "full"):
            raise RuntimeError("checkpoint kind must be head or full")
        state_dict = artifact["state_dict"]
        if not isinstance(state_dict, dict) or not all(isinstance(k, str) for k in state_dict):
            raise RuntimeError("checkpoint state_dict must have string keys")
        expected = self.model.state_dict()
        if kind == "head":
            expected = {
                k: v for k, v in expected.items() if k.startswith("head.") or k == "model_features"
            }
        if set(state_dict) != set(expected):
            raise RuntimeError("checkpoint has missing or unexpected state keys")
        self._validate_tensors_finite(state_dict)
        for key, value in state_dict.items():
            target = expected[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError(f"checkpoint tensor {key!r} has incompatible shape or dtype")
        if kind == "full":
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.head.load_state_dict(
                {
                    k.removeprefix("head."): v
                    for k, v in state_dict.items()
                    if k.startswith("head.")
                },
                strict=True,
            )
            with torch.no_grad():
                self.model.model_features.copy_(state_dict["model_features"])

    @staticmethod
    def _validate_tensors_finite(state_dict: dict) -> None:
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"checkpoint value {key!r} is not a tensor")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"checkpoint tensor {key!r} contains non-finite values")

    # ------------------------------------------------------------------ #
    # Controller construction
    # ------------------------------------------------------------------ #

    def load_controller(self) -> Any:
        self.controller = None
        self.model = None
        try:
            return self._build_controller()
        except Exception:
            self.controller = None
            self.model = None
            raise

    def _build_controller(self) -> Any:
        model_id = self.config.routellm_embedding_model or "google/embeddinggemma-300m"
        logger.info("Initializing SOTA Single-Tower Router with model: %s", model_id)

        device_obj = self._resolve_device()
        # R06: quantization is decided before construction, never mutated
        # after the fact.
        use_int4 = self.config.routellm_use_int4
        if use_int4 and device_obj.type != "cuda":
            raise ValueError("routellm_use_int4 requires a CUDA device")

        checkpoint_path = self._resolve_checkpoint_path(model_id, use_int4)
        if checkpoint_path is None:
            raise RuntimeError(
                "SOTA inference requires a trained checkpoint; construct SingleTowerStudent directly for training"
            )

        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        self.model = SingleTowerStudent(
            model_id=model_id,
            use_int4=use_int4,
            device=device_obj,
            dtype=self.config.routellm_embedding_dtype,
            hf_token=hf_token,
        )

        self._load_checkpoint(checkpoint_path, device_obj)

        self.controller = SotaRouterController(self.model)
        logger.info("✅ SOTA Router Controller initialized")
        return self.controller
