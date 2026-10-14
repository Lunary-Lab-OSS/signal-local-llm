"""
SOTA Router Strategy

Implementation of the Single-Tower Scalar Regression router strategy.

Contract:

- Checkpoint loading is strict: a versioned schema distinguishes
  head-only and full state dicts; missing/unexpected/partial keys,
  non-finite tensors, and shape mismatches fail initialization and leave
  the strategy unloaded. Schema version 2 additionally requires an
  ``artifact_metadata`` block binding the checkpoint to its embedding
  model id/revision, dimensions, normalization policy, feature version,
  and model-pair values; the recorded model-pair values must equal the
  checkpoint's ``model_features`` tensor. A controller is published only
  after validation.
- Explicit checkpoint paths are honoured: a relative path resolves against
  ``models_dir`` when one is set (otherwise against the working directory);
  generated filename discovery applies only when no explicit path is
  configured.
- Devices parse through ``torch.device`` semantics: ``cuda:1`` keeps its
  index instead of silently becoming CPU.
- INT4 quantization is decided before construction: no post-hoc flag
  mutation, and a failed quantization leaves the backbone clean.
- Scores are validated at the boundary: the controller returns a finite
  scalar in [0, 1] or raises.
"""

import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

from signal_llm.config import LLMConfig

from .base import RouterStrategy
from .model import SingleTowerStudent

logger = logging.getLogger(__name__)

_CHECKPOINT_SCHEMA_VERSION = 2
_CHECKPOINT_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "kind", "state_dict", "artifact_metadata"}
)
_MODEL_PAIR_TOLERANCE = 1e-6


class SotaRouterController:
    """Controller for the SOTA router (RouteLLM-compatible surface)."""

    def __init__(self, model: SingleTowerStudent):
        self.model = model
        self.routers = {"sota": self}

    def calculate_strong_win_rate(self, prompt: str) -> float:
        """Return a validated difficulty score in [0, 1].

        Non-finite logits are rejected before the sigmoid, so the returned
        probability is always a finite value in [0, 1].
        """
        self.model.eval()
        with torch.no_grad():
            score_logits = self.model([prompt])
            if score_logits.numel() != 1:
                raise RuntimeError(
                    f"router produced unexpected output shape {tuple(score_logits.shape)}"
                )
            if not torch.isfinite(score_logits).all():
                raise RuntimeError("router produced non-finite logits")
            prob = torch.sigmoid(score_logits.float()).reshape(-1)[0].item()
        return float(prob)


class SotaRouterStrategy(RouterStrategy):
    """SOTA router strategy implementation using SingleTowerStudent."""

    def __init__(self, config: LLMConfig, device: str, platform: str):
        super().__init__(config, device, platform)
        self.model: SingleTowerStudent | None = None
        self.models_dir: Path | None = None

    # ------------------------------------------------------------------ #
    # Device resolution
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
    # Checkpoint discovery
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
    # Artifact metadata (schema v2)
    # ------------------------------------------------------------------ #

    def _expected_artifact_metadata(
        self, student: SingleTowerStudent, model_pair_values: torch.Tensor
    ) -> dict[str, Any]:
        """Build the metadata contract for the constructed student.

        ``model_pair_values`` is the checkpoint's own ``model_features``
        tensor, so the metadata block must agree with the weights it ships
        with rather than with the freshly initialised (all-zero) buffer.
        """
        model_id = self.config.routellm_embedding_model or "google/embeddinggemma-300m"
        return {
            "schema_version": 2,
            "embedding_model_id": model_id,
            "embedding_revision": getattr(self.config, "routellm_embedding_revision", None),
            "embedding_dimension": int(student.head.input_dim) - 3 - 2,
            "normalize_embeddings": bool(student.normalize_embeddings),
            "linguistic_feature_version": 1,
            "architecture": "single-tower-scalar-regression-v2",
            "model_pair_values": [float(v) for v in model_pair_values.reshape(-1).tolist()],
        }

    @staticmethod
    def _validate_artifact_metadata(actual: Any, expected: dict[str, Any]) -> None:
        """Reject a checkpoint whose training identity does not match this build."""
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(
                "checkpoint artifact_metadata must contain exactly: " + ", ".join(sorted(expected))
            )
        for key, wanted in expected.items():
            got = actual[key]
            if isinstance(wanted, list):
                matches = (
                    isinstance(got, list)
                    and len(got) == len(wanted)
                    and all(
                        isinstance(g, (int, float))
                        and not isinstance(g, bool)
                        and math.isfinite(g)
                        and abs(float(g) - w) <= _MODEL_PAIR_TOLERANCE
                        for g, w in zip(got, wanted, strict=True)
                    )
                )
            else:
                # Exact type match: ``True == 1`` and ``2.0 == 2`` must not pass.
                matches = type(got) is type(wanted) and got == wanted
            if not matches:
                raise RuntimeError(
                    f"checkpoint artifact_metadata[{key!r}] = {got!r} does not match "
                    f"the deployed embedding/model-pair identity (expected {wanted!r})"
                )

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #

    def _load_checkpoint(self, checkpoint_path: str, device_obj: torch.device) -> None:
        if self.model is None:
            raise RuntimeError("model must be constructed before loading a checkpoint")
        artifact = torch.load(checkpoint_path, map_location=device_obj, weights_only=True)
        if not isinstance(artifact, dict) or set(artifact) != _CHECKPOINT_TOP_LEVEL_KEYS:
            raise RuntimeError(
                "checkpoint requires schema_version, kind, state_dict, and artifact_metadata"
            )
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
        self._validate_artifact_metadata(
            artifact["artifact_metadata"],
            self._expected_artifact_metadata(self.model, state_dict["model_features"]),
        )
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
        # Quantization is decided before construction, never mutated after
        # the fact.
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
