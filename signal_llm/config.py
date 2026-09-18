"""
LLM Configuration Dataclasses

Every public field is validated at construction (L17): token counts must be
non-negative ints (not bools), sampling values finite, bounds ordered, and
enabled roles must have non-empty priority lists. Unknown backends and
router types are rejected immediately.
"""

import math
import re
from dataclasses import dataclass

_VALID_BACKENDS = ("exllamav2", "vllm", "transformers", "mlx", "auto")
_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")
_VALID_ROUTER_TYPES = ("llm", "routellm", "coreml")
_VALID_DTYPES = ("float16", "bfloat16", "float32")


def validate_device(device: str | None) -> tuple[str, int | None]:
    if device is not None and not isinstance(device, str):
        raise ValueError("device must be a string or None")
    requested = (device or "auto").strip().lower() or "auto"
    if requested in _VALID_DEVICES:
        return requested, None
    if re.fullmatch(r"cuda:[0-9]+", requested):
        return "cuda", int(requested.split(":")[1])
    raise ValueError(
        f"unsupported device {device!r}; expected auto, cpu, mps, cuda or cuda:<index>"
    )


def validate_remote_code(trust: bool, revision: str | None) -> None:
    if not isinstance(trust, bool):
        raise ValueError("trust_remote_code must be a bool")
    if trust and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)):
        raise ValueError(
            "trust_remote_code=True requires a pinned revision (immutable 40-digit SHA)"
        )


def _validate_token_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


def _validate_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _validate_probability(name: str, value: object) -> float:
    result = _validate_finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0], got {result!r}")
    return result


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    repo_id: str
    local_path: str | None = None
    quantized: bool = False
    device: str = "auto"
    dtype: str = "float16"
    thinking_tokens: int = 0
    revision: str | None = None
    backend: str = "exllamav2"
    #: Opt-in for executing custom tokenizer/model code from the repo.
    #: Disabled by default; requires a pinned revision for a meaningful
    #: trust decision (L03).
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ModelConfig.name must be a non-empty string")
        if not self.repo_id or not isinstance(self.repo_id, str):
            raise ValueError("ModelConfig.repo_id must be a non-empty string")
        _validate_token_count("ModelConfig.thinking_tokens", self.thinking_tokens)
        backend = (self.backend or "").strip().lower()
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"ModelConfig.backend must be one of {sorted(_VALID_BACKENDS)}, "
                f"got {self.backend!r}"
            )
        self.backend = backend
        kind, index = validate_device(self.device)
        self.device = f"{kind}:{index}" if index is not None else kind
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"ModelConfig.dtype must be one of {sorted(_VALID_DTYPES)}, got {self.dtype!r}"
            )
        validate_remote_code(self.trust_remote_code, self.revision)


@dataclass
class SpeculativeDecodingConfig:
    """Speculative Decoding Configuration."""

    enabled: bool = True
    num_speculative_tokens: int = 5

    def __post_init__(self) -> None:
        _validate_token_count(
            "SpeculativeDecodingConfig.num_speculative_tokens", self.num_speculative_tokens
        )


@dataclass
class LLMConfig:
    """LLM Configuration."""

    router_priority: list[ModelConfig]
    semantic_priority: list[ModelConfig]
    agent_priority: list[ModelConfig]
    semantic_enabled: bool = True
    agent_enabled: bool = True
    max_tokens: int = 2048
    max_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    router_thinking_tokens: int = 0
    semantic_thinking_tokens: int = 0
    agent_thinking_tokens: int = 512
    speculative_decoding: SpeculativeDecodingConfig | None = None
    router_type: str = "llm"
    routellm_router_name: str = "mf"
    routellm_threshold: float = 0.5
    routellm_embedding_model: str = "nomic-ai/modernbert-embed-base"
    routellm_embedding_device: str = "cuda"
    routellm_embedding_dtype: str = "bfloat16"
    routellm_embedding_compile: bool = True
    routellm_checkpoint_path: str | None = None
    routellm_text_dim: int | None = None
    models_dir: str | None = None
    cache_dir: str | None = None

    routellm_use_int4: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.routellm_use_int4, bool):
            raise ValueError("LLMConfig.routellm_use_int4 must be a bool")
        for role in ("router_priority", "semantic_priority", "agent_priority"):
            value = getattr(self, role)
            if not isinstance(value, list):
                raise ValueError(f"LLMConfig.{role} must be a list of ModelConfig")
            for item in value:
                if not isinstance(item, ModelConfig):
                    raise ValueError(
                        f"LLMConfig.{role} entries must be ModelConfig, got {type(item).__name__}"
                    )

        # Enabled roles must be able to load something (L17).
        if self.semantic_enabled and not self.semantic_priority:
            raise ValueError("semantic_enabled=True requires a non-empty semantic_priority")
        if self.agent_enabled and not self.agent_priority:
            raise ValueError("agent_enabled=True requires a non-empty agent_priority")

        _validate_token_count("LLMConfig.max_tokens", self.max_tokens)
        if self.max_new_tokens is not None:
            _validate_token_count("LLMConfig.max_new_tokens", self.max_new_tokens)
        for name in ("router_thinking_tokens", "semantic_thinking_tokens", "agent_thinking_tokens"):
            _validate_token_count(f"LLMConfig.{name}", getattr(self, name))
        if self.max_new_tokens is not None and self.max_new_tokens > self.max_tokens:
            raise ValueError(
                f"max_new_tokens ({self.max_new_tokens}) must not exceed max_tokens "
                f"({self.max_tokens}) — the total cap is a hard limit (L11)"
            )
        _validate_probability("LLMConfig.temperature", self.temperature)
        _validate_probability("LLMConfig.top_p", self.top_p)

        router_type = (self.router_type or "").strip().lower()
        if router_type not in _VALID_ROUTER_TYPES:
            raise ValueError(
                f"LLMConfig.router_type must be one of {sorted(_VALID_ROUTER_TYPES)}, "
                f"got {self.router_type!r}"
            )
        self.router_type = router_type
        # R13: canonicalise the router name exactly once, here, so the
        # factory and every downstream strategy read one value.
        self.routellm_router_name = (self.routellm_router_name or "mf").strip().lower()
        _validate_probability("LLMConfig.routellm_threshold", self.routellm_threshold)

        if self.routellm_text_dim is not None and (
            isinstance(self.routellm_text_dim, bool)
            or not isinstance(self.routellm_text_dim, int)
            or self.routellm_text_dim < 1
        ):
            raise ValueError("LLMConfig.routellm_text_dim must be a positive int")
