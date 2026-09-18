"""
LLM Loader

Downloads and loads LLM models from HuggingFace with explicit backend
selection and platform-aware device handling.

Contract (remediation L01-L05):

- Dispatch is driven by the **configured backend** first; platform/device
  compatibility is validated *before* any download is attempted.
- CUDA devices keep their index (``cuda:1`` stays ``cuda:1``); unsupported
  device strings are rejected instead of silently mapping to CPU.
- ``trust_remote_code`` is False by default; custom tokenizer/model code
  requires explicit per-model opt-in (``trust_remote_code=True`` on the
  ModelConfig) plus a pinned revision.
- Local cache directories are keyed by a collision-resistant encoding of
  ``repo_id`` + ``revision``; revision components can never traverse out
  of the models root, and equivalent spellings cannot collide.
- A snapshot is considered complete only when a manifest written *after* a
  successful download exists; interrupted downloads resume instead of
  being mistaken for complete models.
"""

import contextlib
import hashlib
import json
import logging
import os
import platform as platform_module
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import ModelConfig, validate_device, validate_remote_code

logger = logging.getLogger(__name__)

_MANIFEST_NAME = ".signal-llm-manifest.json"
_MANIFEST_VERSION = 2
_SUPPORTED_BACKENDS = ("exllamav2", "vllm", "transformers", "mlx")
_MANIFEST_LOCKS: dict[str, threading.Lock] = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded under the configured backend."""


@dataclass(frozen=True)
class LoadMetadata:
    config: ModelConfig
    backend: str
    model_path: Path
    draft_model_path: Path | None = None
    device: str | None = None


class ModelLoadResult(tuple):
    """Two-item (handle, name) result with metadata independent of handle mutability."""

    metadata: LoadMetadata

    def __new__(cls, handle: Any, name: str, metadata: LoadMetadata):
        result = super().__new__(cls, (handle, name))
        result.metadata = metadata
        return result


class BackendHandle(tuple):
    """Two-item (engine, tokenizer) handle with the successfully activated draft path."""

    draft_model_path: Path | None

    def __new__(cls, engine: Any, tokenizer: Any, draft_model_path: Path | None = None):
        result = super().__new__(cls, (engine, tokenizer))
        result.draft_model_path = draft_model_path
        return result


def parse_device(device: str | None) -> tuple[str, int | None]:
    """Split a device string into ``(kind, index)``.

    ``"cuda"`` -> ``("cuda", None)``; ``"cuda:1"`` -> ``("cuda", 1)``;
    ``"cpu"`` -> ``("cpu", None)``; ``"auto"`` -> ``("auto", None)``.
    Anything else raises ``ValueError``.
    """
    return validate_device(device)


def model_cache_key(repo_id: str, revision: str | None) -> str:
    """Return a collision-resistant cache key for repo/revision identity.

    A truncated SHA-256 of the full identity is embedded so distinct
    ``(repo, revision)`` tuples can never share a directory (L04: the old
    ``repo.replace('/', '_') + '_' + revision`` scheme collided for names
    like ``a_b/c`` vs ``a/b_c`` and allowed revision components such as
    ``x/../../outside`` to escape the cache root).
    """
    identity = f"{repo_id}@{revision or 'main'}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    readable = repo_id.replace("/", "--")
    return f"{readable}-{digest}"


class LocalLLMModelLoader:
    """Handles LLM model loading with automatic download from HuggingFace."""

    def __init__(self, models_dir: Path, cache_dir: Path, device: str = "auto", system_config=None):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device_config = device
        self.system_config = system_config

        self.platform = platform_module.system().lower()
        self.machine = platform_module.machine()

        import os

        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    #: Attached to every successful load: the ModelConfig that actually
    #: won (which may be a fallback candidate, not priority[0]). Callers
    #: must use this for identity decisions instead of assuming the first
    #: configured choice loaded (L06).
    LAST_LOADED_CONFIG_ATTRIBUTE = "loaded_model_config"

    def load_llm_model(
        self,
        model_priority_list,
        draft_model_path: Path | None = None,
        speculative: Any | None = None,
    ) -> ModelLoadResult:
        """Load the first loadable model from a priority list.

        Returns ``(handle, model_name)`` where handle is backend-specific
        (see the ``_load_*`` methods). ``speculative`` is an optional
        ``SpeculativeDecodingConfig`` honoured by supporting backends.

        The authoritative identity is returned in ``result.metadata``:
        a copy of the winning config, resolved backend, absolute model path,
        and absolute draft path only when speculation actually activated.
        Tuple unpacking remains ``handle, name = result``. Backend handles
        containing tokenizers remain ``(engine, tokenizer)`` (including vLLM).

        The winning candidate's ModelConfig is also attached to the handle as
        ``handle.loaded_model_config`` (and to tuple handles' first
        element) so identity reflects what actually loaded, not what was
        configured first (L06: fallback candidates change repo/revision/
        backend silently otherwise).
        """
        if not model_priority_list:
            raise ModelLoadError("model priority list is empty")

        parse_device(self.device_config)
        for candidate in model_priority_list:
            self._resolve_backend(candidate)
            if not isinstance(candidate, ModelConfig):
                raise ValueError("model priority entries must be ModelConfig")
            replace(candidate)
        if speculative is not None:
            replace(speculative)

        last_error: Exception | None = None
        errors: list[str] = []

        for model_config in model_priority_list:
            name = getattr(model_config, "name", None) or getattr(model_config, "repo_id", "?")
            try:
                handle = self._load_one(model_config, draft_model_path, speculative)
            except Exception as exc:
                logger.warning("   ⚠️ Failed to load %s: %s", name, exc)
                last_error = exc
                errors.append(f"{name}: {exc}")
                continue
            logger.info("   ✅ Successfully loaded: %s", name)
            with contextlib.suppress(AttributeError, TypeError):
                setattr(handle, self.LAST_LOADED_CONFIG_ATTRIBUTE, model_config)
            if isinstance(handle, tuple) and handle:
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(handle[0], self.LAST_LOADED_CONFIG_ATTRIBUTE, model_config)
            metadata = LoadMetadata(
                config=replace(model_config),
                backend=self._resolve_backend(model_config),
                model_path=(
                    Path(model_config.local_path).resolve()
                    if model_config.local_path
                    else self._get_model_path(model_config.repo_id, model_config.revision).resolve()
                ),
                draft_model_path=getattr(handle, "draft_model_path", None),
                device=self._loaded_device(model_config, handle),
            )
            return ModelLoadResult(handle, str(name), metadata)

        raise ModelLoadError(
            "failed to load any model. Attempts: "
            + "; ".join(errors)
            + (f" (last error: {last_error})" if last_error else "")
        )

    # ------------------------------------------------------------------ #
    # Backend dispatch
    # ------------------------------------------------------------------ #

    def _requested_device(self, model_config) -> tuple[str, int | None]:
        device = getattr(model_config, "device", None)
        parsed = parse_device(device)
        return parse_device(self.device_config) if parsed[0] == "auto" else parsed

    def _loaded_device(self, model_config, handle) -> str:
        engine = handle[0] if isinstance(handle, tuple) else handle
        actual = getattr(engine, "device", None)
        if actual is not None:
            return str(actual)
        backend = self._resolve_backend(model_config)
        if backend in ("exllamav2", "vllm"):
            return "cuda"
        if backend == "mlx":
            return "mps"
        kind, index = self._requested_device(model_config)
        if kind == "auto":
            kind = "cuda" if self._has_cuda() else "cpu"
        return f"{kind}:{index}" if index is not None else kind

    def _resolve_backend(self, model_config) -> str:
        """Validate and return the configured backend (L01).

        The backend must be explicitly set (default ``exllamav2`` is kept
        for backwards compatibility with existing ModelConfig instances);
        ``auto`` is supported and resolves per-platform.
        """
        backend = (getattr(model_config, "backend", "exllamav2") or "").strip().lower()
        if backend == "auto":
            if self.platform == "darwin" and self.machine == "arm64":
                return "mlx"
            if self.platform == "windows":
                return "exllamav2"
            return "transformers"
        if backend in _SUPPORTED_BACKENDS:
            return backend
        raise ValueError(
            f"unknown backend {backend!r}; use one of {sorted(_SUPPORTED_BACKENDS)} or 'auto'"
        )

    def _validate_platform_support(self, backend: str, model_config) -> None:
        """Reject unsupported backend/platform/device combos *before* download."""
        kind, index = self._requested_device(model_config)
        if backend in ("exllamav2", "vllm"):
            if kind not in ("auto", "cuda"):
                raise ModelLoadError(f"backend {backend!r} requires CUDA; cannot use {kind!r}")
            if index is not None:
                raise ModelLoadError(
                    f"backend {backend!r} does not support explicit CUDA index placement; "
                    "use an externally restricted CUDA_VISIBLE_DEVICES and device='cuda'"
                )
        if backend == "mlx" and not (self.platform == "darwin" and self.machine == "arm64"):
            raise ModelLoadError(
                "backend 'mlx' requires Apple Silicon (darwin/arm64); "
                f"got {self.platform}/{self.machine}"
            )
        if backend == "exllamav2" and not self._has_cuda():
            raise ModelLoadError(
                "backend 'exllamav2' requires CUDA, which is not available on this host"
            )
        if backend == "vllm" and not self._has_cuda():
            raise ModelLoadError("backend 'vllm' requires CUDA, which is not available here")
        if kind == "cuda" and not self._has_cuda():
            raise ModelLoadError(f"CUDA device requested ({self.device_config}) but unavailable")
        if kind == "cuda" and index is not None and not self._cuda_index_available(index):
            raise ModelLoadError(f"CUDA device index {index} is not available on this host")
        del model_config  # platform rules only; unused beyond signature clarity

    def _load_one(self, model_config, draft_model_path, speculative) -> Any:
        validate_remote_code(model_config.trust_remote_code, model_config.revision)
        repo_id = model_config.repo_id
        backend = self._resolve_backend(model_config)

        # L01: validate platform/device support BEFORE downloading anything.
        self._validate_platform_support(backend, model_config)

        local_path = self._get_model_path(repo_id, model_config.revision)
        if model_config.local_path:
            # L16: an explicit local path bypasses the hub entirely.
            local_path = Path(model_config.local_path)
            if not (local_path / "config.json").exists():
                raise ModelLoadError(
                    f"local_path {local_path} does not contain a model (config.json missing)"
                )
        else:
            self._ensure_downloaded(repo_id, local_path, model_config.revision)

        if backend == "mlx":
            return self._load_llm_mlx(local_path, model_config)
        if backend == "exllamav2":
            return self._load_exllamav2(local_path, model_config, draft_model_path, speculative)
        if backend == "vllm":
            return self._load_vllm(local_path, model_config, draft_model_path, speculative)
        if backend == "transformers":
            return self._load_transformers(local_path, model_config)
        raise ValueError(f"unhandled backend {backend!r}")  # pragma: no cover - guarded above

    # ------------------------------------------------------------------ #
    # Local cache paths (L04)
    # ------------------------------------------------------------------ #

    def _get_model_path(self, repo_id: str, revision: str | None = None) -> Path:
        """Return (and create) the cache directory for a repo/revision.

        The directory name embeds a hash of the full identity, so revisions
        containing path separators or ``..`` cannot traverse out of the
        models root, and distinct identities never collide.
        """
        key = model_cache_key(repo_id, revision)
        model_path = self.models_dir / key
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    # ------------------------------------------------------------------ #
    # Download + completion manifest (L05)
    # ------------------------------------------------------------------ #

    def _manifest_path(self, local_path: Path) -> Path:
        return local_path / _MANIFEST_NAME

    def _manifest_lock(self, local_path: Path) -> threading.Lock:
        with _MANIFEST_LOCKS_GUARD:
            return _MANIFEST_LOCKS.setdefault(str(local_path), threading.Lock())

    def _is_complete(
        self, local_path: Path, repo_id: str | None = None, revision: str | None = None
    ) -> bool:
        manifest = self._manifest_path(local_path)
        if not manifest.exists():
            return False
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if not isinstance(data, dict):
            return False
        if (
            type(data.get("version")) is not int
            or data["version"] != _MANIFEST_VERSION
            or data.get("complete") is not True
            or not isinstance(data.get("repo_id"), str)
            or not data["repo_id"]
            or (data.get("revision") is not None and not isinstance(data["revision"], str))
            or (repo_id is not None and data["repo_id"] != repo_id)
            or (repo_id is not None and (data.get("revision") or "main") != (revision or "main"))
        ):
            return False
        files = data.get("files")
        if not isinstance(files, dict) or "config.json" not in files:
            return False
        try:
            for name, size in files.items():
                relative = Path(name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or type(size) is not int
                    or size < 0
                ):
                    return False
                path = local_path / relative
                if not path.resolve().is_relative_to(local_path.resolve()):
                    return False
                if not path.is_file() or path.stat().st_size != size:
                    return False
        except (OSError, ValueError, RuntimeError):
            return False
        return True

    def _write_manifest(
        self, local_path: Path, repo_id: str, revision: str | None, resolved
    ) -> None:
        payload = {
            "version": _MANIFEST_VERSION,
            "complete": True,
            "repo_id": repo_id,
            "revision": revision,
            "resolved_commit": resolved,
            "files": {
                str(path.relative_to(local_path)): path.stat().st_size
                for path in sorted(local_path.rglob("*"))
                if path.is_file()
                and path.name != _MANIFEST_NAME
                and not path.name.startswith(f"{_MANIFEST_NAME}.")
                and ".cache" not in path.relative_to(local_path).parts
            },
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=local_path,
                prefix=f"{_MANIFEST_NAME}.",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self._manifest_path(local_path))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _ensure_downloaded(self, repo_id: str, local_path: Path, revision: str | None) -> None:
        if self._is_complete(local_path, repo_id, revision):
            return
        with self._manifest_lock(local_path):
            if self._is_complete(local_path, repo_id, revision):
                return
            if (local_path / "config.json").exists():
                logger.info(
                    "   Incomplete snapshot detected (no completion manifest); resuming download"
                )
            self._download_model(repo_id, local_path, revision)

    def _download_model(self, repo_id: str, local_path: Path, revision: str | None = None):
        from huggingface_hub import snapshot_download

        local_path.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s to %s ...", repo_id, local_path)
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": str(local_path),
            # NOTE: local_dir_use_symlinks was removed in huggingface_hub 1.x;
            # current versions always materialise real files in local_dir.
        }
        if revision:
            kwargs["revision"] = revision
        if self.hf_token:
            kwargs["token"] = self.hf_token

        result = snapshot_download(**kwargs)
        if not (local_path / "config.json").is_file():
            raise ModelLoadError(
                f"download of {repo_id} did not produce config.json in {local_path}"
            )
        # snapshot_download returns the local directory path, not a commit
        # id; do not record it under a misleading name.
        del result
        self._write_manifest(local_path, repo_id, revision, None)
        logger.info("[OK] Download complete: %s", local_path)

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #

    def _load_llm_mlx(self, model_path: Path, config) -> Any:
        """Load LLM using MLX (Apple Silicon)."""
        try:
            from mlx_lm import load
        except ImportError:
            raise ModelLoadError(
                "mlx_lm not available - install with: pip install mlx-lm"
            ) from None

        logger.info("Loading LLM with MLX from %s", model_path)
        if model_path.exists() and (model_path / "config.json").exists():
            model, tokenizer = load(str(model_path))
        else:
            repo_id = getattr(config, "repo_id", str(model_path.name))
            model, tokenizer = load(repo_id)

        logger.info("MLX LLM model loaded successfully")
        return model, tokenizer

    def _load_exllamav2(
        self,
        model_path: Path,
        config,
        draft_model_path: Path | None = None,
        speculative: Any | None = None,
    ) -> Any:
        self._validate_platform_support("exllamav2", config)
        validate_remote_code(config.trust_remote_code, config.revision)
        from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator

        logger.info("Loading LLM with ExLlamaV2 from %s", model_path)
        if not (model_path / "config.json").exists():
            raise ModelLoadError(f"Config file not found at {model_path / 'config.json'}")

        model_config = ExLlamaV2Config(str(model_path))
        model = ExLlamaV2(model_config)
        model.load()
        cache = ExLlamaV2Cache(model, max_seq_len=8192)
        tokenizer = ExLlamaV2Tokenizer(model_config)

        draft_model = None
        draft_cache = None
        num_spec_tokens = 5

        spec_cfg = (
            speculative
            if speculative is not None
            else getattr(config, "speculative_decoding", None)
        )
        if draft_model_path and draft_model_path.exists() and spec_cfg and spec_cfg.enabled:
            num_spec_tokens = spec_cfg.num_speculative_tokens
            try:
                draft_config_path = draft_model_path / "config.json"
                if draft_config_path.exists():
                    draft_model_config = ExLlamaV2Config(str(draft_model_path))
                    draft_model = ExLlamaV2(draft_model_config)
                    draft_model.load()
                    draft_cache = ExLlamaV2Cache(draft_model, max_seq_len=8192)
                    logger.info(
                        "✅ Loaded draft model for speculative decoding (%d tokens)",
                        num_spec_tokens,
                    )
            except Exception:
                logger.warning("Failed to load draft model", exc_info=True)
                draft_model = None
                draft_cache = None

        if draft_model and draft_cache:
            generator = ExLlamaV2DynamicGenerator(
                model,
                cache,
                tokenizer,
                draft_model=draft_model,
                draft_cache=draft_cache,
                num_draft_tokens=num_spec_tokens,
            )
        else:
            generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)

        hf_tokenizer = None
        try:
            from transformers import AutoTokenizer

            # L03: remote code execution requires explicit per-model opt-in.
            trust = bool(getattr(config, "trust_remote_code", False))
            hf_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=trust,
            )
            logger.info("   + Loaded HF Tokenizer for chat templating")
        except Exception:
            logger.warning("   ⚠️ Could not load HF Tokenizer", exc_info=True)

        logger.info("✅ ExLlamaV2 model loaded")
        active_draft = (
            draft_model_path.resolve()
            if draft_model_path is not None and draft_model and draft_cache
            else None
        )
        return BackendHandle(generator, hf_tokenizer, active_draft)

    def _load_vllm(
        self,
        model_path: Path,
        config,
        draft_model_path: Path | None = None,
        speculative: Any | None = None,
    ) -> Any:
        self._validate_platform_support("vllm", config)
        validate_remote_code(config.trust_remote_code, config.revision)
        from vllm import LLM

        model_name = (
            str(model_path)
            if (model_path.exists() and (model_path / "config.json").exists())
            else getattr(config, "repo_id", str(model_path.name))
        )

        is_awq = "awq" in str(model_path).lower()

        spec_decode_config = None
        spec_cfg = (
            speculative
            if speculative is not None
            else getattr(config, "speculative_decoding", None)
        )
        if draft_model_path and spec_cfg and spec_cfg.enabled:
            if not (draft_model_path / "config.json").is_file():
                raise ModelLoadError(f"draft_model_path {draft_model_path} lacks config.json")
            spec_decode_config = {
                "model": str(draft_model_path.resolve()),
                "num_speculative_tokens": spec_cfg.num_speculative_tokens,
            }

        gpu_memory_util = 0.85
        if self.system_config and getattr(self.system_config, "gpu_memory", None):
            gpu_memory_util = self.system_config.gpu_memory.llm_memory_utilization

        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_util,
            max_model_len=32768,
            dtype="auto" if is_awq else "float16",
            quantization="awq" if is_awq else None,
            speculative_config=spec_decode_config,
            trust_remote_code=config.trust_remote_code,
        )
        logger.info("✅ vLLM model loaded")
        active_draft = (
            draft_model_path.resolve()
            if draft_model_path is not None and spec_decode_config is not None
            else None
        )
        return BackendHandle(llm, llm.get_tokenizer(), active_draft)

    def _load_transformers(self, model_path: Path, config) -> Any:
        validate_remote_code(config.trust_remote_code, config.revision)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = (
            str(model_path)
            if (model_path.exists() and (model_path / "config.json").exists())
            else getattr(config, "repo_id", str(model_path.name))
        )

        # L02: honour the requested device precisely, including CUDA index.
        kind, index = self._requested_device(config)
        if kind == "auto":
            kind = "cuda" if self._has_cuda() else "cpu"
            index = None

        trust = bool(getattr(config, "trust_remote_code", False))
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        dtype = getattr(config, "dtype", None) or ("float16" if kind == "cuda" else "float32")
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float32)

        is_awq = "awq" in str(model_path).lower()
        quantize = bool(getattr(config, "quantized", False)) and not is_awq and kind == "cuda"
        target = f"cuda:{index}" if kind == "cuda" and index is not None else kind
        device_map = {"": target} if kind == "cuda" else None

        if quantize:
            from transformers import BitsAndBytesConfig

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch_dtype,
                trust_remote_code=trust,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust,
            )
            if kind != "cuda":
                # CPU/MPS placement is applied explicitly to the loaded model.
                model = model.to(target)  # type: ignore[union-attr,arg-type]

        logger.info("✅ Transformers model loaded on %s", kind)
        return (model, tokenizer)

    # ------------------------------------------------------------------ #
    # Capability probes
    # ------------------------------------------------------------------ #

    def _has_cuda(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _cuda_index_available(self, index: int) -> bool:
        try:
            import torch

            return bool(0 <= index < torch.cuda.device_count())
        except ImportError:
            return False


LLMLoader = LocalLLMModelLoader
