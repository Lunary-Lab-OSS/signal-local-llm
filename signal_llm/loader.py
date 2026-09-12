"""
LLM Loader
Handles downloading and loading LLM models from HuggingFace with platform-specific optimizations.
"""

import logging
import os
import platform
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalLLMModelLoader:
    """Handles LLM model loading with automatic download from HuggingFace"""

    def __init__(self, models_dir: Path, cache_dir: Path, device: str = "cuda", system_config=None):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device_config = device
        self.system_config = system_config

        self.platform = platform.system().lower()
        self.machine = platform.machine()

        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

    def load_llm_model(
        self, model_priority_list, draft_model_path: Path | None = None
    ) -> tuple[Any, str]:
        """
        Load LLM model from priority list.
        Returns (model, model_name) tuple.
        """
        last_error = None

        for idx, model_config in enumerate(model_priority_list):
            repo_id = model_config.repo_id
            model_name = model_config.name
            revision = model_config.revision
            local_path = self._get_model_path(repo_id, revision)

            logger.info(
                f"📦 Trying LLM model {idx + 1}/{len(model_priority_list)}: "
                f"{model_name} ({repo_id})"
            )
            if revision:
                logger.info(f"   Using revision: {revision}")

            try:
                if not local_path.exists() or not (local_path / "config.json").exists():
                    logger.info(f"   Downloading model: {repo_id}")
                    if local_path.exists():
                        logger.warning(
                            "   ⚠️ Model directory exists but config.json missing. Re-downloading..."
                        )
                    self._download_model(repo_id, local_path, revision)

                if self.platform == "darwin" and self.machine == "arm64":
                    result = self._load_llm_mlx(local_path, model_config)
                    if result:
                        logger.info(f"   ✅ Successfully loaded: {model_name} using MLX")
                        return result, model_name
                elif self.platform == "windows":
                    result = self._load_llm_exllama(
                        local_path, model_config, draft_model_path=draft_model_path
                    )
                    if result:
                        logger.info(f"   ✅ Successfully loaded: {model_name} using ExLlamaV2/vLLM")
                        return result, model_name
                else:
                    raise NotImplementedError(f"LLM loading not implemented for {self.platform}")

            except Exception as e:
                logger.warning(f"   ⚠️ Failed to load {model_name}: {e}")
                last_error = e
                continue

        raise RuntimeError(f"Failed to load any LLM model. Last error: {last_error}")

    def _get_model_path(self, repo_id: str, revision: str | None = None) -> Path:
        safe_name = repo_id.replace("/", "_")
        if revision and revision != "main":
            safe_name = f"{safe_name}_{revision}"
        model_path = self.models_dir / safe_name
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    def _download_model(self, repo_id: str, local_path: Path, revision: str | None = None):
        try:
            from huggingface_hub import snapshot_download

            local_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {repo_id} to {local_path}...")
            kwargs = {
                "repo_id": repo_id,
                "local_dir": str(local_path),
                "local_dir_use_symlinks": False,
                "cache_dir": None,
            }
            if revision:
                kwargs["revision"] = revision
            if self.hf_token:
                kwargs["token"] = self.hf_token

            snapshot_download(**kwargs)
            logger.info(f"[OK] Download complete: {local_path}")
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            ) from None
        except Exception as e:
            logger.error(f"Failed to download model {repo_id}: {e}")
            raise

    def _load_llm_mlx(self, model_path: Path, config) -> Any:
        """Load LLM using MLX (Apple Silicon)"""
        try:
            from mlx_lm import load

            logger.info(f"Loading LLM with MLX from {model_path}")
            if model_path.exists() and (model_path / "config.json").exists():
                model, tokenizer = load(str(model_path))
            else:
                repo_id = str(model_path.name).replace("_", "/")
                model, tokenizer = load(repo_id)

            logger.info("MLX LLM model loaded successfully")
            return model, tokenizer
        except ImportError:
            logger.warning("mlx_lm not available - install with: pip install mlx-lm")
            return None
        except Exception as e:
            logger.error(f"Failed to load MLX LLM model: {e}")
            return None

    def _load_llm_exllama(
        self, model_path: Path, config, draft_model_path: Path | None = None
    ) -> Any:
        """Load LLM using ExLlamaV2, vLLM, or Transformers"""
        if self.device_config == "cuda" and not self._has_cuda():
            raise RuntimeError(
                "FATAL: CUDA is configured but not available.\n"
                "   RESTART THE APPLICATION for CUDA to work."
            )
        elif self.device_config == "cpu" and self._has_cuda():
            logger.warning("CUDA is available but CPU is configured. Performance will be limited.")

        backend = getattr(config, "backend", "exllamav2").lower()
        logger.info(f"Loading LLM using configured backend: {backend}")

        if backend == "exllamav2":
            return self._load_exllamav2(model_path, config, draft_model_path)
        elif backend == "vllm":
            return self._load_vllm(model_path, config, draft_model_path)
        elif backend == "transformers":
            return self._load_transformers(model_path, config)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'exllamav2', 'vllm', or 'transformers'."
            )

    def _load_exllamav2(
        self, model_path: Path, config, draft_model_path: Path | None = None
    ) -> Any:
        if not (self.device_config == "cuda" and self._has_cuda()):
            raise RuntimeError("FATAL: ExLlamaV2 requires CUDA.")

        from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator

        logger.info(f"Loading LLM with ExLlamaV2 from {model_path}")

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        model_config = ExLlamaV2Config(str(model_path))
        model = ExLlamaV2(model_config)
        model.load()
        cache = ExLlamaV2Cache(model, max_seq_len=8192)
        tokenizer = ExLlamaV2Tokenizer(model_config)

        draft_model = None
        draft_cache = None
        num_spec_tokens = 5

        if draft_model_path and draft_model_path.exists():
            try:
                spec_cfg = getattr(config, "speculative_decoding", None)
                if spec_cfg and spec_cfg.enabled:
                    num_spec_tokens = spec_cfg.num_speculative_tokens
                    draft_config_path = draft_model_path / "config.json"
                    if draft_config_path.exists():
                        draft_model_config = ExLlamaV2Config(str(draft_model_path))
                        draft_model = ExLlamaV2(draft_model_config)
                        draft_model.load()
                        draft_cache = ExLlamaV2Cache(draft_model, max_seq_len=8192)
                        logger.info(
                            f"✅ Loaded draft model for speculative decoding "
                            f"({num_spec_tokens} tokens)"
                        )
            except Exception as e:
                logger.warning(f"Failed to load draft model: {e}")

        if draft_model and draft_cache:
            generator = ExLlamaV2DynamicGenerator(
                model,
                cache,
                tokenizer,
                draft_model=draft_model,
                draft_cache=draft_cache,
                num_speculative_tokens=num_spec_tokens,
            )
        else:
            generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)

        hf_tokenizer = None
        try:
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            logger.info("   + Loaded HF Tokenizer for chat templating")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not load HF Tokenizer: {e}")

        logger.info("✅ ExLlamaV2 model loaded")
        return (generator, hf_tokenizer)

    def _load_vllm(self, model_path: Path, config, draft_model_path: Path | None = None) -> Any:
        if not (self.device_config == "cuda" and self._has_cuda()):
            raise RuntimeError("FATAL: vLLM requires CUDA.")

        from vllm import LLM

        model_name = (
            str(model_path)
            if (model_path.exists() and (model_path / "config.json").exists())
            else str(model_path.name).replace("_", "/")
        )

        is_awq = "AWQ" in str(model_path) or "awq" in str(model_path).lower()

        spec_decode_config = None
        if draft_model_path:
            draft_name = (
                str(draft_model_path)
                if (draft_model_path.exists() and (draft_model_path / "config.json").exists())
                else str(draft_model_path.name).replace("_", "/")
            )
            num_spec_tokens = getattr(
                getattr(config, "speculative_decoding", None), "num_speculative_tokens", 5
            )
            from vllm import SpeculativeConfig

            spec_decode_config = SpeculativeConfig(
                draft_model=draft_name,
                num_speculative_tokens=num_spec_tokens,
            )

        gpu_memory_util = 0.85
        if self.system_config and getattr(self.system_config, "gpu_memory", None):
            gpu_memory_util = self.system_config.gpu_memory.llm_memory_utilization

        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_util,
            max_model_len=32768,
            dtype="float16" if not is_awq else "auto",
            quantization="awq" if is_awq else None,
            speculative_config=spec_decode_config,
        )
        logger.info("✅ vLLM model loaded")
        return llm

    def _load_transformers(self, model_path: Path, config) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = (
            str(model_path)
            if (model_path.exists() and (model_path / "config.json").exists())
            else str(model_path.name).replace("_", "/")
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = self.device_config

        if device == "cuda" and not self._has_cuda():
            raise RuntimeError("FATAL: CUDA is configured but not available.")

        is_awq = "AWQ" in str(model_path) or "awq" in str(model_path).lower()

        if config.quantized and not is_awq and device == "cuda":
            from transformers import BitsAndBytesConfig

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )

        logger.info(f"✅ Transformers model loaded on {device}")
        return (model, tokenizer)

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False


LLMLoader = LocalLLMModelLoader
