"""
Matrix Factorization Router Strategy

Implementation of the router strategy pattern for Matrix Factorization router.
"""

import contextlib
import logging
import os
from pathlib import Path
from typing import Any

import torch

from signal_llm.config import LLMConfig

from .base import RouterStrategy

logger = logging.getLogger(__name__)


class MatrixFactorizationRouterStrategy(RouterStrategy):
    """
    Matrix Factorization router strategy implementation.

    Handles loading of embedding models and creation of RouteLLM controller
    with Matrix Factorization router.
    """

    def __init__(self, config: LLMConfig, device: str, platform: str):
        """
        Initialize Matrix Factorization router strategy.

        Args:
            config: LLM configuration (contains router settings)
            device: Device string (e.g., "cuda:0", "mps", "cpu")
            platform: Platform string (e.g., "windows", "macos")
        """
        super().__init__(config, device, platform)
        self._embedding_model: Any | None = None

    def load_controller(self) -> Any:
        """
        Load and return the RouteLLM controller with Matrix Factorization router.

        Returns:
            RouteLLM Controller instance
        """
        # Lazy import to avoid OpenAI initialization at module level
        # RouteLLM's import requires OPENAI_API_KEY to be present; the MF router
        # never makes OpenAI calls. Set a dummy only when unset, and restore the
        # original environment state once the controller is constructed.
        _dummy_key_set = "OPENAI_API_KEY" not in os.environ
        _original_key = os.environ.get("OPENAI_API_KEY")
        if _dummy_key_set:
            os.environ["OPENAI_API_KEY"] = "dummy-key-for-mf-router"

        try:
            from routellm.controller import Controller as RouteLLMController

            router_name = self.config.routellm_router_name
            if router_name != "mf":
                raise ValueError(
                    f"MatrixFactorizationRouterStrategy only supports router_name='mf', got '{router_name}'"
                )

            # Load optimized embedding model on startup (stays in VRAM)
            embedding_model = self._load_embedding_model()

            # Build config with embedding model for MF router
            checkpoint_path = self.config.routellm_checkpoint_path or "routellm/mf_gpt4_augmented"
            if self.config.routellm_checkpoint_path:
                logger.info(f"Using custom RouteLLM checkpoint: {checkpoint_path}")
            else:
                logger.info(f"Using default RouteLLM checkpoint: {checkpoint_path}")

            # Read checkpoint metadata to get dim (hidden_size) and other parameters
            checkpoint_dim = None
            if checkpoint_path and Path(checkpoint_path).exists():
                try:
                    checkpoint_data = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                    checkpoint_dim = checkpoint_data.get("dim", None)
                    if checkpoint_dim:
                        logger.info(
                            f"📝 Auto-detected hidden_size (dim) from checkpoint: {checkpoint_dim}"
                        )
                except Exception as e:
                    logger.debug(f"Could not read dim from checkpoint: {e}")

            router_config = {
                router_name: {
                    "checkpoint_path": checkpoint_path,
                    # text_dim will be auto-detected from checkpoint if not provided
                }
            }
            # Add hidden_size (dim) from checkpoint if available
            if checkpoint_dim is not None:
                router_config[router_name]["hidden_size"] = checkpoint_dim
            # Add text_dim if explicitly provided in config (otherwise auto-detected from checkpoint)
            if self.config.routellm_text_dim is not None:
                router_config[router_name]["text_dim"] = self.config.routellm_text_dim
            # Only add embedding_model if it was successfully loaded
            if embedding_model is not None:
                router_config[router_name]["embedding_model"] = embedding_model

            # RouteLLM requires strong_model and weak_model - we use agent and router models
            # For now, we'll use dummy models since RouteLLM just needs the router logic
            self.controller = RouteLLMController(
                routers=[router_name],
                strong_model="dummy-strong",  # Not used for routing decision
                weak_model="dummy-weak",  # Not used for routing decision
                config=router_config,
            )
            logger.info(f"✅ RouteLLM controller initialized with router '{router_name}'")
            return self.controller
        except ImportError as e:
            raise RuntimeError(
                f"FATAL: RouteLLM is configured but not installed. Install with: pip install 'routellm[serve]'. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Failed to initialize RouteLLM controller: {e}. RouteLLM is REQUIRED when router_type='routellm'."
            ) from e
        finally:
            if _dummy_key_set:
                if _original_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = _original_key

    def _load_embedding_model(self) -> Any | None:
        """
        Load optimized embedding model for Matrix Factorization router.

        Returns:
            SentenceTransformer model instance, or None if loading fails
        """
        try:
            from pathlib import Path

            from sentence_transformers import SentenceTransformer

            # Try to auto-detect embedding model from checkpoint
            checkpoint_path = self.config.routellm_checkpoint_path or "routellm/mf_gpt4_augmented"
            embedding_model_name = self.config.routellm_embedding_model

            # If checkpoint is a local file, try to read embedding_model from it
            if checkpoint_path and Path(checkpoint_path).exists():
                try:
                    checkpoint_data = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                    checkpoint_embedding_model = checkpoint_data.get("embedding_model", None)
                    if checkpoint_embedding_model:
                        logger.info(
                            f"📝 Auto-detected embedding model from checkpoint: {checkpoint_embedding_model}"
                        )
                        # Use checkpoint embedding model if config doesn't specify one, or if they match
                        if (
                            not embedding_model_name
                            or embedding_model_name == "nomic-ai/modernbert-embed-base"
                        ):
                            embedding_model_name = checkpoint_embedding_model
                            logger.info(
                                f"   Using embedding model from checkpoint: {embedding_model_name}"
                            )
                        elif embedding_model_name != checkpoint_embedding_model:
                            logger.warning(
                                f"⚠️  Config embedding model ({embedding_model_name}) differs from checkpoint ({checkpoint_embedding_model}). Using config value."
                            )
                except Exception as e:
                    logger.debug(f"Could not read embedding model from checkpoint: {e}")

            embedding_device = self.config.routellm_embedding_device
            embedding_dtype_str = self.config.routellm_embedding_dtype
            embedding_compile = self.config.routellm_embedding_compile

            logger.info(f"Loading embedding model '{embedding_model_name}' for RouteLLM...")
            logger.info(
                f"   Device: {embedding_device}, Dtype: {embedding_dtype_str}, Compile: {embedding_compile}"
            )

            # Load model - CPU recommended to save VRAM (~0.5GB savings)
            # Trade-off: ~15ms latency (CPU) vs ~2ms (GPU), but imperceptible for voice commands
            if embedding_device == "cpu":
                device_str = "cpu"
                logger.info(
                    "   💡 Using CPU for embeddings (saves ~0.5GB VRAM, ~15ms latency is imperceptible)"
                )
            elif embedding_device == "cuda":
                # Check for MPS (macOS) first if CUDA requested but on macOS
                if self.platform == "macos" and torch.backends.mps.is_available():
                    device_str = "mps"
                    logger.info("   💡 Using MPS for embeddings (macOS)")
                elif torch.cuda.is_available():
                    device_str = "cuda"
                else:
                    device_str = "cpu"
                    logger.info("   ⚠️  CUDA requested but unavailable, falling back to CPU")
            else:
                device_str = embedding_device
            device = torch.device(device_str)

            # Suppress Flash Attention warning during load (harmless - model works fine)
            # The warning occurs because SentenceTransformer loads on CPU first, then moves to GPU
            # Suppress both Python warnings and transformers logger warnings
            import logging as _logging
            import warnings

            # Temporarily suppress transformers logger warnings
            transformers_logger = _logging.getLogger("transformers")
            original_level = transformers_logger.level
            transformers_logger.setLevel(_logging.ERROR)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Flash Attention.*")
                    warnings.filterwarnings("ignore", message=".*flash_attention.*")
                    warnings.filterwarnings("ignore", message=".*attn_implementation.*")
                    # Some models (e.g., Alibaba-NLP/gte-base-en-v1.5) require trust_remote_code=True
                    embedding_model = SentenceTransformer(
                        embedding_model_name, device=device_str, trust_remote_code=True
                    )
            finally:
                # Restore original logger level
                transformers_logger.setLevel(original_level)

            # Convert to specified dtype immediately after loading
            if device.type == "cpu":
                # CPU uses float32 (no bfloat16/float16 support)
                logger.info("   ✅ Using float32 on CPU (optimal for CPU inference)")
            elif embedding_dtype_str == "bfloat16" and device.type == "cuda":
                # Convert all modules to bfloat16
                embedding_model = embedding_model.to(torch.bfloat16)
                # Also ensure underlying transformer modules are converted
                if hasattr(embedding_model, "_modules"):
                    for module in embedding_model._modules.values():
                        if hasattr(module, "to"):
                            with contextlib.suppress(Exception):
                                module.to(device=device, dtype=torch.bfloat16)
                logger.info("   ✅ Converted to bfloat16 (optimal for RTX 4090)")
            elif embedding_dtype_str == "float16" and (
                device.type == "cuda" or device.type == "mps"
            ):
                embedding_model = embedding_model.to(torch.float16)
                if hasattr(embedding_model, "_modules"):
                    for module in embedding_model._modules.values():
                        if hasattr(module, "to"):
                            with contextlib.suppress(Exception):
                                module.to(device=device, dtype=torch.float16)
                logger.info("   ✅ Converted to float16")

            # Compile for maximum speed (CUDA kernels only - not needed for CPU/MPS)
            if embedding_compile and device.type == "cuda":
                embedding_model.encode = torch.compile(
                    embedding_model.encode, mode="reduce-overhead"
                )
                logger.info("   ✅ Compiled with torch.compile (raw CUDA kernels)")
            elif embedding_compile and device.type == "mps":
                logger.info("   ⚠️ torch.compile not supported on MPS, skipping")

            if device.type == "cpu":
                logger.info("✅ Embedding model loaded on CPU (saves VRAM, stays in RAM)")
            elif device.type == "mps":
                logger.info("✅ Embedding model loaded and optimized (stays in MPS memory)")
            else:
                logger.info("✅ Embedding model loaded and optimized (stays in VRAM)")

            self._embedding_model = embedding_model
            return embedding_model

        except Exception as e:
            logger.warning(f"Failed to load optimized embedding model: {e}. Will use default.")
            return None
