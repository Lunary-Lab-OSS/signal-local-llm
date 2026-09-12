"""
SOTA Router Strategy

Implementation of the Single-Tower Scalar Regression router strategy.
Leverages torchao for sub-millisecond inference and Interaction Distillation.
"""

import logging
from pathlib import Path
from typing import Any

import torch

from signal_llm.config import LLMConfig

from .base import RouterStrategy
from .model import SingleTowerStudent

logger = logging.getLogger(__name__)


class SotaRouterController:
    """
    Controller for the SOTA router.
    Mimics the interface needed by IntentRouter but uses SingleTowerStudent.
    """

    def __init__(self, model: SingleTowerStudent):
        self.model = model
        self.routers = {"sota": self}  # Self-reference to mimic structure if needed

    def calculate_strong_win_rate(self, prompt: str) -> float:
        """
        Calculate difficulty score (win rate equivalent).
        Returns a float between 0.0 and 1.0.
        """
        self.model.eval()
        with torch.no_grad():
            # Model returns a raw scalar score (logits)
            # Apply sigmoid to map to 0-1 probability
            score_logits = self.model([prompt])
            prob = torch.sigmoid(score_logits).item()
        return prob


class SotaRouterStrategy(RouterStrategy):
    """
    SOTA router strategy implementation using SingleTowerStudent.
    """

    def __init__(self, config: LLMConfig, device: str, platform: str):
        super().__init__(config, device, platform)
        self.model: SingleTowerStudent | None = None
        self.models_dir: Path | None = None  # Will be set by factory if available

    def load_controller(self) -> Any:
        """
        Load the SOTA router model and wrap it in a controller.
        """
        try:
            # Get model ID from config, default to GIST-small if not specified
            # The config might pass specific router settings
            # Default to google/embeddinggemma-300m to match the trained checkpoint
            # If a different model is specified in config, use that instead
            model_id = self.config.routellm_embedding_model or "google/embeddinggemma-300m"

            logger.info(f"Initializing SOTA Single-Tower Router with model: {model_id}")

            # Determine device
            if self.device == "cuda" and torch.cuda.is_available():
                device_obj = torch.device("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                device_obj = torch.device("mps")
            else:
                device_obj = torch.device("cpu")

            # Initialize Student Model
            # INT4 quantization works on both CPU and CUDA, but is most efficient on CUDA
            # For CPU, FP32 or FP16 may actually be faster than INT4
            # We'll use INT4 if explicitly requested or if CUDA is available
            use_int4 = device_obj.type == "cuda"  # Default to INT4 on CUDA, FP32/FP16 on CPU/MPS

            # Get HF token from environment (set by main app or evaluate script)
            # The token should already be set in environment variables by the main app
            import os

            hf_token = (
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_TOKEN")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
            )

            self.model = SingleTowerStudent(
                model_id=model_id,
                use_int4=use_int4,
                device=device_obj,
                dtype=self.config.routellm_embedding_dtype,  # Pass dtype from config
                hf_token=hf_token,  # Pass HF token for gated models
            )

            # Load checkpoint if exists (The "head" weights)
            # Prefer quantized model if available, fallback to non-quantized
            checkpoint_path = self.config.routellm_checkpoint_path

            # If checkpoint_path is relative and models_dir is available, resolve it
            if checkpoint_path and self.models_dir:
                checkpoint_path_obj = Path(checkpoint_path)
                if not checkpoint_path_obj.is_absolute():
                    # Try to find model in models_dir based on embedding model name
                    model_name = model_id.replace("/", "_").replace("-", "_")
                    # Try quantized first (if use_int4), then non-quantized
                    if use_int4:
                        quant_path = self.models_dir / f"sota_router_{model_name}_int4.pt"
                        if quant_path.exists():
                            checkpoint_path = str(quant_path)
                            logger.info(f"Found quantized model in models_dir: {checkpoint_path}")
                        else:
                            # Fallback to non-quantized
                            non_quant_path = self.models_dir / f"sota_router_{model_name}_fp32.pt"
                            if non_quant_path.exists():
                                checkpoint_path = str(non_quant_path)
                                logger.info(
                                    f"Found non-quantized model in models_dir: {checkpoint_path}"
                                )
                    else:
                        # CPU/MPS: prefer non-quantized, but allow INT4 if that's all we have
                        non_quant_path = self.models_dir / f"sota_router_{model_name}_fp32.pt"
                        quant_path = self.models_dir / f"sota_router_{model_name}_int4.pt"
                        if non_quant_path.exists():
                            checkpoint_path = str(non_quant_path)
                            logger.info(
                                f"Found non-quantized model in models_dir: {checkpoint_path}"
                            )
                        elif quant_path.exists():
                            checkpoint_path = str(quant_path)
                            logger.warning(
                                f"⚠️  Using INT4 model on {device_obj.type.upper()}: Works but may be slower than FP32"
                            )
                            logger.info(f"Found quantized model in models_dir: {checkpoint_path}")
                            use_int4 = True  # Allow INT4 on CPU if that's what we have
                else:
                    # Absolute path, use as-is
                    checkpoint_path_obj = Path(checkpoint_path)
                    if not checkpoint_path_obj.exists() and self.models_dir:
                        # Try to find in models_dir
                        model_name = model_id.replace("/", "_").replace("-", "_")
                        if use_int4:
                            quant_path = self.models_dir / f"sota_router_{model_name}_int4.pt"
                            if quant_path.exists():
                                checkpoint_path = str(quant_path)
                                logger.info(
                                    f"Checkpoint not found at {checkpoint_path}, using models_dir: {checkpoint_path}"
                                )

            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Loading router checkpoint from {checkpoint_path}")
                try:
                    # Load state dict - checkpoint should only contain head weights (not quantized backbone)
                    # The backbone is quantized in-place, so we only save/load the trainable head
                    state_dict = torch.load(
                        checkpoint_path, map_location=device_obj, weights_only=True
                    )

                    # Filter to only head weights (avoid quantized backbone tensors)
                    # Head weights are in 'head.' namespace, but model.head expects keys without 'head.' prefix
                    head_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("head."):
                            # Strip 'head.' prefix for loading into model.head
                            new_key = k[5:]  # Remove 'head.' prefix
                            head_state_dict[new_key] = v

                    if head_state_dict:
                        # Only load head weights (these are FP32, not quantized)
                        self.model.head.load_state_dict(head_state_dict, strict=False)
                        logger.info(
                            f"✅ Checkpoint loaded successfully (head weights only, {len(head_state_dict)} keys)"
                        )
                    else:
                        # Try loading full state dict if no head prefix found (backward compatibility)
                        # But skip quantized tensors that cause shallow_copy errors
                        filtered_dict = {}
                        for k, v in state_dict.items():
                            # Skip AffineQuantizedTensor and other problematic types
                            if hasattr(v, "__class__") and "AffineQuantizedTensor" in str(type(v)):
                                logger.debug(f"Skipping quantized tensor: {k}")
                                continue
                            filtered_dict[k] = v

                        if filtered_dict:
                            self.model.load_state_dict(filtered_dict, strict=False)
                            logger.info("✅ Checkpoint loaded successfully (filtered state dict)")
                        else:
                            logger.warning(
                                "No loadable weights found in checkpoint (all quantized?)"
                            )
                except Exception as e:
                    error_str = str(e)
                    if "shallow_copy" in error_str or "AffineQuantizedTensor" in error_str:
                        # Try loading only head weights as fallback
                        try:
                            state_dict = torch.load(
                                checkpoint_path, map_location=device_obj, weights_only=True
                            )
                            # Strip 'head.' prefix for loading into model.head
                            head_state_dict = {}
                            for k, v in state_dict.items():
                                if k.startswith("head."):
                                    head_state_dict[k[5:]] = v  # Remove 'head.' prefix
                            if head_state_dict:
                                self.model.head.load_state_dict(head_state_dict, strict=False)
                                logger.info(
                                    f"✅ Checkpoint loaded successfully (head weights only, fallback, {len(head_state_dict)} keys)"
                                )
                            else:
                                logger.warning(
                                    "Checkpoint contains quantized tensors that can't be loaded. Using untrained head."
                                )
                        except Exception as e2:
                            logger.warning(f"Failed to load checkpoint even with fallback: {e2}")
                    else:
                        logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            else:
                logger.warning(
                    "No checkpoint provided/found. Using initialized weights (untrained head)."
                )

            self.controller = SotaRouterController(self.model)
            logger.info("✅ SOTA Router Controller initialized")
            return self.controller

        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to initialize SOTA router: {e}") from e
