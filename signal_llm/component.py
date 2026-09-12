"""
LLM Component
Language model loading and generation with platform-specific optimizations.
"""

import logging
import re
from typing import Any

from .loader import LLMLoader

logger = logging.getLogger(__name__)


# Name markers used as a fallback when no tokenizer is available to ask
# whether a model exposes a chat template. Keep in sync with the model
# families used in ModelConfig priority lists.
_CHAT_MODEL_MARKERS = ("qwen", "chat", "instruct", "olmo", "think")


def _is_chat_model(model_name: str, tokenizer: Any = None) -> bool:
    """Return True when the model should receive chat-template formatting.

    Prefers the tokenizer's own ``chat_template`` attribute, which is the
    authoritative signal transformers exposes; falls back to documented
    name markers when no tokenizer is available.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return True
    lowered = (model_name or "").lower()
    return any(marker in lowered for marker in _CHAT_MODEL_MARKERS)


class LLMComponent:
    """
    LLM component with optimized settings for router and agent models.
    Handles model loading and generation with platform-specific optimizations.
    """

    def __init__(self, config, model_loader=None, device: str = "cpu", platform: str = "linux"):
        self.raw_config = config
        self.config = getattr(config, "llm", config)
        self.model_loader = model_loader or self._build_default_loader(config, device)
        self.device = device
        self.platform = platform
        self.router_model: Any | None = None
        self.semantic_model: Any | None = None
        self.agent_model: Any | None = None
        self._router_tokenizer: Any | None = None
        self._semantic_tokenizer: Any | None = None
        self._agent_tokenizer: Any | None = None
        self._router_model_name: str | None = None
        self._semantic_model_name: str | None = None
        self._agent_model_name: str | None = None
        self._optimized_settings = self._get_optimized_settings()

    @staticmethod
    def _build_default_loader(config, device: str) -> LLMLoader:
        models_dir = getattr(config, "models_dir", None)
        cache_dir = getattr(config, "cache_dir", None)
        if models_dir is None or cache_dir is None:
            raise ValueError(
                "model_loader is required unless config exposes models_dir and cache_dir"
            )
        return LLMLoader(
            models_dir=models_dir,
            cache_dir=cache_dir,
            device=device,
            system_config=getattr(config, "system", None),
        )

    def _get_optimized_settings(self) -> dict:
        """Get optimized LLM settings for the platform"""
        if self.device.startswith("cuda"):
            # CUDA (Windows) - optimized for RTX 3090/4090
            return {
                "router": {
                    "max_tokens": 10,  # Fast classification
                    "temperature": 0.1,  # Low temperature for deterministic routing
                    "top_p": 0.9,
                    "top_k": 40,
                },
                "agent": {
                    # Keep generations very short and low-temperature for voice responses
                    # to avoid gibberish and reduce latency. The global llm.max_tokens
                    # config will further cap this.
                    "max_tokens": 256,  # Increased slightly but still reasonable
                    "temperature": 0.3,  # Slightly higher for more natural responses
                    "top_p": 0.9,
                    "top_k": 40,
                },
            }
        elif self.device == "mps":
            # MPS (macOS) - optimized for Apple Silicon
            return {
                "router": {
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                },
                "agent": {
                    "max_tokens": 128,  # Lower for MPS memory constraints
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 40,
                },
            }
        else:
            # CPU fallback
            return {
                "router": {
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 20,
                },
                "agent": {
                    "max_tokens": 128,  # Very limited on CPU
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 20,
                },
            }

    def load_router_model(self):
        """Load fast router model from priority list (only for LLM-based routing)"""
        # Skip loading router model if using RouteLLM (neural matrix factorization)
        router_type = getattr(self.config, "router_type", "llm")
        if router_type == "routellm":
            logger.info(
                "⏭️  Router model loading skipped - using RouteLLM (neural matrix factorization) for routing"
            )
            return

        if self.router_model is not None:
            logger.debug(f"Router model already loaded: {self._router_model_name}")
            return

        logger.info(f"Loading router LLM for {self.platform} on {self.device}...")
        try:
            result, model_name = self.model_loader.load_llm_model(self.config.router_priority)
            self._router_model_name = model_name

            # Handle different return types
            if isinstance(result, tuple):
                self.router_model, self._router_tokenizer = result
            else:
                self.router_model = result
                self._router_tokenizer = None

            logger.info(f"✅ Router model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load router model: {e}")
            raise

    def load_semantic_model(self):
        """Load semantic model, reusing agent model if same repo_id"""
        # Check if semantic model loading is disabled
        if not self.config.semantic_enabled:
            logger.info("⏭️  Semantic model loading disabled in config")
            return

        if self.semantic_model is not None:
            logger.debug(f"Semantic model already loaded: {self._semantic_model_name}")
            return

        logger.info(f"Loading semantic LLM for {self.platform} on {self.device}...")
        try:
            # Check if semantic and agent models have the same repo_id
            semantic_config = (
                self.config.semantic_priority[0] if self.config.semantic_priority else None
            )
            agent_config = self.config.agent_priority[0] if self.config.agent_priority else None

            if semantic_config and agent_config and semantic_config.repo_id == agent_config.repo_id:
                # Same model - reuse agent model instance
                logger.info(
                    f"Semantic model ({semantic_config.repo_id}) same as agent model - reusing instance"
                )
                if self.agent_model is None:
                    self.load_agent_model()
                self.semantic_model = self.agent_model
                self._semantic_tokenizer = self._agent_tokenizer
                self._semantic_model_name = self._agent_model_name
                logger.info(
                    f"✅ Semantic model reused from agent model: {self._semantic_model_name}"
                )
                return

            # Different model - load separately
            result, model_name = self.model_loader.load_llm_model(self.config.semantic_priority)
            self._semantic_model_name = model_name

            # Handle different return types
            if isinstance(result, tuple):
                self.semantic_model, self._semantic_tokenizer = result
            else:
                self.semantic_model = result
                self._semantic_tokenizer = None

            logger.info(f"✅ Semantic model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            raise

    def load_agent_model(self):
        """Load agent model from priority list with speculative decoding if router model is available"""
        # Check if agent model loading is disabled
        if not self.config.agent_enabled:
            logger.info("⏭️  Agent model loading disabled in config")
            return

        if self.agent_model is not None:
            logger.debug(f"Agent model already loaded: {self._agent_model_name}")
            return

        logger.info(f"Loading agent LLM for {self.platform} on {self.device}...")
        try:
            # Check if speculative decoding is enabled
            spec_decode_enabled = (
                self.config.speculative_decoding and self.config.speculative_decoding.enabled
            )

            # Load router model first if speculative decoding is enabled
            draft_model_path = None
            if spec_decode_enabled and self.router_model is None:
                try:
                    self.load_router_model()
                except Exception as e:
                    logger.warning(f"Router model not available for speculative decoding: {e}")

            if spec_decode_enabled and self.router_model is not None:
                # Get router model path for speculative decoding
                router_config = self.config.router_priority[0]
                draft_model_path = self.model_loader._get_model_path(router_config.repo_id)
                logger.info(
                    f"Using router model ({router_config.name}) as draft for speculative decoding: {draft_model_path}"
                )

            result, model_name = self.model_loader.load_llm_model(
                self.config.agent_priority, draft_model_path=draft_model_path
            )
            self._agent_model_name = model_name

            # Handle different return types
            if isinstance(result, tuple):
                self.agent_model, self._agent_tokenizer = result
            else:
                self.agent_model = result
                self._agent_tokenizer = None

            # If semantic model should reuse this, set it up
            semantic_config = (
                self.config.semantic_priority[0] if self.config.semantic_priority else None
            )
            agent_config = self.config.agent_priority[0] if self.config.agent_priority else None
            if semantic_config and agent_config and semantic_config.repo_id == agent_config.repo_id:
                self.semantic_model = self.agent_model
                self._semantic_tokenizer = self._agent_tokenizer
                self._semantic_model_name = self._agent_model_name
                logger.info("✅ Semantic model will reuse agent model instance")

            logger.info(f"✅ Agent model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load agent model: {e}")
            raise

    def load_all_models(self):
        """Load router, semantic, and agent models"""
        self.load_router_model()
        self.load_semantic_model()
        self.load_agent_model()

    def generate_router(self, prompt: str) -> str:
        """Generate with router model (fast, for classification)"""
        if self.router_model is None:
            self.load_router_model()

        if self.router_model is None:
            logger.warning("Router model not loaded")
            return ""

        try:
            settings = dict(self._optimized_settings["router"])
            # Respect global max token cap if present in config
            max_cfg = getattr(self.config, "max_tokens", None)
            if isinstance(max_cfg, int):
                settings["max_tokens"] = min(settings["max_tokens"], max_cfg)
            # Cap at reasonable limit to prevent over-generation
            settings["max_tokens"] = min(settings["max_tokens"], 50)
            # Get thinking tokens from config (0 for fast mode)
            thinking_tokens = self.config.router_thinking_tokens
            return self._generate(self.router_model, prompt, settings, thinking_tokens)
        except Exception as e:
            logger.error(f"Router generation failed: {e}")
            return ""

    def generate_semantic(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> str:
        """Generate with semantic model (low difficulty, no thinking)"""
        if self.semantic_model is None:
            self.load_semantic_model()

        if self.semantic_model is None:
            logger.warning("Semantic model not loaded")
            return ""

        try:
            settings = dict(self._optimized_settings["agent"])  # Use agent settings as base
            if max_tokens is not None:
                # Explicit override (cleanup/rewrite paths need room for a full prompt)
                settings["max_tokens"] = max_tokens
            else:
                # Respect global max token cap if present in config
                max_cfg = getattr(self.config, "max_tokens", None)
                if isinstance(max_cfg, int):
                    settings["max_tokens"] = min(settings["max_tokens"], max_cfg)
                # Cap at reasonable limit to prevent over-generation and repetition
                # Voice responses should be concise - 300 tokens is plenty
                settings["max_tokens"] = min(settings["max_tokens"], 300)
            # Semantic model uses no thinking tokens
            thinking_tokens = 0
            return self._generate(
                self.semantic_model, prompt, settings, thinking_tokens, system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"Semantic generation failed: {e}")
            return ""

    def generate_agent(self, prompt: str, thinking_tokens: int | None = None) -> str:
        """
        Generate with agent model, optionally with dynamic thinking budget.

        Args:
            prompt: Input prompt
            thinking_tokens: Optional thinking token budget (from cascading router)
                           If None, uses config default (agent_thinking_tokens)
        """
        if self.agent_model is None:
            self.load_agent_model()

        if self.agent_model is None:
            logger.warning("Agent model not loaded")
            return ""

        try:
            settings = dict(self._optimized_settings["agent"])
            # Use max_new_tokens from config if available (for phi-4-reasoning)
            max_new_tokens_cfg = getattr(self.config, "max_new_tokens", None)
            if isinstance(max_new_tokens_cfg, int):
                settings["max_new_tokens"] = max_new_tokens_cfg
            # Respect global max token cap if present in config
            max_cfg = getattr(self.config, "max_tokens", None)
            if isinstance(max_cfg, int):
                settings["max_tokens"] = min(settings["max_tokens"], max_cfg)
            # Cap at reasonable limit to prevent over-generation and repetition
            # Voice responses should be concise - 300 tokens is plenty
            settings["max_tokens"] = min(settings["max_tokens"], 300)

            # Check if model is phi-4-reasoning (handles thinking tokens automatically)
            is_phi4_reasoning = (
                "phi-4" in (self._agent_model_name or "").lower()
                and "reasoning" in (self._agent_model_name or "").lower()
            )

            # For phi-4-reasoning, don't pass thinking tokens (it handles them automatically)
            if is_phi4_reasoning:
                thinking_tokens = 0  # phi-4 handles thinking tokens internally
            else:
                # Use dynamic thinking tokens if provided, otherwise fall back to config
                if thinking_tokens is None:
                    thinking_tokens = self.config.agent_thinking_tokens

            return self._generate(self.agent_model, prompt, settings, thinking_tokens)
        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            return ""

    def _generate(
        self,
        model: Any,
        prompt: str,
        settings: dict,
        thinking_tokens: int = 0,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using model with optimized settings and thinking tokens"""
        # Get model name for template detection
        model_name = ""
        if model == self.router_model:
            model_name = self._router_model_name or ""
        elif model == self.agent_model:
            model_name = self._agent_model_name or ""

        # Prepare formatted prompt
        formatted_prompt = prompt

        # Check for ExLlamaV2DynamicGenerator
        is_exllamav2 = "ExLlamaV2DynamicGenerator" in str(type(model))

        # Retrieve tokenizer if available
        tokenizer = None
        model_obj = model

        if isinstance(model, tuple) and len(model) == 2:
            # Handle tuple format (for backwards compatibility)
            model_obj, tokenizer = model
        elif (
            model == self.router_model
            and hasattr(self, "_router_tokenizer")
            and self._router_tokenizer is not None
        ):
            tokenizer = self._router_tokenizer
        elif (
            model == self.agent_model
            and hasattr(self, "_agent_tokenizer")
            and self._agent_tokenizer is not None
        ):
            tokenizer = self._agent_tokenizer
        elif (
            model == self.semantic_model
            and hasattr(self, "_semantic_tokenizer")
            and self._semantic_tokenizer is not None
        ):
            tokenizer = self._semantic_tokenizer

        # Handle MLX-LM models (Apple Silicon). Detect by module namespace.
        if type(model).__module__.split(".")[0] in ("mlx", "mlx_lm"):
            from mlx_lm import generate as mlx_generate

            formatted_prompt = prompt
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt or "You are a concise voice assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(f"MLX: could not apply chat template: {e}")

            max_new = settings.get("max_new_tokens", settings["max_tokens"]) + (
                thinking_tokens or 0
            )
            try:
                text = mlx_generate(
                    model,
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_new,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"MLX generation failed: {e}")
                return ""

            if "</think>" in text:
                text = text.rsplit("</think>", 1)[-1]
            elif "<think>" in text:
                text = ""
            import re as _re

            text = _re.sub(r"</?think>", "", text).strip()
            return text

        # Handle vLLM (fastest for Windows CUDA)
        if hasattr(model, "generate") and hasattr(model, "llm_engine"):
            # vLLM model
            from vllm import SamplingParams

            # Apply chat template for vLLM if it's a chat model
            # vLLM usually handles this via tokenizer, but we need to pass formatted prompt
            is_chat_model = _is_chat_model(model_name)
            # Basic ChatML for Qwen/Chat models if raw prompt passed
            if is_chat_model and not prompt.startswith("<|im_start|>"):
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            # Use max_new_tokens if available (for phi-4-reasoning), otherwise use max_tokens
            max_tokens_param = settings.get("max_new_tokens", settings["max_tokens"])

            sampling_params = SamplingParams(
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings.get("top_k", 40),
                max_tokens=max_tokens_param,
            )

            # For OLMo-Think, thinking tokens are handled via the model's internal reasoning
            # vLLM will automatically handle the thinking tokens
            outputs = model.generate([formatted_prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()

        # Handle ExLlamaV2 (ExLlamaV2DynamicGenerator)
        if is_exllamav2:
            # 1. Prompt formatting
            formatted_prompt = prompt

            # Apply strict ChatML template with thinking logic
            if tokenizer:
                try:
                    messages = [
                        {"role": "system", "content": "You are a concise voice assistant."},
                        {"role": "user", "content": prompt},
                    ]

                    # Qwen 3 logic:
                    # Router (0 tokens) -> disable thinking (fastest command mode)
                    # Agent (512+ tokens) -> enable thinking (reasoning mode)
                    enable_thinking = thinking_tokens > 0

                    try:
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        )
                    except TypeError:
                        # Fallback for specific HF versions without enable_thinking param
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                except Exception as e:
                    logger.warning(f"Could not apply chat template: {e}")
                    # Fallback manual formatting
                    if not prompt.startswith("<|im_start|>"):
                        formatted_prompt = f"<|im_start|>system\nYou are a concise voice assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                # Manual formatting fallback
                is_chat_model = _is_chat_model(model_name)
                if is_chat_model and not prompt.startswith("<|im_start|>"):
                    formatted_prompt = (
                        f"<|im_start|>system\nYou are a concise voice assistant.<|im_end|>\n"
                        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

            # 2. Settings
            # Dynamic import to avoid circular dependency or import error if not installed
            from exllamav2.generator import ExLlamaV2Sampler

            gen_settings = ExLlamaV2Sampler.Settings()

            # Use stricter settings for router (low temp)
            gen_settings.temperature = settings["temperature"] if thinking_tokens > 0 else 0.4
            gen_settings.top_p = settings["top_p"]
            gen_settings.top_k = settings.get("top_k", 40)
            # Higher repetition penalty to prevent loops - tuned to stop repetition at source
            gen_settings.token_repetition_penalty = 1.2
            # Add frequency penalty to discourage repeating recent tokens
            gen_settings.token_frequency_penalty = 0.1

            # 3. Stop conditions & Thinking buffer
            stop_conditions = ["<|im_end|>", "<|endoftext|>"]
            # For phi-4-reasoning, use max_new_tokens directly (it handles thinking tokens internally)
            # Otherwise, add thinking_tokens to max_tokens
            if "phi-4" in model_name.lower() and "reasoning" in model_name.lower():
                max_new = settings.get("max_new_tokens", settings["max_tokens"])
            else:
                max_new = settings.get("max_new_tokens", settings["max_tokens"]) + thinking_tokens

            # CRITICAL FIX: Qwen3 models should NOT have BOS token added
            # This was causing gibberish output. Qwen3 doesn't use BOS tokens.
            is_qwen3 = "qwen3" in model_name.lower()

            output = model.generate(
                prompt=formatted_prompt,
                max_new_tokens=max_new,
                gen_settings=gen_settings,
                stop_conditions=stop_conditions,
                completion_only=True,  # Strips the input prompt
                encode_special_tokens=True,  # Essential for Qwen 3 control tokens
                add_bos=not is_qwen3,  # Qwen3: False (no BOS), others: True (default)
            )

            # 4. Post-processing (Strip Thoughts)
            # If the model thought, it produced: "<think> ... </think> Answer"
            if thinking_tokens > 0:
                # Regex to remove the think block (non-greedy)
                clean_output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
            else:
                clean_output = output.strip()

            return clean_output

        # Handle Transformers model - check if we have a tokenizer stored separately
        # Models are stored unpacked (self.agent_model, self._agent_tokenizer) not as tuples
        if tokenizer is not None:
            # Transformers model with tokenizer
            import torch

            # OLMo-Think automatically handles reasoning with <think> tags internally
            # No need to manually add thinking tokens - the model does it automatically

            # Use chat template
            if _is_chat_model(model_name, tokenizer):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(f"Could not apply chat template: {e}")
                    formatted_prompt = prompt

            inputs = tokenizer(formatted_prompt, return_tensors="pt", return_token_type_ids=False)
            # For quantized models (bitsandbytes), explicitly move to CUDA
            if hasattr(model_obj, "device"):
                inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}
            elif torch.cuda.is_available():
                # For quantized models, use inputs.input_ids.to('cuda') as recommended
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                # For phi-4-reasoning, use max_new_tokens directly (it handles thinking tokens internally)
                # Otherwise, use max_tokens
                max_new_tokens_param = settings.get("max_new_tokens", settings["max_tokens"])

                outputs = model_obj.generate(
                    **inputs,  # Unpack inputs dict
                    max_new_tokens=max_new_tokens_param,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    top_k=settings.get("top_k", 50),  # OLMo uses top_k=50
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (skip the input prompt).
            # Note: This simple removal might fail if the prompt was formatted differently than decode
            # Better to use the length of input_ids
            input_len = inputs["input_ids"].shape[1]
            generated_only = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

            return generated_only.strip()

        # Check if it's ExLlamaV2DynamicGenerator
        # This block is now redundant as ExLlamaV2 is handled above, but kept as safety fallback
        # for any cases that might have slipped through without tokenizer
        if is_exllamav2:
            # Manual formatting for ExLlamaV2 if needed
            is_chat_model = _is_chat_model(model_name)
            if is_chat_model and not prompt.startswith("<|im_start|>"):
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            # Dynamic import to avoid circular dependency or import error if not installed
            from exllamav2.generator import ExLlamaV2Sampler

            gen_settings = ExLlamaV2Sampler.Settings()
            gen_settings.temperature = settings["temperature"]
            gen_settings.top_p = settings["top_p"]
            gen_settings.top_k = settings.get("top_k", 40)
            gen_settings.token_repetition_penalty = 1.05  # Reduced from 1.15 to avoid artifacts
            # gen_settings.min_p = 0.05  # Removed min_p for stability

            # Generate with stop tokens
            stop_conditions = ["<|im_end|>", "<|endoftext|>"]

            # CRITICAL FIX: Qwen3 models should NOT have BOS token added
            # This was causing gibberish output. Qwen3 doesn't use BOS tokens.
            is_qwen3 = "qwen3" in model_name.lower()
            add_bos = not is_qwen3  # Only add BOS for non-Qwen3 models

            # For phi-4-reasoning, use max_new_tokens directly (it handles thinking tokens internally)
            # Otherwise, use max_tokens
            max_new_tokens_param = settings.get("max_new_tokens", settings["max_tokens"])

            output = model.generate(
                formatted_prompt,
                max_new_tokens=max_new_tokens_param,
                gen_settings=gen_settings,
                stop_conditions=stop_conditions,
                add_bos=add_bos,  # Qwen3: False, others: True
            )

            # ExLlamaV2 returns the full text including prompt
            # We need to strip the prompt from the output
            if output.startswith(formatted_prompt):
                output = output[len(formatted_prompt) :]

            return output.strip()

        elif hasattr(model, "settings"):
            # Legacy ExLlamaV2 style
            model.settings.temperature = settings["temperature"]
            model.settings.top_p = settings["top_p"]
            model.settings.top_k = settings.get("top_k", 40)
            return model.generate(prompt, max_new_tokens=settings["max_tokens"])
        elif hasattr(model, "generate"):
            # MLX-LM style (or other models with generate method)
            # Note: Transformers models also have generate, but they should be caught above by tokenizer check
            return model.generate(
                prompt,
                max_tokens=settings["max_tokens"],
                temp=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings.get("top_k", 40),
            )
        else:
            # Generic fallback
            logger.warning(f"Unknown model type: {type(model)}")
            return ""
