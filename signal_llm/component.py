"""
LLM Component

Language model loading and generation with platform-specific optimizations.

Contract (remediation L06-L12, L16, L19-L21):

- Each role (router/semantic/agent) is loaded independently behind its own
  enablement gate. Sharing happens only between *actually loaded* handles
  whose identity (source repo, revision, backend, device, quantization,
  tokenizer) matches the semantic request — never by comparing configured
  first choices (L06).
- A loaded handle is a :class:`LoadedModel` record carrying its identity
  and tokenizer, so template/BOS decisions use real metadata instead of
  display-name guessing (L10).
- ``system_prompt`` is honoured by every backend that supports chat
  templates; backends without templates use an explicit, documented
  ChatML formatting rule (L09).
- Token budgets enforce only a total new-token cap: the answer allowance
    plus the thinking allowance, bounded by ``max_tokens``. Separate visible
    answer/reasoning caps are not enforced by these backend APIs.
- Reasoning extraction is shared across backends: complete ``<think>``
  blocks are always removed; an unterminated opening tag means the model
  was still reasoning, and the visible answer is empty rather than leaking
  raw chain-of-thought into voice output (L12).
- Failures raise :class:`GenerationError`; an empty completion is returned
  as ``""`` and is distinguishable from an error (L19).
"""

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate

from .loader import LLMLoader, ModelLoadError, parse_device

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = "You are a concise voice assistant."


#: Sentinel distinguishing "caller did not choose a system prompt" (use the
#: default) from an explicit ``None`` (no system block at all) — L09.
class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()

_THINK_TAG_RE = re.compile(r"</?think>")


class GenerationError(RuntimeError):
    """Raised when generation fails or the model backend is unusable."""


def extract_answer(raw_output: str, *, had_thinking: bool) -> str:
    """Return the visible answer from raw model output (L12).

    - Complete ``<think>...</think>`` blocks are removed wherever they
      appear, regardless of the configured budget.
    - An unterminated ``<think>`` opener means the model was still
      reasoning — whether or not reasoning was requested — so the visible
      answer is empty; partial chain-of-thought never reaches output.
        - ``had_thinking`` means the prompt ends inside a reasoning block,
            NOT merely that a positive thinking allowance was requested. Text
            before its closing delimiter is suppressed, including incomplete output.
    """
    if not isinstance(raw_output, str):
        raise GenerationError("backend returned non-string output")
    thinking = int(had_thinking)
    answer: list[str] = []
    position = 0
    for match in _THINK_TAG_RE.finditer(raw_output):
        if match.group() == "</think>" and not thinking:
            answer.clear()
            position = match.end()
            continue
        if not thinking:
            answer.append(raw_output[position : match.start()])
        if match.group() == "<think>":
            if not (had_thinking and match.start() == 0):
                thinking += 1
        else:
            thinking -= 1
        position = match.end()
    if thinking:
        return ""
    answer.append(raw_output[position:])
    text = "".join(answer)
    if any(text.endswith(tag[:n]) for tag in ("<think>", "</think>") for n in range(1, len(tag))):
        return ""
    return text.strip()


def _synchronized[**P, R](
    method: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    @wraps(method)
    def locked(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            if self._closed:
                raise GenerationError("component is closed")
            return method(self, *args, **kwargs)

    return locked


def _prefilled_thinking(formatted: str) -> bool:
    """Recognize an open reasoning delimiter at the generation boundary only."""
    return formatted.rstrip().endswith("<think>")


@dataclass
class LoadedModel:
    """A loaded backend handle plus the metadata needed to use it safely."""

    handle: Any
    name: str
    backend: str
    tokenizer: Any = None
    source_repo: str | None = None
    revision: str | None = None
    device: str | None = None
    quantized: bool = False
    local_path: str | None = None
    model_path: Path | None = None
    dtype: str = "float16"
    trust_remote_code: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> tuple:
        """Identity used for safe model sharing between roles (L06)."""
        return (
            self.source_repo,
            self.revision,
            self.backend,
            self.device,
            self.quantized,
            self.local_path,
            self.dtype,
            self.trust_remote_code,
        )

    def matches_request(self, requested, default_device=None) -> bool:
        """True when this loaded handle satisfies the requested ModelConfig."""
        requested_device = default_device if requested.device == "auto" else requested.device
        if requested_device == "cuda" and self.device != "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    return False
                requested_device = f"cuda:{torch.cuda.current_device()}"
            except (ImportError, RuntimeError, AssertionError):
                return False
        return (
            self.device not in (None, "auto")
            and self.source_repo == requested.repo_id
            and self.revision == requested.revision
            and self.backend == (getattr(requested, "backend", "") or "").strip().lower()
            and self.device == requested_device
            and self.quantized == bool(getattr(requested, "quantized", False))
            and self.local_path
            == (str(Path(requested.local_path).resolve()) if requested.local_path else None)
            and self.dtype == requested.dtype
            and self.trust_remote_code == requested.trust_remote_code
            and not self.extras.get("draft_model_path")
        )


def _is_chat_model(model_name: str, tokenizer: Any = None) -> bool:
    """Return True when the model should receive chat-template formatting.

    The tokenizer's own ``chat_template`` attribute is authoritative; the
    name markers are a documented fallback only when no tokenizer exists
    (L10).
    """
    if tokenizer is not None:
        if getattr(tokenizer, "chat_template", None):
            return True
        # A tokenizer that explicitly has no template is authoritative too.
        if hasattr(tokenizer, "chat_template"):
            return False
    lowered = (model_name or "").lower()
    return any(marker in lowered for marker in ("qwen", "chat", "instruct", "olmo", "think"))


class LLMComponent:
    """LLM component with optimized settings for router and agent models."""

    def __init__(self, config, model_loader=None, device: str = "cpu", platform: str = "linux"):
        self.raw_config = config
        self.config = getattr(config, "llm", config)
        self.device = device
        self.platform = platform
        self._lock = threading.RLock()
        self._closed = False
        if model_loader is not None:
            self.model_loader = model_loader
        else:
            self.model_loader = self._build_default_loader(config, device)
        self.router_model: LoadedModel | None = None
        self.semantic_model: LoadedModel | None = None
        self.agent_model: LoadedModel | None = None
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

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #

    def _get_optimized_settings(self) -> dict:
        kind, _index = parse_device(self.device)
        if kind == "cuda":
            base = {"router_max": 10, "agent_max": 256, "temperature": 0.3, "top_k": 40}
        elif kind == "mps":
            base = {"router_max": 10, "agent_max": 128, "temperature": 0.2, "top_k": 40}
        else:
            base = {"router_max": 10, "agent_max": 128, "temperature": 0.2, "top_k": 20}
        return {
            "router": {
                "max_tokens": base["router_max"],
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": base["top_k"],
            },
            "agent": {
                "max_tokens": base["agent_max"],
                "temperature": base["temperature"],
                "top_p": 0.9,
                "top_k": base["top_k"],
            },
        }

    def _resolve_budget(
        self,
        role: str,
        *,
        answer_override: int | None = None,
        thinking_tokens: int,
    ) -> dict:
        """Resolve one generation budget contract (L11).

        ``min(answer_tokens + thinking_tokens, hard_total)`` is the sole
        enforced completion cap. The split controls template preference and
        total allowance, not distinct reasoning/visible-answer limits.
        Zero total returns an empty completion without invoking the backend.
        Sampler values come from the platform-tuned role settings, with the
        explicit ``config.temperature`` / ``config.top_p`` taking precedence
        (L16).
        """
        # Semantic generation uses the agent profile as its base.
        settings = dict(self._optimized_settings.get(role) or self._optimized_settings["agent"])

        config_temperature = getattr(self.config, "temperature", None)
        if isinstance(config_temperature, (int, float)):
            settings["temperature"] = float(config_temperature)
        config_top_p = getattr(self.config, "top_p", None)
        if isinstance(config_top_p, (int, float)):
            settings["top_p"] = float(config_top_p)

        hard_total = int(getattr(self.config, "max_tokens", 2048))
        max_new_cfg = getattr(self.config, "max_new_tokens", None)

        answer = answer_override if answer_override is not None else settings["max_tokens"]
        for value in (answer, thinking_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GenerationError("token allowances must be non-negative integers")
        answer = min(answer, hard_total, 300)
        if max_new_cfg is not None and answer_override is None:
            answer = min(answer, int(max_new_cfg))
        if answer < 0:
            raise GenerationError("answer token budget must be non-negative")
        # The hard total caps answer + reasoning allowance together (L11):
        # a huge thinking request can never push generation past max_tokens.
        thinking = max(0, int(thinking_tokens))
        thinking = min(thinking, max(0, hard_total - answer))
        return {
            **settings,
            "answer_tokens": answer,
            "thinking_tokens": thinking,
            "hard_total": hard_total,
        }

    # ------------------------------------------------------------------ #
    # Loading (L06/L07)
    # ------------------------------------------------------------------ #

    @_synchronized
    def load_router_model(self):
        """Load the fast router model (LLM routing only)."""
        router_type = getattr(self.config, "router_type", "llm")
        if router_type != "llm":
            logger.info(
                "⏭️  Router model loading skipped - router_type=%s does not use an LLM",
                router_type,
            )
            return
        if self.router_model is not None:
            return
        self.router_model = self._load_role(
            "router",
            self.config.router_priority,
            require_nonempty=True,
            speculative=None,
        )

    def _find_shareable(self, requested) -> LoadedModel | None:
        """Return an already-loaded handle that matches the request (L06)."""
        for loaded in (self.agent_model, self.router_model, self.semantic_model):
            if loaded is not None and loaded.matches_request(requested, self.device):
                return loaded
        return None

    @_synchronized
    def load_semantic_model(self):
        """Load the semantic model, reusing a matching loaded handle."""
        if not self.config.semantic_enabled:
            logger.info("⏭️  Semantic model loading disabled in config")
            return
        if self.semantic_model is not None:
            return

        requested = self.config.semantic_priority[0]
        shared = self._find_shareable(requested)
        if shared is not None:
            logger.info("✅ Semantic model reuses loaded handle: %s", shared.name)
            self.semantic_model = shared
            return

        self.semantic_model = self._load_role(
            "semantic", self.config.semantic_priority, require_nonempty=True, speculative=None
        )

    @_synchronized
    def load_agent_model(self):
        """Load the agent model with optional speculative decoding."""
        if not self.config.agent_enabled:
            logger.info("⏭️  Agent model loading disabled in config")
            return
        if self.agent_model is not None:
            return

        spec_cfg = self.config.speculative_decoding
        draft_model_path = None
        if spec_cfg and spec_cfg.enabled and self.config.router_priority:
            # The draft must be an actually-usable exllamav2/vllm model; try
            # to load the router model first and reuse its local path (L08).
            try:
                if self.router_model is None:
                    self.load_router_model()
            except ModelLoadError:
                logger.warning("draft model unavailable; using non-speculative generation")
            if self.router_model is not None:
                draft = self.router_model
                if draft.backend in ("exllamav2", "vllm") and all(
                    candidate.backend == draft.backend
                    and candidate.repo_id == draft.source_repo
                    and candidate.revision == draft.revision
                    and (
                        str(Path(candidate.local_path).resolve()) if candidate.local_path else None
                    )
                    == draft.local_path
                    and (self.device if candidate.device == "auto" else candidate.device)
                    == draft.device
                    and candidate.trust_remote_code == draft.trust_remote_code
                    for candidate in self.config.agent_priority
                ):
                    draft_model_path = draft.model_path

        self.agent_model = self._load_role(
            "agent",
            self.config.agent_priority,
            require_nonempty=True,
            speculative=spec_cfg,
            draft_model_path=draft_model_path,
        )

        # Backfill semantic sharing if semantic is enabled and matches.
        if self.config.semantic_enabled and self.config.semantic_priority:
            requested = self.config.semantic_priority[0]
            if self.semantic_model is None and self.agent_model.matches_request(
                requested, self.device
            ):
                self.semantic_model = self.agent_model
                logger.info("✅ Semantic model will reuse agent model instance")

    def _load_role(
        self,
        role: str,
        priority,
        *,
        require_nonempty: bool,
        speculative=None,
        draft_model_path=None,
    ) -> LoadedModel:
        del require_nonempty  # both branches identical; kept for API clarity
        if not priority:
            raise ModelLoadError(f"{role} model priority list is empty")
        result = self.model_loader.load_llm_model(
            priority,
            draft_model_path=draft_model_path,
            speculative=speculative,
        )
        # L06: identity comes from the candidate that ACTUALLY loaded (the
        # loader attaches it), never from the configured first choice — a
        # fallback candidate changes repo/revision/backend silently.
        metadata = getattr(result, "metadata", None)
        if metadata is None:
            raise ModelLoadError("loader must return ModelLoadResult with authoritative metadata")
        handle, name = result
        requested = metadata.config
        backend = metadata.backend

        if isinstance(handle, tuple) and len(handle) == 2:
            model_obj, tokenizer = handle
        else:
            model_obj, tokenizer = handle, None
        return LoadedModel(
            handle=model_obj,
            name=name,
            backend=backend,
            tokenizer=tokenizer,
            source_repo=getattr(requested, "repo_id", None),
            revision=getattr(requested, "revision", None),
            device=metadata.device
            or str(
                getattr(model_obj, "device", None)
                or (
                    getattr(self.model_loader, "device_config", self.device)
                    if requested.device == "auto"
                    else requested.device
                )
            ),
            quantized=bool(getattr(requested, "quantized", False)),
            local_path=str(Path(requested.local_path).resolve()) if requested.local_path else None,
            model_path=Path(metadata.model_path),
            dtype=requested.dtype,
            trust_remote_code=requested.trust_remote_code,
            extras={"draft_model_path": metadata.draft_model_path},
        )

    def load_all_models(self):
        """Load router, agent, then semantic models.

        The agent loads before semantic so the standard identical-
        semantic/agent configuration shares one loaded handle instead of
        doubling VRAM: semantic resolves against the already-loaded agent
        identity.
        """
        self.load_router_model()
        self.load_agent_model()
        self.load_semantic_model()

    # ------------------------------------------------------------------ #
    # Generation entry points
    # ------------------------------------------------------------------ #

    @_synchronized
    def generate_router(self, prompt: str) -> str:
        """Generate with the router model (fast classification)."""
        if self.router_model is None:
            self.load_router_model()
        if self.router_model is None:
            raise GenerationError("router model not loaded")

        budget = self._resolve_budget(
            "router",
            thinking_tokens=int(getattr(self.config, "router_thinking_tokens", 0)),
        )
        return self._generate(
            self.router_model,
            prompt,
            budget,
            system_prompt=_UNSET,
        )

    @_synchronized
    def generate_semantic(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> str:
        """Generate with the semantic model (low difficulty, no thinking)."""
        if self.semantic_model is None:
            self.load_semantic_model()
        if self.semantic_model is None:
            raise GenerationError("semantic model not loaded")

        budget = self._resolve_budget(
            "semantic",
            answer_override=max_tokens,
            thinking_tokens=int(getattr(self.config, "semantic_thinking_tokens", 0)),
        )
        return self._generate(self.semantic_model, prompt, budget, system_prompt=system_prompt)

    @_synchronized
    def generate_agent(self, prompt: str, thinking_tokens: int | None = None) -> str:
        """Generate with the agent model and an optional thinking budget."""
        if self.agent_model is None:
            self.load_agent_model()
        if self.agent_model is None:
            raise GenerationError("agent model not loaded")

        if thinking_tokens is None:
            thinking_tokens = int(getattr(self.config, "agent_thinking_tokens", 512))

        # phi-4-reasoning manages its own reasoning budget (L11).
        is_phi4 = (
            "phi-4" in (self.agent_model.name or "").lower()
            and "reasoning" in (self.agent_model.name or "").lower()
        )
        if is_phi4:
            thinking_tokens = 0

        budget = self._resolve_budget("agent", thinking_tokens=thinking_tokens)
        return self._generate(self.agent_model, prompt, budget, system_prompt=_UNSET)

    # ------------------------------------------------------------------ #
    # Backend dispatch
    # ------------------------------------------------------------------ #

    @_synchronized
    def _generate(
        self,
        loaded: LoadedModel,
        prompt: str,
        budget: dict,
        *,
        system_prompt: Any = _UNSET,
    ) -> str:
        model = loaded.handle
        if isinstance(model, tuple) and len(model) == 2:
            model, loaded_tokenizer = model
            tokenizer = loaded_tokenizer if loaded.tokenizer is None else loaded.tokenizer
        else:
            tokenizer = loaded.tokenizer

        adapters = {
            "mlx": self._generate_mlx,
            "vllm": self._generate_vllm,
            "exllamav2": self._generate_exllama,
            "transformers": self._generate_transformers,
        }
        adapter = adapters.get(loaded.backend)
        if adapter is None:
            raise GenerationError("unsupported model type or backend")
        if min(budget["answer_tokens"] + budget["thinking_tokens"], budget["hard_total"]) == 0:
            return ""
        try:
            return adapter(model, tokenizer, loaded, prompt, budget, system_prompt)
        except Exception:
            raise GenerationError(f"{loaded.backend} generation failed") from None

    def _format_chat(
        self,
        tokenizer,
        loaded: LoadedModel,
        prompt: str,
        *,
        system_prompt: Any = _UNSET,
        enable_thinking: bool | None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Build the formatted prompt with the tokenizer's template (L09/L10).

        ``system_prompt`` semantics: ``_UNSET`` (default) uses the standard
        assistant prompt; ``None`` omits the system block entirely; a string
        is used verbatim.
        """
        system = _DEFAULT_SYSTEM_PROMPT if system_prompt is _UNSET else system_prompt
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            try:
                formatted: str = tokenizer.apply_chat_template(messages, **kwargs)
                return str(formatted)
            except Exception:
                raise GenerationError("chat template failed") from None
        # Explicit, documented fallback (L09): ChatML with the caller's
        # system prompt preserved verbatim (None means no system block).
        parts = []
        if system is not None:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _generate_mlx(self, model, tokenizer, loaded, prompt, budget, system_prompt) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        formatted = self._format_chat(
            tokenizer,
            loaded,
            prompt,
            system_prompt=system_prompt,
            enable_thinking=budget["thinking_tokens"] > 0,
        )

        total = min(budget["answer_tokens"] + budget["thinking_tokens"], budget["hard_total"])
        try:
            text = mlx_generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=total,
                verbose=False,
                sampler=make_sampler(
                    temp=budget["temperature"], top_p=budget["top_p"], top_k=budget["top_k"]
                ),
            )
        except Exception as exc:
            raise GenerationError(f"MLX generation failed: {exc}") from exc
        return extract_answer(text, had_thinking=_prefilled_thinking(formatted))

    def _generate_vllm(self, model, tokenizer, loaded, prompt, budget, system_prompt) -> str:
        from vllm import SamplingParams

        formatted = self._format_chat(
            tokenizer,
            loaded,
            prompt,
            system_prompt=system_prompt,
            enable_thinking=budget["thinking_tokens"] > 0,
        )

        total = min(budget["answer_tokens"] + budget["thinking_tokens"], budget["hard_total"])
        sampling_params = SamplingParams(
            temperature=budget["temperature"],
            top_p=budget["top_p"],
            top_k=budget.get("top_k", 40),
            max_tokens=total,
        )
        try:
            outputs = model.generate([formatted], sampling_params)
        except Exception as exc:
            raise GenerationError(f"vLLM generation failed: {exc}") from exc
        return extract_answer(
            outputs[0].outputs[0].text, had_thinking=_prefilled_thinking(formatted)
        )

    def _generate_exllama(self, model, tokenizer, loaded, prompt, budget, system_prompt) -> str:
        from exllamav2.generator import ExLlamaV2Sampler

        enable_thinking = budget["thinking_tokens"] > 0
        formatted = self._format_chat(
            tokenizer, loaded, prompt, system_prompt=system_prompt, enable_thinking=enable_thinking
        )

        gen_settings = ExLlamaV2Sampler.Settings()
        gen_settings.temperature = budget["temperature"]
        gen_settings.top_p = budget["top_p"]
        gen_settings.top_k = budget.get("top_k", 40)
        gen_settings.token_repetition_penalty = 1.2
        gen_settings.token_frequency_penalty = 0.1

        total = min(budget["answer_tokens"] + budget["thinking_tokens"], budget["hard_total"])
        try:
            output = model.generate(
                prompt=formatted,
                max_new_tokens=total,
                gen_settings=gen_settings,
                stop_conditions=["<|im_end|>", "<|endoftext|>"],
                completion_only=True,
                encode_special_tokens=True,
                add_bos=False,
            )
        except Exception as exc:
            raise GenerationError(f"ExLlamaV2 generation failed: {exc}") from exc
        return extract_answer(output, had_thinking=_prefilled_thinking(formatted))

    def _generate_transformers(
        self, model, tokenizer, loaded, prompt, budget, system_prompt
    ) -> str:
        import torch

        formatted = self._format_chat(
            tokenizer,
            loaded,
            prompt,
            system_prompt=system_prompt,
            enable_thinking=budget["thinking_tokens"] > 0,
        )

        inputs = tokenizer(formatted, return_tensors="pt", return_token_type_ids=False)
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        total = min(budget["answer_tokens"] + budget["thinking_tokens"], budget["hard_total"])
        sampling = {"do_sample": budget["temperature"] > 0}
        if sampling["do_sample"]:
            sampling.update(
                temperature=budget["temperature"],
                top_p=budget["top_p"],
                top_k=budget.get("top_k", 50),
            )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=total,
                **sampling,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
        eos_token = getattr(tokenizer, "eos_token", None)
        if isinstance(eos_token, str) and eos_token:
            generated = generated.removesuffix(eos_token)
        return extract_answer(generated, had_thinking=_prefilled_thinking(formatted))

    def close(self):
        """Terminal, idempotent close; serialize with generation and release each handle once.

        Calls an engine's explicit close() when provided; otherwise releases
        references only. Device allocators may retain memory outside this component.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            models = (self.router_model, self.agent_model, self.semantic_model)
            self.router_model = self.agent_model = self.semantic_model = None
            seen = set()
            failed = False
            for loaded in models:
                if loaded is not None and id(loaded.handle) not in seen:
                    seen.add(id(loaded.handle))
                    close = getattr(loaded.handle, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            failed = True
            if failed:
                raise GenerationError("backend close failed") from None

    # Compatibility surface ------------------------------------------------ #

    @property
    def router_tokenizer(self):
        return self.router_model.tokenizer if self.router_model else None

    @property
    def semantic_tokenizer(self):
        return self.semantic_model.tokenizer if self.semantic_model else None

    @property
    def agent_tokenizer(self):
        return self.agent_model.tokenizer if self.agent_model else None
