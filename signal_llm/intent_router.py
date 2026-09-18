"""
Intent Router Component

Cascading router architecture with three layers:

1. Reflex: exact-match command patterns (0ms latency)
2. Estimator: difficulty score (RouteLLM matrix factorization or CoreML
   complexity classifier, ~4-30ms)
3. Thinker: difficulty-to-thinking-budget mapping (quadratic curve)

Contract (remediation L13, L14, L17, L18, L22):

- LLM label parsing accepts exactly one of the three labels after
  normalisation; negations, multi-label output, prompt echoes, and
  non-string results are errors, not guesses (L13).
- High-confidence code-command shortcuts run *before* the estimator;
  anchored, code-specific patterns keep ordinary questions ("should I go
  to the hospital?") out of the semantic route (L14).
- Thinking budgets are computed for agentic routes only; non-agentic
  results carry ``None`` budgets and never invoke the regressor (L18).
- Scores and budgets are validated at the boundary: nonfinite or
  out-of-range scores raise, and can never silently route traffic (L17).
- Logs carry metadata only — never prompt content (L22).
"""

import logging
import math
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

VALID_INTENTS = ("reflex", "semantic", "agentic")

# Anchored, code-specific command patterns. These are deliberately narrow:
# they must only match editor commands, never ordinary questions (L14).
_SEMANTIC_COMMAND_PATTERNS = (
    "go to definition",
    "go to declaration",
    "go to function",
    "go to class",
    "go to method",
    "go to symbol",
    "go to line",
    "jump to definition",
    "jump to declaration",
    "find definition",
    "find declaration",
    "find usage",
    "find usages",
    "find reference",
    "find references",
    "open function",
    "open class",
    "open method",
    "rename function",
    "rename class",
    "rename method",
    "rename variable",
    "rename parameter",
    "extract function",
    "extract method",
    "extract variable",
    "inline function",
    "inline method",
    "inline variable",
    "delete function",
    "delete class",
    "delete method",
    "delete line",
    "move function",
    "move class",
    "move method",
    "navigate to definition",
    "navigate to declaration",
    "navigate to symbol",
)


def _normalize_label(raw: Any) -> str:
    """Normalise a classifier label strictly (L13).

    Returns one of ``VALID_INTENTS``. Anything else raises ``ValueError``:
    negations, multiple labels, explanations, prompt echoes, empty output,
    and non-strings are rejected rather than guessed.
    """
    if not isinstance(raw, str):
        raise ValueError(f"classifier returned non-string output: {type(raw).__name__}")
    text = raw.strip().lower()
    # Strip a single trailing punctuation character (".", "!", ",").
    if text and text[-1] in ".,!;:":
        text = text[:-1].strip()
    if not text:
        raise ValueError("classifier returned empty output")
    # Exactly one full-token label is required. Substring matching would
    # accept "not reflex" / "semantic or agentic" / echoed prompts.
    tokens = text.split()
    for intent in VALID_INTENTS:
        if tokens == [intent]:
            return intent
    raise ValueError("classifier returned ambiguous or unknown label")


def validate_score(name: str, score: float) -> float:
    """Require a finite score within [0, 1] (L17)."""
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"{name} returned a non-finite score: {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} returned a score outside [0, 1]: {value!r}")
    return value


class IntentRouter:
    """Cascading router: reflex patterns -> difficulty estimator -> budget."""

    def __init__(
        self,
        router_type: str = "llm",
        routellm_controller: Any | None = None,
        routellm_threshold: float = 0.5,
        routellm_router_name: str = "mf",
        min_thinking_tokens: int = 0,
        max_thinking_tokens: int = 2500,
        coreml_classifier: Any | None = None,
    ):
        if (
            isinstance(min_thinking_tokens, bool)
            or not isinstance(min_thinking_tokens, int)
            or min_thinking_tokens < 0
        ):
            raise ValueError("min_thinking_tokens must be a non-negative int")
        if (
            isinstance(max_thinking_tokens, bool)
            or not isinstance(max_thinking_tokens, int)
            or max_thinking_tokens < 0
        ):
            raise ValueError("max_thinking_tokens must be a non-negative int")
        if max_thinking_tokens < min_thinking_tokens:
            raise ValueError(
                "max_thinking_tokens must be >= min_thinking_tokens "
                f"(got {max_thinking_tokens} < {min_thinking_tokens})"
            )
        validate_score("routellm_threshold", routellm_threshold)

        self._reflex_patterns = self._build_reflex_patterns()
        self._router_type = router_type
        self._coreml_classifier = coreml_classifier
        self._min_thinking_tokens = min_thinking_tokens
        self._max_thinking_tokens = max_thinking_tokens

        if router_type == "routellm":
            if routellm_controller is None:
                raise ValueError("router_type='routellm' requires a routellm_controller")
            self._routellm_controller = routellm_controller
            self._routellm_threshold = routellm_threshold
            self._routellm_router_name = routellm_router_name
        elif router_type == "coreml":
            if coreml_classifier is None:
                raise ValueError("router_type='coreml' requires a coreml_classifier")
            self._routellm_threshold = routellm_threshold
        elif router_type == "llm":
            pass
        else:
            raise ValueError(
                f"invalid router_type {router_type!r}; must be one of {VALID_INTENTS[:0] or ('routellm', 'coreml', 'llm')}"
            )
        logger.info(
            "IntentRouter initialised (type=%s, threshold=%s, budget=%d-%d tokens)",
            router_type,
            getattr(self, "_routellm_threshold", None),
            min_thinking_tokens,
            max_thinking_tokens,
        )

    # ------------------------------------------------------------------ #
    # Pattern tables
    # ------------------------------------------------------------------ #

    def _build_reflex_patterns(self) -> set[str]:
        return {
            # Window management
            "close window",
            "open window",
            "minimize",
            "maximize",
            "restore",
            "switch window",
            "next window",
            "previous window",
            # Navigation
            "scroll up",
            "scroll down",
            "scroll left",
            "scroll right",
            "page up",
            "page down",
            "go to top",
            "go to bottom",
            # Actions
            "click",
            "double click",
            "right click",
            "type",
            "paste",
            "copy",
            "cut",
            "select all",
            "undo",
            "redo",
            "save",
            "open",
            "close",
            # System
            "lock screen",
            "sleep",
            "shutdown",
            "restart",
            # Simple greetings only - questions go to the router
            "hello",
            "hi",
            "hey",
            "thanks",
            "thank you",
            "yes",
            "no",
            "ok",
            "okay",
        }

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def route(
        self,
        text: str,
        router_model: Any | None = None,
        router_generate_func: Callable[[str], str] | None = None,
    ) -> tuple[str, float | None, int | None]:
        """Route ``text`` through the cascade.

        Returns ``(intent, difficulty_score, thinking_tokens)`` where the
        score is ``None`` for reflex/semantic and the budget is ``None``
        for every non-agentic intent (L18).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("route() requires a non-empty string")

        text_lower = text.lower().strip()

        # LAYER 0: high-confidence command shortcuts run before any neural
        # work (L14) — but only for anchored code commands, never questions.
        if self._is_reflex(text_lower):
            return ("reflex", None, None)

        if self._is_semantic_command(text_lower):
            # High-confidence code command: no estimator call, no budget.
            return ("semantic", None, None)

        difficulty_score: float | None = None
        if self._router_type == "routellm":
            router = self._routellm_controller.routers[self._routellm_router_name]
            difficulty_score = validate_score(
                "routellm router", router.calculate_strong_win_rate(text)
            )
        elif self._router_type == "coreml":
            classifier = self._coreml_classifier
            if classifier is None:  # pragma: no cover - guarded in __init__
                raise ValueError("coreml_classifier disappeared")
            difficulty_score = validate_score(
                "coreml classifier", classifier.complexity_score(text)
            )

        if self._router_type in ("routellm", "coreml"):
            assert difficulty_score is not None
            agentic = difficulty_score > self._routellm_threshold
            intent = "agentic" if agentic else "semantic"
            thinking_tokens = self._thinking_budget(text, difficulty_score) if agentic else None
            return (intent, difficulty_score, thinking_tokens)

        # LLM classification path.
        if router_generate_func is not None:
            intent = self._classify_with_func(router_generate_func, text)
        elif router_model is not None:
            intent = self._classify_with_model(router_model, text)
        else:
            raise ValueError("router_type='llm' requires router_model or router_generate_func")
        thinking_tokens = (
            self._calculate_thinking_budget(
                difficulty_score if difficulty_score is not None else 0.5
            )
            if intent == "agentic"
            else None
        )
        return (intent, difficulty_score, thinking_tokens)

    # ------------------------------------------------------------------ #
    # Layers
    # ------------------------------------------------------------------ #

    def _is_reflex(self, text_lower: str) -> bool:
        """Exact-match reflex commands only; questions never match."""
        return text_lower in self._reflex_patterns

    def _is_semantic_command(self, text_lower: str) -> bool:
        """Anchored code-command detection (L14).

        Only an exact command, optionally prefixed with "please", bypasses
        classification. Negations, explanations and arbitrary suffixes never do.
        """
        return text_lower.removeprefix("please ") in _SEMANTIC_COMMAND_PATTERNS

    def _thinking_budget(self, text: str, difficulty_score: float) -> int:
        """Agentic budget via the MF regressor when available, else curve."""
        score = validate_score("difficulty score", difficulty_score)
        if self._router_type == "routellm":
            router = self._routellm_controller.routers[self._routellm_router_name]
            predict = getattr(getattr(router, "model", None), "predict_thinking_tokens", None)
            if callable(predict):
                try:
                    budget = predict(
                        text,
                        difficulty_score=score,
                        min_tokens=self._min_thinking_tokens,
                        max_tokens=self._max_thinking_tokens,
                    )
                except ValueError:
                    # Contract violations (non-int / out-of-range budgets)
                    # must surface, not silently fall back (L17).
                    raise ValueError("thinking regressor rejected input or configuration") from None
                except Exception:
                    logger.warning(
                        "thinking regressor failed; using difficulty curve",
                    )
                else:
                    return self._validate_budget(budget)
        return self._calculate_thinking_budget(score)

    def _validate_budget(self, budget: Any) -> int:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError("thinking regressor returned non-int budget")
        if budget < self._min_thinking_tokens or budget > self._max_thinking_tokens:
            raise ValueError(
                f"thinking regressor returned budget {budget} outside "
                f"[{self._min_thinking_tokens}, {self._max_thinking_tokens}]"
            )
        return int(budget)

    def _calculate_thinking_budget(self, difficulty_score: float) -> int:
        """Quadratic difficulty-to-budget curve, clamped to the bounds."""
        score = max(0.0, min(1.0, float(difficulty_score)))
        budget = self._min_thinking_tokens + (
            self._max_thinking_tokens - self._min_thinking_tokens
        ) * (score * score)
        return int(budget)

    # ------------------------------------------------------------------ #
    # LLM classification (L13/L15)
    # ------------------------------------------------------------------ #

    _CLASSIFIER_PROMPT = (
        "Classify this command into one category:\nCommand: {command}\n\n"
        "Categories:\n"
        "- reflex: Simple commands like 'click', 'scroll', 'close window'\n"
        "- semantic: Code navigation like 'go to function', 'find definition'\n"
        "- agentic: Complex queries requiring reasoning like 'how do I...', "
        "'explain...', 'write code for...'\n\n"
        "Answer with only one word:"
    )

    def _classify_with_func(self, func: Callable[[str], str], text: str) -> str:
        """Classify via a ``str -> str`` callable (L15 contract)."""
        raw = func(self._CLASSIFIER_PROMPT.format(command=text))
        try:
            return _normalize_label(raw)
        except ValueError as exc:
            raise ValueError(f"router classification failed: {exc}") from None

    def _classify_with_model(self, router_model: Any, text: str) -> str:
        """Classify via the component's generation adapter.

        Raw backend objects (transformers/vLLM/MLX) do not satisfy a
        ``str -> str`` contract; the caller must pass
        ``LLMComponent.generate_router`` as ``router_generate_func``
        instead (L15).
        """
        generate = getattr(router_model, "generate_router", None)
        if callable(generate):
            return self._classify_with_func(generate, text)
        raise ValueError(
            "router_model must expose a str->str generate interface; pass "
            "LLMComponent.generate_router as router_generate_func instead "
            "of a raw backend object"
        )
