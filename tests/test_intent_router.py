"""Tests for the strict intent-routing contract (L13/L14/L17/L18/L22)."""

from __future__ import annotations

import pytest

from signal_llm.intent_router import (
    IntentRouter,
    _normalize_label,
    validate_score,
)


@pytest.mark.parametrize(
    "text",
    [
        "do not go to definition",
        "explain go to definition",
        "what does find references mean",
        "please do not rename variable",
    ],
)
def test_semantic_shortcut_does_not_accept_negation_or_explanation(text):
    calls = []
    router = IntentRouter(router_type="llm")

    def classify(prompt):
        calls.append(prompt)
        return "agentic"

    assert router.route(text, router_generate_func=classify)[0] == "agentic"
    assert len(calls) == 1


# --------------------------------------------------------------------- #
# Label normalisation (L13)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("label", ["reflex", "semantic", "agentic"])
def test_exact_labels_accepted(label) -> None:
    assert _normalize_label(label) == label


@pytest.mark.parametrize(
    "raw",
    [
        "  REFLEX  ",
        "Semantic.",
        "agentic!",
        "reflex,",
    ],
)
def test_benign_normalisation_accepted(raw) -> None:
    assert _normalize_label(raw) in ("reflex", "semantic", "agentic")


@pytest.mark.parametrize(
    "raw",
    [
        "not reflex",
        "semantic or agentic",
        "semantic and agentic",
        "agentic - definitely",
        "I think this is semantic",
        "Classify this command into one category: reflex",
        "",
        "   ",
        "...",
        "reflex semantic",
        "unknown",
    ],
)
def test_ambiguous_output_rejected(raw) -> None:
    with pytest.raises(ValueError):
        _normalize_label(raw)


def test_non_string_output_rejected() -> None:
    with pytest.raises(ValueError, match="non-string"):
        _normalize_label(None)
    with pytest.raises(ValueError, match="non-string"):
        _normalize_label(42)


# --------------------------------------------------------------------- #
# Score validation (L17)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_rejected(bad) -> None:
    with pytest.raises(ValueError):
        validate_score("estimator", bad)


def test_valid_scores_pass() -> None:
    assert validate_score("estimator", 0.0) == 0.0
    assert validate_score("estimator", 1.0) == 1.0
    assert validate_score("estimator", 0.42) == 0.42


# --------------------------------------------------------------------- #
# Router fixtures
# --------------------------------------------------------------------- #


class FakeMfRouter:
    def __init__(self, score=0.5):
        self.score = score
        self.win_rate_calls = 0

    def calculate_strong_win_rate(self, text):
        self.win_rate_calls += 1
        return self.score


class FakeController:
    def __init__(self, router):
        self.routers = {"mf": router}


class FakeCoreml:
    def __init__(self, score=0.5):
        self.score = score
        self.calls = 0

    def complexity_score(self, text):
        self.calls += 1
        return self.score


def _llm_router() -> IntentRouter:
    return IntentRouter(router_type="llm")


# --------------------------------------------------------------------- #
# Reflex layer
# --------------------------------------------------------------------- #


def test_exact_reflex_commands_bypass_everything() -> None:
    router = _llm_router()
    mf = FakeMfRouter(0.9)
    controller = FakeController(mf)
    routellm = IntentRouter(
        router_type="routellm", routellm_controller=controller, routellm_threshold=0.5
    )
    for text in ("click", "scroll up", "hello", "thank you"):
        assert router.route(text)[0] == "reflex"
        # Reflex never touches the estimator (L14/L18).
        assert routellm.route(text) == ("reflex", None, None)
    assert mf.win_rate_calls == 0


def test_questions_never_match_reflex() -> None:
    router = _llm_router()
    intents = {
        text: router.route(text, router_generate_func=lambda p: "agentic")[0]
        for text in ("how fast is this?", "what is this?")
    }
    assert all(intent != "reflex" for intent in intents.values())


# --------------------------------------------------------------------- #
# Semantic command anchoring (L14)
# --------------------------------------------------------------------- #


def test_concrete_code_commands_are_semantic() -> None:
    router = _llm_router()
    for text in (
        "go to definition",
        "find references",
        "rename function",
        "please rename function",
        "extract method",
    ):
        assert router.route(text)[0] == "semantic", text


def test_ordinary_go_to_questions_are_not_semantic() -> None:
    router = _llm_router()
    # Must NOT take the semantic shortcut; classify via LLM instead.
    calls: list[str] = []

    def classifier(prompt: str) -> str:
        calls.append(prompt)
        return "agentic"

    for text in (
        "should I go to the hospital?",
        "how do I go to the park",
        "explain how a function differs from a class",
    ):
        intent, _score, _budget = router.route(text, router_generate_func=classifier)
        assert intent == "agentic", text


def test_code_commands_skip_estimator_calls() -> None:
    mf = FakeMfRouter(0.9)
    router = IntentRouter(
        router_type="routellm", routellm_controller=FakeController(mf), routellm_threshold=0.5
    )
    intent, _score, budget = router.route("go to definition")
    assert intent == "semantic"
    assert budget is None
    assert mf.win_rate_calls == 0, "high-confidence commands must skip the estimator (L14)"


# --------------------------------------------------------------------- #
# Scorer routing + budgets (L17/L18)
# --------------------------------------------------------------------- #


def test_high_difficulty_routes_agentic_with_budget() -> None:
    mf = FakeMfRouter(0.9)
    router = IntentRouter(
        router_type="routellm",
        routellm_controller=FakeController(mf),
        routellm_threshold=0.5,
        min_thinking_tokens=0,
        max_thinking_tokens=2500,
    )
    intent, score, budget = router.route("write a parser with error recovery")
    assert intent == "agentic"
    assert score == 0.9
    assert budget is not None
    assert 0 <= budget <= 2500
    # Quadratic curve: 0.9^2 = 0.81 -> 2025 tokens.
    assert budget == 2025


def test_low_difficulty_routes_semantic_without_budget() -> None:
    mf = FakeMfRouter(0.1)
    router = IntentRouter(
        router_type="routellm", routellm_controller=FakeController(mf), routellm_threshold=0.5
    )
    intent, score, budget = router.route("what is a monad in simple terms")
    assert intent == "semantic"
    assert score == 0.1
    assert budget is None


def test_nonfinite_scorer_output_raises() -> None:
    class NanRouter:
        def calculate_strong_win_rate(self, text):
            return float("nan")

    router = IntentRouter(router_type="routellm", routellm_controller=FakeController(NanRouter()))
    with pytest.raises(ValueError, match="non-finite"):
        router.route("write me a program")


def test_out_of_range_scorer_output_raises() -> None:
    class RangeRouter:
        def calculate_strong_win_rate(self, text):
            return 7.5

    router = IntentRouter(router_type="routellm", routellm_controller=FakeController(RangeRouter()))
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        router.route("write me a program")


def test_budget_curve_endpoints() -> None:
    router = IntentRouter(
        router_type="routellm",
        routellm_controller=FakeController(FakeMfRouter(0.0)),
        min_thinking_tokens=10,
        max_thinking_tokens=1000,
    )
    assert router._calculate_thinking_budget(0.0) == 10
    assert router._calculate_thinking_budget(1.0) == 1000
    # 10 + (1000-10) * 0.25 = 257.5 -> int() truncates to 257.
    assert router._calculate_thinking_budget(0.5) == 257


def test_regressor_budget_validated() -> None:
    class BadRegressorRouter:
        def calculate_strong_win_rate(self, text):
            return 0.9

        class model:
            @staticmethod
            def predict_thinking_tokens(text, difficulty_score, min_tokens, max_tokens):
                return 99_999

    router = IntentRouter(
        router_type="routellm",
        routellm_controller=FakeController(BadRegressorRouter()),
        min_thinking_tokens=0,
        max_thinking_tokens=100,
    )
    with pytest.raises(ValueError, match="outside"):
        router.route("write a compiler")


def test_regressor_non_int_budget_rejected() -> None:
    class FloatRegressorRouter:
        def calculate_strong_win_rate(self, text):
            return 0.9

        class model:
            @staticmethod
            def predict_thinking_tokens(text, difficulty_score, min_tokens, max_tokens):
                return 42.0

    router = IntentRouter(
        router_type="routellm",
        routellm_controller=FakeController(FloatRegressorRouter()),
    )
    with pytest.raises(ValueError, match="non-int"):
        router.route("write a compiler")


def test_coreml_uses_complexity_score() -> None:
    coreml = FakeCoreml(0.8)
    router = IntentRouter(router_type="coreml", coreml_classifier=coreml, routellm_threshold=0.5)
    intent, score, budget = router.route("refactor the whole module graph")
    assert intent == "agentic"
    assert score == 0.8
    assert budget is not None
    assert coreml.calls == 1


# --------------------------------------------------------------------- #
# LLM classification (L13/L15)
# --------------------------------------------------------------------- #


def test_llm_classification_via_func() -> None:
    router = _llm_router()
    calls: list[str] = []

    def classifier(prompt: str) -> str:
        calls.append(prompt)
        return "agentic"

    intent, _score, budget = router.route("write a web server", router_generate_func=classifier)
    assert intent == "agentic"
    assert len(calls) == 1
    assert budget is not None


def test_llm_classification_ambiguous_result_raises() -> None:
    router = _llm_router()
    with pytest.raises(ValueError, match="ambiguous"):
        router.route("write a web server", router_generate_func=lambda p: "maybe agentic?")


def test_llm_classification_uses_generate_router_adapter(monkeypatch) -> None:
    router = _llm_router()

    class FakeComponent:
        def generate_router(self, prompt: str) -> str:
            return "semantic"

    intent, _, _ = router.route("explain the tradeoffs here", router_model=FakeComponent())
    assert intent == "semantic"


def test_llm_classification_rejects_raw_backend_objects() -> None:
    router = _llm_router()

    class RawTransformersModel:
        def generate(self, **kwargs):  # wrong signature: kwargs, not str->str
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="str->str"):
        router.route("write a program", router_model=RawTransformersModel())


def test_llm_router_requires_model_or_func() -> None:
    router = _llm_router()
    with pytest.raises(ValueError, match="router_model or router_generate_func"):
        router.route("write a program")


# --------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------- #


def test_invalid_router_type_rejected() -> None:
    with pytest.raises(ValueError, match="router_type"):
        IntentRouter(router_type="psychic")


def test_routellm_requires_controller() -> None:
    with pytest.raises(ValueError, match="routellm_controller"):
        IntentRouter(router_type="routellm", routellm_controller=None)


def test_coreml_requires_classifier() -> None:
    with pytest.raises(ValueError, match="coreml_classifier"):
        IntentRouter(router_type="coreml", coreml_classifier=None)


def test_inverted_budget_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="max_thinking_tokens"):
        IntentRouter(min_thinking_tokens=100, max_thinking_tokens=10)


def test_route_rejects_non_string_input() -> None:
    router = _llm_router()
    with pytest.raises(ValueError):
        router.route("")
    with pytest.raises(ValueError):
        router.route("   ")
