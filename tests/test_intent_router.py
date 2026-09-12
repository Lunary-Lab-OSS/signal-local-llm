"""Comprehensive tests for IntentRouter cascade behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from signal_llm.intent_router import IntentRouter


def make_router(**kwargs):
    defaults = dict(router_type="llm")
    defaults.update(kwargs)
    return IntentRouter(**defaults)


class FakeController:
    def __init__(self, win_rate, thinking=None):
        router = SimpleNamespace(calculate_strong_win_rate=lambda text: win_rate)
        if thinking is not None:
            router.model = SimpleNamespace(
                predict_thinking_tokens=lambda text, difficulty_score, min_tokens, max_tokens: (
                    thinking
                )
            )
        self.routers = {"mf": router}


class FakeCoreML:
    def __init__(self, score):
        self.complexity_score = lambda text: score


# --- Constructor validation ---


def test_routellm_without_controller_raises():
    with pytest.raises(RuntimeError, match="controller is None"):
        IntentRouter(router_type="routellm", routellm_controller=None)


def test_coreml_without_classifier_raises():
    with pytest.raises(RuntimeError, match="coreml_classifier is None"):
        IntentRouter(router_type="coreml", coreml_classifier=None)


def test_invalid_router_type_raises():
    with pytest.raises(RuntimeError, match="Invalid router_type"):
        IntentRouter(router_type="bogus")


# --- Layer 1: Reflex ---


@pytest.mark.parametrize(
    "text", ["hello", "HI", "  okay  ", "scroll up", "close window", "thank you"]
)
def test_reflex_exact_matches(text):
    intent, score, tokens = make_router().route(text)
    assert intent == "reflex"
    assert score is None
    assert tokens is None


@pytest.mark.parametrize(
    "text",
    ["hello there", "how fast is this", "okay but what about the file", "clicking sounds good"],
)
def test_reflex_is_strict_no_substring_matching(text):
    router = make_router(router_type="llm")
    intent, _, _ = router.route(text, router_generate_func=lambda t: "agentic")
    assert intent != "reflex"


# --- Layer 2: Semantic ---


def test_semantic_two_keywords():
    intent, _, _ = make_router().route("rename function")
    assert intent == "semantic"


def test_semantic_code_phrase():
    intent, _, _ = make_router().route("please go to definition of main")
    assert intent == "semantic"


def test_semantic_single_keyword_not_enough():
    router = make_router(router_type="llm")
    intent, _, _ = router.route("delete", router_generate_func=lambda t: "agentic")
    assert intent != "semantic"


# --- RouteLLM routing ---


def test_routellm_high_difficulty_is_agentic():
    router = make_router(
        router_type="routellm", routellm_controller=FakeController(0.9), routellm_threshold=0.5
    )
    intent, score, tokens = router.route("explain the compiler pipeline")
    assert intent == "agentic"
    assert score == 0.9
    assert tokens is not None


def test_routellm_low_difficulty_is_semantic():
    router = make_router(
        router_type="routellm", routellm_controller=FakeController(0.2), routellm_threshold=0.5
    )
    intent, score, _ = router.route("what is a variable name")
    assert intent == "semantic"
    assert score == 0.2


def test_routellm_uses_mf_thinking_regressor_when_present():
    router = make_router(
        router_type="routellm",
        routellm_controller=FakeController(0.8, thinking=1234),
        routellm_threshold=0.5,
    )
    _, _, tokens = router.route("hard reasoning question")
    assert tokens == 1234


def test_routellm_falls_back_to_quadratic_budget_on_regressor_error():
    controller = FakeController(0.8)
    controller.routers["mf"].model = SimpleNamespace(
        predict_thinking_tokens=lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    )
    router = make_router(
        router_type="routellm",
        routellm_controller=controller,
        routellm_threshold=0.5,
        min_thinking_tokens=0,
        max_thinking_tokens=2500,
    )
    _, _, tokens = router.route("hard question")
    assert tokens == 2500 * 0.64  # 0.8^2


# --- CoreML routing ---


def test_coreml_high_complexity_is_agentic():
    router = make_router(
        router_type="coreml", coreml_classifier=FakeCoreML(0.9), routellm_threshold=0.5
    )
    intent, score, _ = router.route("derive the asymptotic bound")
    assert intent == "agentic"
    assert score == 0.9


def test_coreml_low_complexity_is_semantic():
    router = make_router(
        router_type="coreml", coreml_classifier=FakeCoreML(0.1), routellm_threshold=0.5
    )
    intent, _, _ = router.route("tell me about the weather")
    assert intent == "semantic"


# --- LLM-router fallback path ---


def test_llm_router_uses_generate_func():
    router = make_router(router_type="llm")
    intent, score, _ = router.route(
        "what is the meaning of this", router_generate_func=lambda t: "agentic"
    )
    assert intent == "agentic"
    assert score == 0.5


def test_llm_router_uses_model_when_no_func():
    class Model:
        def generate(self, prompt, **kwargs):
            return "semantic"

    router = make_router(router_type="llm")
    intent, _, _ = router.route("find the function definition", router_model=Model())
    assert intent in ("semantic", "agentic")


def test_llm_router_without_model_or_func_raises():
    router = make_router(router_type="llm")
    with pytest.raises(RuntimeError):
        router.route("non reflex non semantic query here")


# --- Thinking budget curve ---


def test_thinking_budget_quadratic_curve_values():
    router = make_router(min_thinking_tokens=0, max_thinking_tokens=2500)
    assert router._calculate_thinking_budget(0.0) == 0
    assert router._calculate_thinking_budget(0.5) == 625
    assert router._calculate_thinking_budget(1.0) == 2500
    assert router._calculate_thinking_budget(0.9) == 2025


def test_thinking_budget_clamps_out_of_range_scores():
    router = make_router(min_thinking_tokens=10, max_thinking_tokens=100)
    assert router._calculate_thinking_budget(-5.0) == 10
    assert router._calculate_thinking_budget(7.0) == 100


def test_thinking_budget_with_nonzero_min():
    router = make_router(min_thinking_tokens=100, max_thinking_tokens=1100)
    assert router._calculate_thinking_budget(0.0) == 100
    assert router._calculate_thinking_budget(1.0) == 1100
    assert router._calculate_thinking_budget(0.5) == 350
