"""Property-based tests with hypothesis for pure routing logic."""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from signal_llm.intent_router import IntentRouter


def make_router(**kwargs):
    defaults = dict(router_type="llm")
    defaults.update(kwargs)
    return IntentRouter(**defaults)


@settings(max_examples=200)
@given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
def test_thinking_budget_always_within_bounds(score):
    router = make_router(min_thinking_tokens=0, max_thinking_tokens=2500)
    budget = router._calculate_thinking_budget(score)
    assert 0 <= budget <= 2500


@settings(max_examples=200)
@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.integers(min_value=0, max_value=500),
    st.integers(min_value=500, max_value=5000),
)
def test_thinking_budget_monotonic_and_within_min_max(score, lo, hi):
    if hi <= lo:
        return
    router = make_router(min_thinking_tokens=lo, max_thinking_tokens=hi)
    budget = router._calculate_thinking_budget(score)
    assert lo <= budget <= hi
    # Quadratic curve is non-decreasing in score.
    higher = router._calculate_thinking_budget(min(1.0, score + 0.01))
    assert higher >= budget


@settings(max_examples=100)
@given(st.text(alphabet=string.printable, min_size=0, max_size=48))
def test_reflex_layer_never_crashes(text):
    router = make_router()
    assert isinstance(router._is_reflex(text.lower().strip()), bool)


@settings(max_examples=100)
@given(st.text(alphabet=string.ascii_lowercase + " ", min_size=0, max_size=64))
def test_semantic_layer_never_crashes(text):
    router = make_router()
    assert isinstance(router._is_semantic(text.lower()), bool)


@settings(max_examples=50)
@given(st.sampled_from(["", "qwen2.5-7b", "llama-3-base", "mistral-instruct", "OLMo-2"]))
def test_chat_model_detection_consistent(model_name):
    from signal_llm.component import _is_chat_model

    without_tok = _is_chat_model(model_name)
    with_tok = _is_chat_model(model_name, tokenizer=type("T", (), {"chat_template": "<x>"})())
    # Tokenizer presence can only upgrade to True, never downgrade.
    assert with_tok is True or with_tok == without_tok
