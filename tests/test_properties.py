"""Property-based tests with complete reference predicates."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from signal_llm.component import extract_answer
from signal_llm.intent_router import IntentRouter, _normalize_label, validate_score
from signal_llm.loader import model_cache_key, parse_device

# --------------------------------------------------------------------- #
# Device parsing is total and matches the reference
# --------------------------------------------------------------------- #


def _reference_parse(device: str | None):
    raw = (device or "auto").strip().lower()
    if raw in ("auto", "cpu", "mps", ""):
        return (raw or "auto", None)
    if (raw == "cuda" or raw.startswith("cuda:")) and raw.count(":") <= 1:
        if ":" in raw:
            idx = raw.split(":", 1)[1]
            if idx and all(c in "0123456789" for c in idx):
                return ("cuda", int(idx))
            return None
        return ("cuda", None)
    return None


@settings(max_examples=200)
@given(st.one_of(st.none(), st.text(max_size=12)))
def test_parse_device_total_and_reference_checked(device) -> None:
    expected = _reference_parse(device)
    if expected is None:
        try:
            parse_device(device)
        except ValueError:
            return
        raise AssertionError(f"expected rejection for {device!r}")
    assert parse_device(device) == expected


# --------------------------------------------------------------------- #
# Cache keys are injective over sane identities
# --------------------------------------------------------------------- #


_repo_ids = st.sampled_from(["org/m", "a_b/c", "a/b_c", "csukuangfj/sherpa-onnx-x", "x"])
_revisions = st.one_of(
    st.none(), st.just("main"), st.text(alphabet="abcdef0123456789", min_size=6, max_size=8)
)


@settings(max_examples=300)
@given(_repo_ids, _revisions, _repo_ids, _revisions)
def test_cache_keys_injective(repo1, rev1, repo2, rev2) -> None:
    key1 = model_cache_key(repo1, rev1)
    key2 = model_cache_key(repo2, rev2)
    identity1 = (repo1, rev1 or "main")
    identity2 = (repo2, rev2 or "main")
    if identity1 == identity2:
        assert key1 == key2
    else:
        assert key1 != key2, f"collision between {identity1} and {identity2}"


@settings(max_examples=100)
@given(_repo_ids, _revisions)
def test_cache_keys_are_safe_path_components(repo, revision) -> None:
    key = model_cache_key(repo, revision)
    assert "/" not in key
    assert ".." not in key
    assert key.strip() == key and key != ""


# --------------------------------------------------------------------- #
# Answer extraction matches the reference semantics
# --------------------------------------------------------------------- #


_plain_text = st.text(
    alphabet="think/abcXYZ .,!\n",
    min_size=0,
    max_size=80,
)


@settings(max_examples=300)
@given(_plain_text, _plain_text, st.booleans())
def test_extract_answer_matches_reference(reasoning, answer, prefilled) -> None:
    raw = ("" if prefilled else "<think>") + reasoning + "</think>" + answer
    assert extract_answer(raw, had_thinking=prefilled) == answer.strip()
    incomplete = ("" if prefilled else "<think>") + reasoning
    assert extract_answer(incomplete, had_thinking=prefilled) == ""
    assert extract_answer(answer, had_thinking=False) == answer.strip()


# --------------------------------------------------------------------- #
# Label normalisation and score validation
# --------------------------------------------------------------------- #


@settings(max_examples=200)
@given(
    st.one_of(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz .,!;:\n", max_size=24),
        st.sampled_from(["reflex", "semantic!", "agentic", "not reflex", "semantic or agentic"]),
    )
)
def test_normalize_label_never_guesses(text) -> None:
    import pytest

    normalized = text.strip().lower()
    if normalized and normalized[-1] in ".,!;:":
        normalized = normalized[:-1].strip()
    if normalized not in ("reflex", "semantic", "agentic"):
        with pytest.raises(ValueError):
            _normalize_label(text)
        return
    assert _normalize_label(text) == normalized


@settings(max_examples=100)
@given(
    st.one_of(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=-5.0, max_value=5.0),
    )
)
def test_validate_score_boundary(value) -> None:
    if math.isfinite(value) and 0.0 <= value <= 1.0:
        assert validate_score("t", value) == value
    else:
        import pytest

        with pytest.raises(ValueError):
            validate_score("t", value)


# --------------------------------------------------------------------- #
# Budget curve reference
# --------------------------------------------------------------------- #


def _reference_budget(score: float, lo: int, hi: int) -> int:
    clamped = max(0.0, min(1.0, score))
    return int(lo + (hi - lo) * (clamped * clamped))


@settings(max_examples=200)
@given(st.floats(min_value=-1.0, max_value=2.0), st.integers(0, 500), st.integers(0, 500))
def test_budget_curve_matches_reference(score, lo, hi) -> None:
    if hi < lo:
        lo, hi = hi, lo
    router = IntentRouter(router_type="llm", min_thinking_tokens=lo, max_thinking_tokens=hi)
    assert router._calculate_thinking_budget(score) == _reference_budget(score, lo, hi)
