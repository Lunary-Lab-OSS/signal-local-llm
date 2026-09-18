"""Tests for the matrix-factorization strategy and neural building blocks.

- MF strategy runs against a fake routellm controller (R02/R08/R10).
- Model blocks run under real CPU torch.
- CoreML classifier runs with fake coremltools (R07 boundaries).
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from signal_llm.config import LLMConfig, ModelConfig  # noqa: E402
from signal_llm.router.features import LinguisticFeatureExtractor  # noqa: E402
from signal_llm.router.matrix_factorization import (  # noqa: E402
    MatrixFactorizationRouterStrategy,
)
from signal_llm.router.model import (  # noqa: E402
    GatedFusionHead,
    PreLNResidualSwiGLUBlock,
    ResidualSwiGLUBlock,
    SwiGLUBlock,
    SwiGLUTransform,
    swish,
)


def _config(**overrides) -> LLMConfig:
    defaults = {
        "router_priority": [ModelConfig(name="r", repo_id="org/r", backend="transformers")],
        "semantic_priority": [ModelConfig(name="s", repo_id="org/s", backend="transformers")],
        "agent_priority": [ModelConfig(name="a", repo_id="org/a", backend="transformers")],
        "router_type": "routellm",
        "routellm_router_name": "mf",
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


# --------------------------------------------------------------------- #
# Neural building blocks (real torch)
# --------------------------------------------------------------------- #


def test_swish_matches_reference() -> None:
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = x * torch.sigmoid(x)
    assert torch.allclose(swish(x), expected)


def test_swiglu_transform_output_shape() -> None:
    block = SwiGLUTransform(8, 16)
    out = block(torch.randn(4, 8))
    assert out.shape == (4, 16)


def test_swiglu_block_output_shape() -> None:
    block = SwiGLUBlock(8, 16)
    out = block(torch.randn(3, 8))
    assert out.shape == (3, 16)


def test_residual_block_identity_path_when_dims_match() -> None:
    block = ResidualSwiGLUBlock(8, 8)
    x = torch.randn(2, 8)
    out = block(x)
    assert out.shape == (2, 8)
    assert block.residual_proj is None


def test_residual_block_projects_when_dims_differ() -> None:
    block = ResidualSwiGLUBlock(8, 16)
    out = block(torch.randn(2, 8))
    assert out.shape == (2, 16)
    assert block.residual_proj is not None


def test_preln_residual_block_shapes() -> None:
    block = PreLNResidualSwiGLUBlock(8, 8)
    out = block(torch.randn(2, 8))
    assert out.shape == (2, 8)
    wide = PreLNResidualSwiGLUBlock(8, 32)
    assert wide(torch.randn(2, 8)).shape == (2, 32)


def test_head_forward_variants() -> None:
    for use_residual in (False, True):
        for style in ("post_ln", "pre_ln"):
            head = GatedFusionHead(
                embedding_dim=8,
                linguistic_dim=3,
                model_dim=2,
                use_residual=use_residual,
                residual_style=style,
            )
            out = head(torch.randn(5, 13))
            assert out.shape == (5, 1)


def test_head_multi_layer() -> None:
    head = GatedFusionHead(embedding_dim=8, linguistic_dim=3, model_dim=2, num_layers=3)
    out = head(torch.randn(2, 13))
    assert out.shape == (2, 1)
    assert len(head.layers) == 3


# --------------------------------------------------------------------- #
# Linguistic features (real textstat)
# --------------------------------------------------------------------- #


def test_features_shape_and_range() -> None:
    extractor = LinguisticFeatureExtractor()
    out = extractor.extract(["hello world", "complex polysyllabic lexicon"])
    assert out.shape == (2, 3)
    assert bool(((out >= 0.0) & (out <= 1.0001)).all())


def test_features_empty_batch_shape_contract() -> None:
    extractor = LinguisticFeatureExtractor()
    out = extractor.extract([])
    assert out.shape == (0, 3)


def test_features_longer_text_has_greater_length_feature() -> None:
    extractor = LinguisticFeatureExtractor()
    out = extractor.extract(["hi", "one two three four five six seven eight"])
    assert out[1, 1] > out[0, 1]


def test_features_failure_fallback_is_zero(monkeypatch) -> None:
    extractor = LinguisticFeatureExtractor()

    def broken_stat(text):
        raise RuntimeError("textstat exploded")

    import signal_llm.router.features as features_module

    monkeypatch.setattr(
        features_module,
        "_get_textstat",
        lambda: types.SimpleNamespace(
            flesch_kincaid_grade=broken_stat, polysyllabcount=broken_stat
        ),
    )
    out = extractor.extract(["anything"])
    assert torch.equal(out, torch.zeros(1, 3))


# --------------------------------------------------------------------- #
# MF strategy with fake routellm (R02/R08/R10)
# --------------------------------------------------------------------- #


class _RecordingController:
    instances: ClassVar[list[_RecordingController]] = []

    def __init__(self, routers, strong_model, weak_model, config):
        self.routers_arg = routers
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.config = config
        _RecordingController.instances.append(self)


@pytest.fixture
def fake_routellm(monkeypatch):
    _RecordingController.instances = []
    module = types.ModuleType("routellm")
    controller_module = types.ModuleType("routellm.controller")
    controller_module.Controller = _RecordingController
    module.controller = controller_module
    monkeypatch.setitem(sys.modules, "routellm", module)
    monkeypatch.setitem(sys.modules, "routellm.controller", controller_module)
    return _RecordingController


class _FakeSentenceTransformer:
    calls: ClassVar[list[dict]] = []

    def __init__(self, model_id, device=None, trust_remote_code=None):
        self.model_id = model_id
        self.device = device
        self.trust = trust_remote_code
        _FakeSentenceTransformer.calls.append(
            {"model_id": model_id, "trust": trust_remote_code, "device": device}
        )

    def to(self, x):
        return self


@pytest.fixture
def fake_st(monkeypatch):
    _FakeSentenceTransformer.calls = []
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return _FakeSentenceTransformer


def test_mf_controller_uses_configured_embedding(fake_routellm, fake_st, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    strategy = MatrixFactorizationRouterStrategy(
        _config(routellm_embedding_device="cpu"), "cpu", "linux"
    )
    with pytest.raises(RuntimeError, match="Local matrix-factorization routing is unsupported"):
        strategy.load_controller()
    assert not _RecordingController.instances
    assert not _FakeSentenceTransformer.calls


def test_mf_rejects_checkpoint_with_different_embedding(
    fake_routellm, fake_st, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    checkpoint = tmp_path / "mf.pt"
    torch.save({"dim": 32, "embedding_model": "attacker/other-model"}, checkpoint)
    strategy = MatrixFactorizationRouterStrategy(
        _config(
            routellm_checkpoint_path=str(checkpoint),
            routellm_embedding_device="cpu",
        ),
        "cpu",
        "linux",
    )
    with pytest.raises(RuntimeError, match="Local matrix-factorization routing is unsupported"):
        strategy.load_controller()


def test_mf_checkpoint_matching_embedding_still_unsupported(
    fake_routellm, fake_st, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    checkpoint = tmp_path / "mf.pt"
    torch.save({"dim": 32, "embedding_model": "nomic-ai/modernbert-embed-base"}, checkpoint)
    strategy = MatrixFactorizationRouterStrategy(
        _config(
            routellm_checkpoint_path=str(checkpoint),
            routellm_embedding_device="cpu",
        ),
        "cpu",
        "linux",
    )
    with pytest.raises(RuntimeError, match="Local matrix-factorization routing is unsupported"):
        strategy.load_controller()
    assert not _RecordingController.instances
    assert not _FakeSentenceTransformer.calls


def test_mf_openai_key_restored_after_init(fake_routellm, fake_st, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    strategy = MatrixFactorizationRouterStrategy(
        _config(routellm_embedding_device="cpu"), "cpu", "linux"
    )
    with pytest.raises(RuntimeError, match="unsupported"):
        strategy.load_controller()
    import os

    assert "OPENAI_API_KEY" not in os.environ, "dummy key must be removed (R10)"


def test_mf_openai_key_legitimate_value_preserved(fake_routellm, fake_st, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "legitimate-key")
    strategy = MatrixFactorizationRouterStrategy(
        _config(routellm_embedding_device="cpu"), "cpu", "linux"
    )
    with pytest.raises(RuntimeError, match="unsupported"):
        strategy.load_controller()
    import os

    assert os.environ["OPENAI_API_KEY"] == "legitimate-key"


def test_mf_embedding_failure_raises_clearly(fake_routellm, monkeypatch) -> None:
    class ExplodingST:
        def __init__(self, *args, **kwargs):
            raise OSError("network down")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = ExplodingST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    strategy = MatrixFactorizationRouterStrategy(
        _config(routellm_embedding_device="cpu"), "cpu", "linux"
    )
    with pytest.raises(RuntimeError, match="Local matrix-factorization routing is unsupported"):
        strategy.load_controller()


def test_mf_rejects_non_mf_router_name(fake_routellm) -> None:
    strategy = MatrixFactorizationRouterStrategy(
        _config(routellm_router_name="sota"), "cpu", "linux"
    )
    with pytest.raises(ValueError, match="only supports router_name='mf'"):
        strategy.load_controller()


def test_feature_error_logs_do_not_disclose_prompt(monkeypatch, caplog):
    import logging

    import signal_llm.router.features as features_module

    secret = "private prompt sentinel"

    def fail(text):
        raise ValueError(text)

    monkeypatch.setattr(
        features_module,
        "_get_textstat",
        lambda: types.SimpleNamespace(flesch_kincaid_grade=fail, polysyllabcount=fail),
    )
    with caplog.at_level(logging.DEBUG, logger=features_module.__name__):
        LinguisticFeatureExtractor().extract([secret])
    assert "feature extraction failed" in caplog.text
    assert secret not in caplog.text


# --------------------------------------------------------------------- #
# CoreML complexity classifier boundaries (R07) with fakes
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# CoreML complexity classifier (R07) with fakes
# --------------------------------------------------------------------- #


_N_TASKS = 9
_N_SUB = 5


def _make_coreml_fixture(monkeypatch, tmp_path, logits_overrides=None):
    """Build a fully-faked CoreML classifier around a real config.json."""
    import json as json_module

    import numpy as np

    from signal_llm.router.coreml_complexity import CoreMLComplexityClassifier

    model_dir = tmp_path / "coreml-model"
    model_dir.mkdir()
    config = {
        "task_type_map": {str(i): f"task{i}" for i in range(_N_TASKS)},
        "weights_map": {
            target: [float(j) for j in range(_N_SUB)]
            for target in (
                "creativity_scope",
                "reasoning",
                "contextual_knowledge",
                "number_of_few_shots",
                "domain_knowledge",
                "no_label_reason",
                "constraint_ct",
            )
        },
        "divisor_map": {
            target: float(sum(range(_N_SUB)))
            for target in (
                "creativity_scope",
                "reasoning",
                "contextual_knowledge",
                "number_of_few_shots",
                "domain_knowledge",
                "no_label_reason",
                "constraint_ct",
            )
        },
    }
    (model_dir / "config.json").write_text(json_module.dumps(config))
    (model_dir / "model_seq128.mlpackage").mkdir()

    class _FakeMLModel:
        compute_unit = "CPU_ONLY"

        def __init__(self, path, compute_units=None):
            self.path = path

        def predict(self, feeds):
            rng = np.random.default_rng(0)
            out = {
                name: rng.normal(size=(1, _N_TASKS if name == "task_type" else _N_SUB))
                for name in (
                    "task_type",
                    "creativity_scope",
                    "reasoning",
                    "contextual_knowledge",
                    "number_of_few_shots",
                    "domain_knowledge",
                    "no_label_reason",
                    "constraint_ct",
                )
            }
            if logits_overrides:
                out.update(logits_overrides)
            return out

    class _FakeTokenizer:
        def __call__(self, texts, **kwargs):
            seq = 8
            return {
                "input_ids": np.zeros((len(texts), seq), dtype=np.int64),
                "attention_mask": np.ones((len(texts), seq), dtype=np.int64),
            }

    fake_ct = types.ModuleType("coremltools")
    fake_ct.models = types.SimpleNamespace(MLModel=_FakeMLModel)

    class _CU:
        CPU_ONLY = "CPU_ONLY"
        CPU_AND_GPU = "CPU_AND_GPU"
        CPU_AND_NE = "CPU_AND_NE"
        ALL = "ALL"

    fake_ct.ComputeUnit = _CU
    monkeypatch.setitem(sys.modules, "coremltools", fake_ct)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda path: _FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    classifier = CoreMLComplexityClassifier(model_dir=model_dir)
    return classifier, _FakeMLModel


def test_coreml_classifier_score_is_validated(monkeypatch, tmp_path) -> None:
    classifier, _ = _make_coreml_fixture(monkeypatch, tmp_path)
    score = classifier.complexity_score("write a sorting algorithm")
    assert 0.0 <= score <= 1.0


def test_coreml_classifier_rejects_nonfinite_logits(monkeypatch, tmp_path) -> None:
    import numpy as np

    classifier, _ = _make_coreml_fixture(
        monkeypatch,
        tmp_path,
        logits_overrides={"reasoning": np.array([[float("nan")] * _N_SUB])},
    )
    with pytest.raises(ValueError, match="non-finite"):
        classifier.complexity_score("anything")


def test_coreml_classifier_task_type_output_shape(monkeypatch, tmp_path) -> None:
    classifier, _ = _make_coreml_fixture(monkeypatch, tmp_path)
    result = classifier.classify("explain quicksort")
    assert result["task_type_1"][0].startswith("task")
    assert isinstance(result["prompt_complexity_score"][0], float)


# --------------------------------------------------------------------- #
# Router __init__ exports
# --------------------------------------------------------------------- #


def test_router_package_exports() -> None:
    import signal_llm.router as router_package

    for name in ("RouterStrategy", "RouterStrategyFactory"):
        assert hasattr(router_package, name)


def test_mf_failure_clears_state_without_imports_or_environment_changes(
    fake_routellm, fake_st, monkeypatch
) -> None:
    """Reject before any optional import, credential mutation, or model load."""
    import builtins
    import os

    strategy = MatrixFactorizationRouterStrategy(
        _config(routellm_embedding_device="cpu"), "cpu", "linux"
    )
    strategy.controller = object()
    strategy._embedding_model = object()
    before = dict(os.environ)
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        assert not name.startswith(("routellm", "openai", "sentence_transformers"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(RuntimeError, match="unsupported"):
        strategy.load_controller()
    assert strategy.controller is None
    assert strategy._embedding_model is None
    assert dict(os.environ) == before
