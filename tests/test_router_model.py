"""Tests for the neural router modules (R01-R08) using real torch.

The heavy SentenceTransformer backbone is replaced with a deterministic
fake through the ``_get_sentence_transformer`` seam; the head, fusion,
checkpoint validation, and scoring logic run for real.
"""

from __future__ import annotations

import math
import types

import pytest

torch = pytest.importorskip("torch")

from signal_llm.config import LLMConfig, ModelConfig  # noqa: E402
from signal_llm.router.model import GatedFusionHead, SingleTowerStudent  # noqa: E402
from signal_llm.router.sota import SotaRouterController, SotaRouterStrategy  # noqa: E402

EMB_DIM = 16


class FakeBackbone(torch.nn.Module):
    """Deterministic SentenceTransformer stand-in."""

    def __init__(self, model_id, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model_id = model_id
        inner = torch.nn.Linear(EMB_DIM, EMB_DIM)
        inner.tokenizer = object()
        self.add_module("0", inner)
        self.eval_called = 0

    def eval(self):
        self.eval_called += 1
        return self

    def to(self, dtype_or_device):
        return self

    def encode(self, texts, convert_to_tensor=True, device=None, **kwargs):
        # Deterministic: average character codes projected into the space.
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            value = sum(ord(c) for c in text) % 97 / 97.0
            rows.append([value * ((i % 7) + 1) - 3.0 for i in range(EMB_DIM)])
        return torch.tensor(rows, dtype=torch.float32)


@pytest.fixture
def fake_st(monkeypatch):
    instances: list[FakeBackbone] = []

    def factory(model_id, **kwargs):
        instance = FakeBackbone(model_id, **kwargs)
        instances.append(instance)
        return instance

    import signal_llm.router.model as model_module

    monkeypatch.setattr(model_module, "SentenceTransformer", factory, raising=False)
    monkeypatch.setattr(model_module, "_get_sentence_transformer", lambda: factory)
    return instances


def _fake_features(monkeypatch):
    """Deterministic linguistic features (3-dim) without textstat."""
    import signal_llm.router.model as model_module

    class FakeExtractor:
        def extract(self, text_list):
            rows = [[0.1, 0.2, 0.3] for _ in text_list]
            return torch.tensor(rows, dtype=torch.float32)

    monkeypatch.setattr(model_module, "LinguisticFeatureExtractor", FakeExtractor)


def _make_student(monkeypatch, fake_st, **kwargs) -> SingleTowerStudent:
    _fake_features(monkeypatch)
    return SingleTowerStudent(
        model_id="fake/embed-model",
        use_int4=False,
        device=torch.device("cpu"),
        dtype="float32",
        **kwargs,
    )


# --------------------------------------------------------------------- #
# R01: forward shape contract
# --------------------------------------------------------------------- #


def test_student_forward_matches_head_input_width(monkeypatch, fake_st):
    student = _make_student(monkeypatch, fake_st)
    out = student(["hello world"])
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()

    out_batch = student(["a", "b", "c"])
    assert out_batch.shape == (3, 1)


def test_student_forward_includes_model_pair_features(monkeypatch, fake_st):
    student = _make_student(monkeypatch, fake_st)
    # The head's first layer must accept emb + 3 linguistic + 2 model feats.
    first = student.head.layers[0]
    input_dim = getattr(first, "input_dim", None)
    if input_dim is None:
        input_dim = student.head.input_dim
    assert input_dim == EMB_DIM + 3 + 2


def test_head_input_dim_documents_model_features():
    head = GatedFusionHead(embedding_dim=EMB_DIM, linguistic_dim=3, model_dim=2)
    assert head.input_dim == EMB_DIM + 3 + 2


def test_model_features_buffer_is_persistent(monkeypatch, fake_st):
    student = _make_student(monkeypatch, fake_st)
    state = student.state_dict()
    assert "model_features" in state
    assert state["model_features"].shape == (2,)
    assert student.backbone.kwargs["local_files_only"] is True


# --------------------------------------------------------------------- #
# Controller score validation (R07)
# --------------------------------------------------------------------- #


def test_controller_returns_finite_score_in_unit_range(monkeypatch, fake_st):
    student = _make_student(monkeypatch, fake_st)
    controller = SotaRouterController(student)
    score = controller.calculate_strong_win_rate("write me a parser")
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_controller_rejects_nonfinite_logits(monkeypatch, fake_st):
    student = _make_student(monkeypatch, fake_st)

    def poison(texts):
        return torch.tensor([[float("nan")]])

    student.forward = poison
    controller = SotaRouterController(student)
    with pytest.raises(RuntimeError):
        controller.calculate_strong_win_rate("x")


# --------------------------------------------------------------------- #
# R03: strict checkpoint validation
# --------------------------------------------------------------------- #


def _config(**overrides) -> LLMConfig:
    defaults = {
        "router_priority": [ModelConfig(name="r", repo_id="org/r", backend="transformers")],
        "semantic_priority": [ModelConfig(name="s", repo_id="org/s", backend="transformers")],
        "agent_priority": [ModelConfig(name="a", repo_id="org/a", backend="transformers")],
        "router_type": "routellm",
        "routellm_router_name": "sota",
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


@pytest.mark.parametrize("use_int4", [None, False, True])
def test_sota_factory_quantization_is_explicit(monkeypatch, tmp_path, use_int4):
    from signal_llm.router import sota
    from signal_llm.router.factory import RouterStrategyFactory

    options = {} if use_int4 is None else {"routellm_use_int4": use_int4}
    config = _config(routellm_checkpoint_path="head.pt", **options)
    (tmp_path / "head.pt").touch()
    strategy = RouterStrategyFactory(config, "cuda:1", "linux", tmp_path).create_strategy()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    calls = []

    def student(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(sota, "SingleTowerStudent", student)
    monkeypatch.setattr(strategy, "_load_checkpoint", lambda *args: None)
    strategy.load_controller()
    assert calls[0]["use_int4"] is (use_int4 is True)
    assert calls[0]["device"] == torch.device("cuda:1")


def test_sota_int4_requires_cuda():
    strategy = SotaRouterStrategy(_config(routellm_use_int4=True), "cpu", "linux")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        strategy.load_controller()
    assert strategy.model is None
    assert strategy.controller is None


@pytest.mark.parametrize("value", [1, "false", None])
def test_sota_quantization_option_requires_bool(value):
    with pytest.raises(ValueError, match="routellm_use_int4 must be a bool"):
        _config(routellm_use_int4=value)


def _make_strategy(monkeypatch, fake_st, tmp_path, **config_overrides):
    _fake_features(monkeypatch)
    config = _config(**config_overrides)
    strategy = SotaRouterStrategy(config, "cpu", "linux")
    strategy.models_dir = tmp_path
    return strategy


def _write_head_checkpoint(path, student) -> None:
    _save_checkpoint(
        path,
        {
            k: v
            for k, v in student.state_dict().items()
            if k.startswith("head.") or k == "model_features"
        },
    )


def _save_checkpoint(path, state, kind="head"):
    torch.save({"schema_version": 1, "kind": kind, "state_dict": state}, path)


def test_valid_head_checkpoint_loads_strictly(monkeypatch, fake_st, tmp_path):
    reference = _make_student(monkeypatch, fake_st)
    checkpoint = tmp_path / "custom_head.pt"
    _write_head_checkpoint(checkpoint, reference)

    strategy = _make_strategy(
        monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="custom_head.pt"
    )
    controller = strategy.load_controller()
    score = controller.calculate_strong_win_rate("test prompt")
    assert 0.0 <= score <= 1.0
    # Every head tensor was restored from the checkpoint.
    for key, value in reference.head.state_dict().items():
        assert torch.equal(strategy.model.head.state_dict()[key], value)


def test_partial_head_checkpoint_is_rejected(monkeypatch, fake_st, tmp_path):
    reference = _make_student(monkeypatch, fake_st)
    state = reference.state_dict()
    # Drop every final-layer weight: strict loading must reject this.
    partial = {k: v for k, v in state.items() if "final_score" not in k}
    assert any("final_score" in k for k in state), "sanity: head has final_score keys"
    checkpoint = tmp_path / "partial.pt"
    _save_checkpoint(checkpoint, partial, "full")

    strategy = _make_strategy(monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="partial.pt")
    with pytest.raises(RuntimeError):
        strategy.load_controller()
    assert strategy.model is None, "invalid checkpoint must leave strategy unloaded (R03)"


def test_unrelated_checkpoint_is_rejected(monkeypatch, fake_st, tmp_path):
    checkpoint = tmp_path / "garbage.pt"
    torch.save({"unrelated.weight": torch.zeros(3)}, checkpoint)
    strategy = _make_strategy(monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="garbage.pt")
    with pytest.raises(RuntimeError):
        strategy.load_controller()
    assert strategy.model is None


def test_nonfinite_checkpoint_tensor_is_rejected(monkeypatch, fake_st, tmp_path):
    reference = _make_student(monkeypatch, fake_st)
    state = reference.state_dict()
    first_head_key = next(k for k in state if k.startswith("head."))
    state[first_head_key] = torch.full_like(state[first_head_key], float("nan"))
    checkpoint = tmp_path / "nan.pt"
    _save_checkpoint(checkpoint, state, "full")

    strategy = _make_strategy(monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="nan.pt")
    with pytest.raises(RuntimeError):
        strategy.load_controller()
    assert strategy.model is None


def test_shape_incompatible_checkpoint_is_rejected(monkeypatch, fake_st, tmp_path):
    reference = _make_student(monkeypatch, fake_st)
    state = reference.state_dict()
    first_head_key = next(k for k in state if k.startswith("head."))
    state[first_head_key] = torch.zeros(state[first_head_key].shape[0] + 1)
    checkpoint = tmp_path / "shape.pt"
    _save_checkpoint(checkpoint, state, "full")

    strategy = _make_strategy(monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="shape.pt")
    with pytest.raises(RuntimeError):
        strategy.load_controller()
    assert strategy.model is None


# --------------------------------------------------------------------- #
# R04: explicit checkpoint resolution
# --------------------------------------------------------------------- #


def test_explicit_relative_checkpoint_resolves_against_models_dir(monkeypatch, fake_st, tmp_path):
    reference = _make_student(monkeypatch, fake_st)
    (tmp_path / "nested").mkdir()
    checkpoint = tmp_path / "nested" / "head.pt"
    _write_head_checkpoint(checkpoint, reference)

    strategy = _make_strategy(
        monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="nested/head.pt"
    )
    controller = strategy.load_controller()
    assert controller is not None


def test_explicit_missing_checkpoint_fails_discovery_does_not_substitute(
    monkeypatch, fake_st, tmp_path
):
    # A generated-name checkpoint exists, but the explicit path is absent:
    # the explicit configuration must fail, not silently load the other one.
    reference = _make_student(monkeypatch, fake_st)
    generated = tmp_path / "sota_router_fake_embed_model_fp32.pt"
    _write_head_checkpoint(generated, reference)

    strategy = _make_strategy(
        monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="missing/head.pt"
    )
    with pytest.raises(FileNotFoundError):
        strategy.load_controller()


def test_generated_discovery_applies_only_without_explicit_path(monkeypatch, fake_st, tmp_path):
    reference = _make_student(monkeypatch, fake_st)
    generated = tmp_path / "sota_router_fake_embed_model_fp32.pt"
    _write_head_checkpoint(generated, reference)

    strategy = _make_strategy(
        monkeypatch, fake_st, tmp_path, routellm_embedding_model="fake/embed-model"
    )
    controller = strategy.load_controller()
    assert controller is not None


# --------------------------------------------------------------------- #
# R05: device parsing
# --------------------------------------------------------------------- #


def test_resolve_device_preserves_cuda_index():
    strategy = SotaRouterStrategy.__new__(SotaRouterStrategy)
    strategy.device = "cuda:0"
    if torch.cuda.is_available():
        assert strategy._resolve_device() == torch.device("cuda:0")
    else:
        with pytest.raises(ValueError, match="unavailable"):
            strategy._resolve_device()


def test_resolve_device_rejects_invalid_names():
    strategy = SotaRouterStrategy.__new__(SotaRouterStrategy)
    strategy.device = "tpu"
    with pytest.raises(ValueError, match="invalid device"):
        strategy._resolve_device()


def test_resolve_device_auto_cpu(monkeypatch):
    strategy = SotaRouterStrategy.__new__(SotaRouterStrategy)
    strategy.device = "auto"
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(torch, "cuda", fake_cuda)
    monkeypatch.setattr(
        torch,
        "backends",
        types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
            cuda=fake_cuda,
        ),
    )
    assert strategy._resolve_device() == torch.device("cpu")


# --------------------------------------------------------------------- #
# R06: quantization state tracking
# --------------------------------------------------------------------- #


def test_int4_state_is_tracked(monkeypatch, fake_st):
    _fake_features(monkeypatch)
    student = SingleTowerStudent(
        model_id="fake/embed-model",
        use_int4=False,
        device=torch.device("cpu"),
        dtype="float32",
    )
    # Without quantization requested the flag is False and honest.
    assert student.int4_applied is False


# --------------------------------------------------------------------- #
# GatedFusionHead input validation
# --------------------------------------------------------------------- #


def test_head_rejects_nonpositive_dims():
    with pytest.raises((ValueError, AssertionError)):
        GatedFusionHead(embedding_dim=0, linguistic_dim=3, model_dim=2)


def test_head_rejects_nan_dropout():
    with pytest.raises((ValueError, AssertionError)):
        GatedFusionHead(
            embedding_dim=EMB_DIM, linguistic_dim=3, model_dim=2, dropout_rate=float("nan")
        )


def test_empty_batch_policy(monkeypatch, fake_st):
    student = _make_student(monkeypatch, fake_st)
    out = student([])
    # Empty input yields an empty score tensor with defined shape.
    assert out.shape[0] == 0


def test_full_checkpoint_restores_backbone_and_pair(monkeypatch, fake_st, tmp_path):
    reference = _make_student(monkeypatch, fake_st)
    with torch.no_grad():
        reference.backbone._modules["0"].weight.fill_(0.75)
        reference.model_features.copy_(torch.tensor([1.5, -0.25]))
    checkpoint = tmp_path / "full.pt"
    _save_checkpoint(checkpoint, reference.state_dict(), "full")
    strategy = _make_strategy(monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="full.pt")
    strategy.load_controller()
    for key, value in reference.state_dict().items():
        assert torch.equal(strategy.model.state_dict()[key], value), key


@pytest.mark.parametrize(
    "defect",
    [
        "missing_pair",
        "pair_matrix",
        "pair_nan",
        "pair_type",
        "unknown",
        "missing_backbone",
        "schema",
        "kind",
        "metadata",
        "dtype",
        "head_with_backbone",
    ],
)
def test_checkpoint_contract_rejects_defects(monkeypatch, fake_st, tmp_path, defect):
    reference = _make_student(monkeypatch, fake_st)
    state = reference.state_dict()
    artifact = {"schema_version": 1, "kind": "full", "state_dict": state}
    if defect == "missing_pair":
        del state["model_features"]
    elif defect == "pair_matrix":
        state["model_features"] = torch.zeros(1, 2)
    elif defect == "pair_nan":
        state["model_features"] = torch.tensor([float("nan"), 0.0])
    elif defect == "pair_type":
        state["model_features"] = [0.0, 0.0]
    elif defect == "unknown":
        state["unknown"] = torch.zeros(1)
    elif defect == "missing_backbone":
        del state[next(k for k in state if k.startswith("backbone."))]
    elif defect == "schema":
        artifact["schema_version"] = True
    elif defect == "kind":
        artifact["kind"] = "other"
    elif defect == "metadata":
        artifact["extra"] = "unknown"
    elif defect == "dtype":
        state["model_features"] = torch.zeros(2, dtype=torch.int64)
    elif defect == "head_with_backbone":
        artifact["kind"] = "head"
    torch.save(artifact, tmp_path / "bad.pt")
    strategy = _make_strategy(monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="bad.pt")
    strategy.controller = object()
    strategy.model = reference
    with pytest.raises(RuntimeError):
        strategy.load_controller()
    assert strategy.model is None
    assert strategy.controller is None


def test_absent_checkpoint_fails_before_backbone_loading(monkeypatch, fake_st, tmp_path):
    strategy = _make_strategy(monkeypatch, fake_st, tmp_path)
    strategy.controller = object()
    with pytest.raises(RuntimeError, match="requires a trained checkpoint"):
        strategy.load_controller()
    assert not fake_st
    assert strategy.model is None and strategy.controller is None


@pytest.mark.parametrize("failure", ["discovery", "construction"])
def test_failed_reload_clears_both_references(monkeypatch, fake_st, tmp_path, failure):
    reference = _make_student(monkeypatch, fake_st)
    _write_head_checkpoint(tmp_path / "valid.pt", reference)
    strategy = _make_strategy(monkeypatch, fake_st, tmp_path, routellm_checkpoint_path="valid.pt")
    strategy.load_controller()
    if failure == "discovery":
        strategy.config.routellm_checkpoint_path = "absent.pt"
    else:
        import signal_llm.router.sota as sota_module

        def broken(**kwargs):
            raise RuntimeError("constructor failed")

        monkeypatch.setattr(sota_module, "SingleTowerStudent", broken)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        strategy.load_controller()
    assert strategy.model is None and strategy.controller is None


def test_forward_moves_pair_to_effective_embedding_device(monkeypatch, fake_st):
    student = _make_student(monkeypatch, fake_st)
    student.backbone.encode = lambda texts, **kw: torch.empty(
        len(texts), EMB_DIM, device="meta", dtype=torch.float16
    )
    captured = []

    class Capture(torch.nn.Module):
        def forward(self, combined):
            captured.append(combined)
            return combined[:, :1]

    student.head = Capture()
    student(["test"])
    assert captured[0].device.type == "meta"
    assert captured[0].dtype == torch.float32
    assert student.model_features.device.type == "cpu"
