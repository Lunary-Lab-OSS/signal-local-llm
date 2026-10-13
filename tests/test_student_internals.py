"""Coverage for SingleTowerStudent dtype/quantization branches (R06)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import signal_llm.router.model as model_module  # noqa: E402
from signal_llm.router.model import SingleTowerStudent  # noqa: E402


class QuantizableBackbone:
    """Fake SentenceTransformer exposing a real nn.Module transformer."""

    def __init__(self, model_id, **kwargs):
        self.kwargs = kwargs
        inner = torch.nn.Sequential(torch.nn.Linear(EMB, EMB), torch.nn.Linear(EMB, EMB))
        self._modules = {"0": types.SimpleNamespace(auto_model=inner, tokenizer=object())}

    def eval(self):
        return self

    def to(self, target):
        return self

    def encode(self, texts, convert_to_tensor=True, device=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[(sum(ord(c) for c in t) % 13) / 13.0] * EMB for t in texts]
        return torch.tensor(rows, dtype=torch.float32)


EMB = 8


@pytest.fixture
def fake_st(monkeypatch):
    instances: list[QuantizableBackbone] = []

    def factory(model_id, **kwargs):
        instance = QuantizableBackbone(model_id, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(model_module, "_get_sentence_transformer", lambda: factory)
    return instances


def _fake_features(monkeypatch):
    class FakeExtractor:
        def extract(self, text_list):
            return torch.zeros((len(text_list), 3), dtype=torch.float32)

    monkeypatch.setattr(model_module, "LinguisticFeatureExtractor", FakeExtractor)


def test_dtype_bfloat16_rejected_on_cpu(monkeypatch, fake_st):
    _fake_features(monkeypatch)
    student = SingleTowerStudent(
        model_id="fake/m",
        use_int4=False,
        device=torch.device("cpu"),
        dtype="bfloat16",
    )
    assert student.backbone.kwargs.get("token") is None


def test_dtype_float16_warning_on_cpu(monkeypatch, fake_st):
    _fake_features(monkeypatch)
    SingleTowerStudent(
        model_id="fake/m", use_int4=False, device=torch.device("cpu"), dtype="float16"
    )


def test_dtype_unsupported_falls_back(monkeypatch, fake_st):
    _fake_features(monkeypatch)
    SingleTowerStudent(model_id="fake/m", use_int4=False, device=torch.device("cpu"), dtype="int8")


def test_int4_quantization_path_runs_and_tracks(monkeypatch, fake_st):
    monkeypatch.setattr(model_module, "TORCHAO_AVAILABLE", True)
    calls = []
    monkeypatch.setattr(model_module, "int4_weight_only", lambda **kw: kw, raising=False)
    monkeypatch.setattr(model_module, "quantize_", lambda *args: calls.append(args), raising=False)
    _fake_features(monkeypatch)
    student = SingleTowerStudent(
        model_id="fake/m",
        use_int4=True,
        device=torch.device("cpu"),
        dtype="float32",
    )
    # R06: success is claimed only after the smoke encode passes.
    assert student.int4_applied is True
    assert len(calls) == 1


def test_int4_without_torchao_fails(monkeypatch, fake_st):
    monkeypatch.setattr(model_module, "TORCHAO_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="TorchAO is unavailable"):
        _fake_features(monkeypatch)
        SingleTowerStudent(
            model_id="fake/m",
            use_int4=True,
            device=torch.device("cpu"),
            dtype="float32",
        )


def test_auto_device_selects_cpu(monkeypatch, fake_st):
    _fake_features(monkeypatch)
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    original_cuda = torch.cuda
    monkeypatch.setattr(torch, "cuda", fake_cuda, raising=False)
    try:
        student = SingleTowerStudent(
            model_id="fake/m", use_int4=False, device=None, dtype="float32"
        )
        assert student.device.type == "cpu"
    finally:
        monkeypatch.setattr(torch, "cuda", original_cuda, raising=False)


def test_backbone_without_inner_tokenizer_uses_autotokenizer(monkeypatch, fake_st):
    class NoTokenizerBackbone(QuantizableBackbone):
        def __init__(self, model_id, **kwargs):
            super().__init__(model_id, **kwargs)
            self._modules = {"0": types.SimpleNamespace(auto_model=self._modules["0"].auto_model)}

    monkeypatch.setattr(model_module, "_get_sentence_transformer", lambda: NoTokenizerBackbone)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name, trust_remote_code=False, token=None, local_files_only=True: (
            object()
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    _fake_features(monkeypatch)
    SingleTowerStudent(
        model_id="fake/m", use_int4=False, device=torch.device("cpu"), dtype="float32"
    )


@pytest.mark.parametrize("failure", ["mutation", "smoke", "nan"])
def test_failed_quantization_never_returns_student(monkeypatch, fake_st, failure):
    _fake_features(monkeypatch)
    monkeypatch.setattr(model_module, "TORCHAO_AVAILABLE", True)
    monkeypatch.setattr(model_module, "int4_weight_only", lambda **kw: kw, raising=False)
    calls = []

    def quantize(transformer, config):
        calls.append(transformer)
        with torch.no_grad():
            transformer[0].weight.fill_(42)
        if failure == "mutation":
            raise RuntimeError("AffineQuantizedTensor shallow_copy")
        if failure == "smoke":

            def broken(*args, **kwargs):
                raise RuntimeError("broken quantized kernel")

            fake_st[-1].encode = broken
        else:
            fake_st[-1].encode = lambda *a, **kw: torch.full((1, EMB), float("nan"))

    monkeypatch.setattr(model_module, "quantize_", quantize, raising=False)
    with pytest.raises(RuntimeError, match="INT4 initialization failed"):
        SingleTowerStudent("fake/m", use_int4=True, device=torch.device("cpu"))
    assert len(calls) == 1
