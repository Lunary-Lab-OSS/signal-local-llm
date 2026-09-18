"""Targeted coverage for loader exllama/vllm paths and component branches."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from signal_llm.component import LLMComponent, LoadedModel
from signal_llm.config import LLMConfig, ModelConfig, SpeculativeDecodingConfig
from signal_llm.loader import LocalLLMModelLoader, ModelLoadError


def _model(name: str, repo: str, backend: str, **kw) -> ModelConfig:
    return ModelConfig(name=name, repo_id=repo, backend=backend, **kw)


def _config(**overrides) -> LLMConfig:
    defaults = {
        "router_priority": [_model("r", "org/r", "transformers")],
        "semantic_priority": [_model("s", "org/s", "transformers")],
        "agent_priority": [_model("a", "org/a", "transformers")],
        "max_tokens": 512,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _loader(tmp_path, device="cpu", cuda=True) -> LocalLLMModelLoader:
    loader = LocalLLMModelLoader(models_dir=tmp_path / "m", cache_dir=tmp_path / "c", device=device)
    loader._has_cuda = lambda: cuda
    loader._cuda_index_available = lambda index: cuda and index in (0, 1)
    return loader


def _seed(tmp_path, repo="org/m") -> Path:
    loader_local = None
    return loader_local


# --------------------------------------------------------------------- #
# ExLlamaV2 backend
# --------------------------------------------------------------------- #


def _install_fake_exllamav2(monkeypatch, recorder):
    exl = types.ModuleType("exllamav2")
    exl.ExLlamaV2 = lambda config: types.SimpleNamespace(load=lambda: None)
    exl.ExLlamaV2Cache = lambda model, max_seq_len=None: object()
    exl.ExLlamaV2Config = lambda path: recorder.setdefault("config_path", path)
    exl.ExLlamaV2Tokenizer = lambda config: object()

    generator_module = types.ModuleType("exllamav2.generator")

    class FakeGenerator:
        def __init__(
            self, model, cache, tokenizer, *, draft_model=None, draft_cache=None, num_draft_tokens=4
        ):
            recorder["generator_kwargs"] = {
                "draft_model": draft_model,
                "draft_cache": draft_cache,
                "num_draft_tokens": num_draft_tokens,
            }

        def generate(self, *args, **kwargs):
            return "out"

    generator_module.ExLlamaV2DynamicGenerator = FakeGenerator
    exl.generator = generator_module
    monkeypatch.setitem(sys.modules, "exllamav2", exl)
    monkeypatch.setitem(sys.modules, "exllamav2.generator", generator_module)

    fake_transformers = types.ModuleType("transformers")

    def tokenizer_from(name, trust_remote_code=False):
        recorder["tokenizer_trust"] = trust_remote_code
        return object()

    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=tokenizer_from)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_exllama_backend_builds_generator_and_tokenizer(monkeypatch, tmp_path):
    recorder: dict = {}
    _install_fake_exllamav2(monkeypatch, recorder)
    loader = _loader(tmp_path, device="cuda")
    local = loader._get_model_path("org/m", None)
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"")

    config = _model("m", "org/m", "exllamav2", device="cuda")
    handle = loader._load_exllamav2(local, config, None, None)
    assert isinstance(handle, tuple)
    assert recorder["tokenizer_trust"] is False, "trust_remote_code must default off"
    assert recorder["generator_kwargs"]["draft_model"] is None
    assert handle.draft_model_path is None


def test_exllama_backend_passes_speculative_config(monkeypatch, tmp_path):
    recorder: dict = {}
    _install_fake_exllamav2(monkeypatch, recorder)
    loader = _loader(tmp_path, device="cuda")
    local = loader._get_model_path("org/m", None)
    (local / "config.json").write_text("{}")

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}")

    config = _model("m", "org/m", "exllamav2", device="cuda")
    spec = SpeculativeDecodingConfig(enabled=True, num_speculative_tokens=9)
    handle = loader._load_exllamav2(local, config, draft, spec)
    assert recorder["generator_kwargs"]["num_draft_tokens"] == 9
    assert handle.draft_model_path == draft.resolve()


def test_exllama_backend_requires_config_json(monkeypatch, tmp_path):
    recorder: dict = {}
    _install_fake_exllamav2(monkeypatch, recorder)
    loader = _loader(tmp_path, device="cuda")
    local = loader._get_model_path("org/m", None)
    local.mkdir(parents=True, exist_ok=True)

    config = _model("m", "org/m", "exllamav2", device="cuda")
    with pytest.raises(ModelLoadError, match="Config file not found"):
        loader._load_exllamav2(local, config, None, None)


# --------------------------------------------------------------------- #
# vLLM backend
# --------------------------------------------------------------------- #


def _install_strict_vllm(monkeypatch, recorder):
    class StrictLLM:
        def __init__(
            self,
            *,
            model,
            tensor_parallel_size,
            gpu_memory_utilization,
            max_model_len,
            dtype,
            quantization,
            speculative_config,
            trust_remote_code,
        ):
            recorder.update(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                quantization=quantization,
                speculative_config=speculative_config,
                trust_remote_code=trust_remote_code,
            )
            if speculative_config is not None:
                assert type(speculative_config) is dict
                assert set(speculative_config) == {"model", "num_speculative_tokens"}
            self.tokenizer = object()

        def get_tokenizer(self):
            return self.tokenizer

    module = types.ModuleType("vllm")
    module.LLM = StrictLLM
    monkeypatch.setitem(sys.modules, "vllm", module)


def test_vllm_backend_builds_llm(monkeypatch, tmp_path):
    recorder: dict = {}
    _install_strict_vllm(monkeypatch, recorder)

    loader = _loader(tmp_path, device="cuda")
    local = loader._get_model_path("org/m", None)
    (local / "config.json").write_text("{}")

    config = _model("m", "org/m", "vllm", device="cuda")
    handle = loader._load_vllm(local, config, None, None)
    assert handle[1] is handle[0].get_tokenizer()
    assert handle.draft_model_path is None
    assert recorder["model"] == str(local)
    assert recorder["quantization"] is None

    # With a draft + spec config, speculative settings propagate.
    draft = loader._get_model_path("org/draft", None)
    (draft / "config.json").write_text("{}")
    spec = SpeculativeDecodingConfig(enabled=True, num_speculative_tokens=4)
    handle = loader._load_vllm(local, config, draft, spec)
    assert recorder["speculative_config"] == {
        "model": str(draft.resolve()),
        "num_speculative_tokens": 4,
    }
    assert handle.draft_model_path == draft.resolve()


def test_vllm_backend_detects_awq(monkeypatch, tmp_path):
    recorder: dict = {}
    _install_strict_vllm(monkeypatch, recorder)

    loader = _loader(tmp_path, device="cuda")
    awq_dir = loader._get_model_path("org/m-AWQ", None)
    awq_dir.mkdir(parents=True, exist_ok=True)
    (awq_dir / "config.json").write_text("{}")

    config = _model("m", "org/m-AWQ", "vllm", device="cuda")
    loader._load_vllm(awq_dir, config, None, None)
    assert recorder["quantization"] == "awq"


def test_vllm_backend_uses_system_gpu_config(monkeypatch, tmp_path):
    recorder: dict = {}
    _install_strict_vllm(monkeypatch, recorder)

    loader = _loader(tmp_path, device="cuda")
    loader.system_config = types.SimpleNamespace(
        gpu_memory=types.SimpleNamespace(llm_memory_utilization=0.7)
    )
    local = loader._get_model_path("org/m", None)
    (local / "config.json").write_text("{}")
    config = _model("m", "org/m", "vllm", device="cuda")
    loader._load_vllm(local, config, None, None)
    assert recorder["gpu_memory_utilization"] == 0.7


# --------------------------------------------------------------------- #
# Component branches
# --------------------------------------------------------------------- #


def _generation_handle(reply):
    import torch

    calls = {}

    class Tokenizer:
        chat_template = "test"
        eos_token_id = 0

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, enable_thinking
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            calls["messages"] = messages
            calls["thinking"] = enable_thinking
            return "formatted prompt"

        def __call__(self, prompt, *, return_tensors, return_token_type_ids):
            assert prompt == "formatted prompt"
            assert return_tensors == "pt"
            assert return_token_type_ids is False
            return {"input_ids": torch.tensor([[11, 12]])}

        def decode(self, tokens, *, skip_special_tokens):
            assert tokens.tolist() == [13]
            assert skip_special_tokens is False
            return reply

    class Model:
        device = "cpu"

        def generate(
            self, *, input_ids, max_new_tokens, do_sample, temperature, top_p, top_k, pad_token_id
        ):
            assert input_ids.tolist() == [[11, 12]]
            assert do_sample is True
            assert temperature > 0
            assert 0 < top_p <= 1
            assert top_k > 0
            assert pad_token_id == 0
            calls["max_new_tokens"] = max_new_tokens
            return torch.tensor([[11, 12, 13]])

    return Model(), Tokenizer(), calls


def test_generate_router_with_loaded_model(monkeypatch):
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    component = LLMComponent(_config(), model_loader=loader, device="cpu")
    handle, tokenizer, calls = _generation_handle("agentic")
    component.router_model = LoadedModel(
        handle=handle, name="r", backend="transformers", tokenizer=tokenizer
    )
    assert component.generate_router("classify") == "agentic"
    assert calls["max_new_tokens"] == 10
    assert calls["thinking"] is False
    assert calls["messages"][-1] == {"role": "user", "content": "classify"}


def test_generate_semantic_uses_semantic_tokens(monkeypatch):
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    component = LLMComponent(_config(semantic_thinking_tokens=8), model_loader=loader, device="cpu")
    handle, tokenizer, calls = _generation_handle("semantic reply")
    component.semantic_model = LoadedModel(
        handle=handle, name="s", backend="transformers", tokenizer=tokenizer
    )
    assert component.generate_semantic("hello") == "semantic reply"
    assert calls["max_new_tokens"] == 136
    assert calls["thinking"] is True
    assert calls["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_agent_phi4_ignores_thinking(monkeypatch):
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    component = LLMComponent(_config(), model_loader=loader, device="cpu")
    handle, tokenizer, calls = _generation_handle("agent reply")
    component.agent_model = LoadedModel(
        handle=handle, name="phi-4-reasoning", backend="transformers", tokenizer=tokenizer
    )
    # phi-4 manages its own thinking; the explicit budget is ignored.
    assert component.generate_agent("question", thinking_tokens=500) == "agent reply"
    assert calls["max_new_tokens"] == 128
    assert calls["thinking"] is False


def test_default_loader_requires_dirs():
    config = LLMConfig(
        router_priority=[_model("r", "org/r", "transformers")],
        semantic_priority=[_model("s", "org/s", "transformers")],
        agent_priority=[_model("a", "org/a", "transformers")],
    )
    with pytest.raises(ValueError, match="models_dir and cache_dir"):
        LLMComponent(config, model_loader=None, device="cpu")


def test_default_loader_builds_from_config_dirs(tmp_path):
    config = LLMConfig(
        router_priority=[_model("r", "org/r", "transformers")],
        semantic_priority=[_model("s", "org/s", "transformers")],
        agent_priority=[_model("a", "org/a", "transformers")],
        models_dir=str(tmp_path / "m"),
        cache_dir=str(tmp_path / "c"),
    )
    component = LLMComponent(config, model_loader=None, device="cpu")
    assert component.model_loader is not None


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("model_device", ["cuda:1", "auto"])
def test_transformers_uses_exact_cuda_index(monkeypatch, tmp_path, quantized, model_device):
    recorder = {}
    module = types.ModuleType("transformers")

    def load_tokenizer(path, *, trust_remote_code):
        assert trust_remote_code is False
        return object()

    def load_model(path, *, device_map, torch_dtype, trust_remote_code, quantization_config=None):
        assert device_map == {"": "cuda:1"}
        recorder["quantization_config"] = quantization_config
        return object()

    class BitsAndBytesConfig:
        def __init__(self, *, load_in_8bit):
            assert load_in_8bit is True

    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=load_tokenizer)
    module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load_model)
    module.BitsAndBytesConfig = BitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", module)
    loader = _loader(tmp_path, device="cuda:1")
    local = loader._get_model_path("org/m")
    (local / "config.json").write_text("{}")
    config = _model("m", "org/m", "transformers", device=model_device, quantized=quantized)
    loader._load_transformers(local, config)
    assert isinstance(recorder["quantization_config"], BitsAndBytesConfig) is quantized


def test_vllm_missing_draft_does_not_guess_remote_repo(monkeypatch, tmp_path):
    recorder = {}
    _install_strict_vllm(monkeypatch, recorder)
    loader = _loader(tmp_path, device="cuda")
    local = loader._get_model_path("org/m")
    (local / "config.json").write_text("{}")
    config = _model("m", "org/m", "vllm", device="cuda")
    with pytest.raises(ModelLoadError, match="draft_model_path"):
        loader._load_vllm(local, config, tmp_path / "missing", SpeculativeDecodingConfig())
    assert recorder == {}


def test_exllama_failed_draft_has_no_active_draft_metadata(monkeypatch, tmp_path):
    recorder = {}
    _install_fake_exllamav2(monkeypatch, recorder)
    loader = _loader(tmp_path, device="cuda")
    local = loader._get_model_path("org/m")
    (local / "config.json").write_text("{}")
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    original = sys.modules["exllamav2"].ExLlamaV2Config

    def load_config(path):
        if path == str(draft):
            raise ValueError("unsupported draft")
        return original(path)

    monkeypatch.setattr(sys.modules["exllamav2"], "ExLlamaV2Config", load_config)
    handle = loader._load_exllamav2(
        local, _model("m", "org/m", "exllamav2", device="cuda"), draft, SpeculativeDecodingConfig()
    )
    assert handle.draft_model_path is None
    assert recorder["generator_kwargs"]["draft_model"] is None
