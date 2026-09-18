"""Backend generation-path tests via module seams (vLLM, ExLlamaV2, MLX).

These run the real dispatch logic in `LLMComponent._generate` with strict
fake backend modules injected into `sys.modules`, asserting the exact
arguments each backend receives.
"""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest

from signal_llm.component import LLMComponent, LoadedModel
from signal_llm.config import LLMConfig, ModelConfig


def _model(name: str, repo: str, backend: str) -> ModelConfig:
    return ModelConfig(name=name, repo_id=repo, backend=backend)


def _config() -> LLMConfig:
    return LLMConfig(
        router_priority=[_model("r", "org/r", "transformers")],
        semantic_priority=[_model("s", "org/s", "transformers")],
        agent_priority=[_model("a", "org/a", "transformers")],
        max_tokens=512,
        temperature=0.4,
        top_p=0.8,
    )


def _component() -> LLMComponent:
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    return LLMComponent(_config(), model_loader=loader, device="cpu", platform="linux")


# --------------------------------------------------------------------- #
# vLLM
# --------------------------------------------------------------------- #


class _FakeSamplingParams:
    instances: ClassVar[list[_FakeSamplingParams]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeSamplingParams.instances.append(self)


class _FakeVllmOutput:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text)]


class _FakeVllmModel:
    def __init__(self):
        self.llm_engine = object()
        self.generate_calls: list[tuple[list, Any]] = []

    def generate(self, prompts, sampling_params):
        self.generate_calls.append((prompts, sampling_params))
        return [_FakeVllmOutput("<think>why</think>vllm answer")]


@pytest.fixture
def fake_vllm(monkeypatch):
    _FakeSamplingParams.instances = []
    module = types.ModuleType("vllm")
    module.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def test_vllm_generation_receives_budget_and_template(fake_vllm) -> None:
    component = _component()
    model = _FakeVllmModel()
    tokenizer = types.SimpleNamespace(
        chat_template="chatml",
        apply_chat_template=lambda messages, **kw: "<formatted>",
    )
    loaded = LoadedModel(handle=model, name="qwen3-8b", backend="vllm", tokenizer=tokenizer)
    budget = component._resolve_budget("agent", thinking_tokens=32)
    result = component._generate(loaded, "write a haiku", budget, system_prompt="be terse")

    assert result == "vllm answer"
    prompts, params = model.generate_calls[0]
    assert prompts == ["<formatted>"]
    assert params.kwargs["temperature"] == 0.4
    assert params.kwargs["top_p"] == 0.8
    assert params.kwargs["max_tokens"] == budget["answer_tokens"] + budget["thinking_tokens"]
    assert params.kwargs["max_tokens"] <= 512


def test_vllm_chatml_fallback_without_tokenizer(fake_vllm) -> None:
    component = _component()
    model = _FakeVllmModel()
    loaded = LoadedModel(handle=model, name="qwen3-instruct", backend="vllm", tokenizer=None)
    budget = component._resolve_budget("agent", thinking_tokens=0)
    component._generate(loaded, "hello", budget, system_prompt="SENTINEL")
    prompts, _ = model.generate_calls[0]
    assert "<|im_start|>system\nSENTINEL<|im_end|>" in prompts[0]


def test_vllm_failure_raises_generation_error(fake_vllm) -> None:
    component = _component()

    class ExplodingModel:
        llm_engine = object()

        def generate(self, prompts, params):
            raise RuntimeError("engine exploded")

    loaded = LoadedModel(handle=ExplodingModel(), name="m", backend="vllm")
    budget = component._resolve_budget("agent", thinking_tokens=0)
    with pytest.raises(Exception, match="vllm generation failed"):
        component._generate(loaded, "x", budget)


# --------------------------------------------------------------------- #
# ExLlamaV2
# --------------------------------------------------------------------- #


class _FakeSamplerSettings:
    def __init__(self):
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.token_repetition_penalty = None
        self.token_frequency_penalty = None


class _FakeExllamaGenerator:
    type_name = "ExLlamaV2DynamicGenerator"

    def __init__(self):
        self.generate_calls: list[dict] = []

    def generate(self, prompt, max_new_tokens, gen_settings, stop_conditions, **kwargs):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "gen_settings": gen_settings,
                "stop_conditions": stop_conditions,
                **kwargs,
            }
        )
        return "<think>reasoning</think>exllama answer"


@pytest.fixture
def fake_exllama(monkeypatch):
    sampler_module = types.ModuleType("exllamav2.generator")
    sampler_module.ExLlamaV2Sampler = types.SimpleNamespace(Settings=_FakeSamplerSettings)
    monkeypatch.setitem(sys.modules, "exllamav2", types.ModuleType("exllamav2"))
    monkeypatch.setitem(sys.modules, "exllamav2.generator", sampler_module)
    return sampler_module


def _exllama_loaded(generator, tokenizer=None) -> LoadedModel:
    # Make str(type(model)) advertise ExLlamaV2DynamicGenerator.
    generator.__class__.__name__ = "ExLlamaV2DynamicGenerator"
    generator.__class__.__qualname__ = "ExLlamaV2DynamicGenerator"
    return LoadedModel(handle=generator, name="qwen3-8b", backend="exllamav2", tokenizer=tokenizer)


def test_exllama_generation_applies_thinking_template(fake_exllama) -> None:
    component = _component()
    generator = _FakeExllamaGenerator()
    template_calls: list[dict] = []

    def apply_template(messages, **kwargs):
        template_calls.append({"messages": messages, **kwargs})
        return "<templated>"

    tokenizer = types.SimpleNamespace(chat_template="chatml", apply_chat_template=apply_template)
    loaded = _exllama_loaded(generator, tokenizer)

    budget = component._resolve_budget("agent", thinking_tokens=64)
    result = component._generate(loaded, "question", budget, system_prompt="sys")

    assert result == "exllama answer"
    call = generator.generate_calls[0]
    assert call["prompt"] == "<templated>"
    assert template_calls[0]["enable_thinking"] is True
    # Qwen3 models must not have BOS added.
    assert call["add_bos"] is False
    assert call["completion_only"] is True
    assert call["max_new_tokens"] == budget["answer_tokens"] + budget["thinking_tokens"]
    assert call["gen_settings"].temperature == 0.4


def test_exllama_disabled_thinking(fake_exllama) -> None:
    component = _component()
    generator = _FakeExllamaGenerator()
    template_calls: list[dict] = []

    tokenizer = types.SimpleNamespace(
        chat_template="chatml",
        apply_chat_template=lambda messages, **kw: (
            template_calls.append({"messages": messages, **kw}) or "<templated>"
        ),
    )
    loaded = _exllama_loaded(generator, tokenizer)

    budget = component._resolve_budget("agent", thinking_tokens=0)
    component._generate(loaded, "quick question", budget)
    assert template_calls[0]["enable_thinking"] is False


def test_exllama_chatml_fallback_without_tokenizer(fake_exllama) -> None:
    component = _component()
    generator = _FakeExllamaGenerator()
    loaded = _exllama_loaded(generator, tokenizer=None)
    budget = component._resolve_budget("agent", thinking_tokens=0)
    result = component._generate(loaded, "hello", budget, system_prompt="SENTINEL")
    call = generator.generate_calls[0]
    assert "<|im_start|>system\nSENTINEL<|im_end|>" in call["prompt"]
    assert result == "exllama answer"


def test_exllama_unterminated_thinking_yields_empty(fake_exllama) -> None:
    component = _component()
    generator = _FakeExllamaGenerator()

    def generate(prompt, max_new_tokens, gen_settings, stop_conditions, **kwargs):
        return "<think>i was still thinking when"

    generator.generate = generate
    loaded = _exllama_loaded(generator, tokenizer=None)
    budget = component._resolve_budget("agent", thinking_tokens=100)
    result = component._generate(loaded, "hard question", budget)
    assert result == ""


# --------------------------------------------------------------------- #
# MLX
# --------------------------------------------------------------------- #


@pytest.fixture
def fake_mlx(monkeypatch):
    mlx_module = types.ModuleType("mlx")
    mlx_lm_module = types.ModuleType("mlx_lm")
    generate_calls: list[dict] = []

    def fake_generate(model, tokenizer, prompt, max_tokens, verbose=False, sampler=None):
        generate_calls.append(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
        )
        return "<think>mlx thoughts</think>mlx answer"

    mlx_lm_module.generate = fake_generate
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_module)
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    class FakeMlxModel:
        pass

    FakeMlxModel.__module__ = "mlx_lm.model"
    return FakeMlxModel, generate_calls


def test_mlx_generation_uses_template_and_budget(fake_mlx) -> None:
    model_cls, generate_calls = fake_mlx
    component = _component()
    model = model_cls()
    tokenizer = types.SimpleNamespace(
        chat_template="chatml",
        apply_chat_template=lambda messages, **kw: "<mlx-formatted>",
    )
    loaded = LoadedModel(handle=model, name="qwen3-8b", backend="mlx", tokenizer=tokenizer)
    budget = component._resolve_budget("agent", thinking_tokens=16)
    result = component._generate(loaded, "hello", budget, system_prompt="sys")
    assert result == "mlx answer"
    call = generate_calls[0]
    assert call["prompt"] == "<mlx-formatted>"
    assert call["max_tokens"] <= 512
    assert call["sampler"] == {"temp": 0.4, "top_p": 0.8, "top_k": 20}


def test_mlx_failure_raises(fake_mlx) -> None:
    model_cls, _ = fake_mlx

    original = sys.modules["mlx_lm"].generate

    def exploding(model, tokenizer, prompt, max_tokens, verbose=False, sampler=None):
        raise RuntimeError("mlx core dumped")

    sys.modules["mlx_lm"].generate = exploding
    try:
        component = _component()
        loaded = LoadedModel(handle=model_cls(), name="m", backend="mlx", tokenizer=None)
        budget = component._resolve_budget("agent", thinking_tokens=0)
        with pytest.raises(Exception, match="mlx generation failed"):
            component._generate(loaded, "x", budget)
    finally:
        sys.modules["mlx_lm"].generate = original


# --------------------------------------------------------------------- #
# Loader backend paths via fake modules
# --------------------------------------------------------------------- #


def _loader(tmp_path):
    from signal_llm.loader import LocalLLMModelLoader

    return LocalLLMModelLoader(models_dir=tmp_path / "m", cache_dir=tmp_path / "c", device="cpu")


def test_loader_transformers_backend_honors_dtype_and_device(monkeypatch, tmp_path):
    loader = _loader(tmp_path)
    loader._has_cuda = lambda: False

    calls: list[dict] = []

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")

    def fake_auto_tokenizer(name, trust_remote_code=False):
        calls.append({"tokenizer": name, "trust": trust_remote_code})
        return object()

    def fake_auto_model(
        name, device_map=None, torch_dtype=None, trust_remote_code=False, quantization_config=None
    ):
        calls.append(
            {
                "model": name,
                "device_map": device_map,
                "dtype": torch_dtype,
                "trust": trust_remote_code,
                "quant": quantization_config,
            }
        )
        model = types.SimpleNamespace(
            device="cpu",
            to=lambda target: calls.append({"moved_to": str(target)}) or model,
        )
        return model

    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=fake_auto_tokenizer)
    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=fake_auto_model)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    config = ModelConfig(
        name="m", repo_id="org/m", backend="transformers", device="cpu", dtype="float32"
    )
    local = loader._get_model_path("org/m", None)
    (local / "config.json").write_text("{}")

    handle = loader._load_transformers(local, config)
    assert isinstance(handle, tuple)
    model_calls = [c for c in calls if "model" in c]
    assert model_calls[0]["dtype"] == "float32"
    assert model_calls[0]["device_map"] is None
    assert model_calls[0]["trust"] is False
    # CPU placement is applied explicitly.
    assert any(c.get("moved_to") == "cpu" for c in calls)


def test_loader_transformers_cuda_index_preserved(monkeypatch, tmp_path):
    loader = _loader(tmp_path)
    loader._has_cuda = lambda: True
    loader._cuda_index_available = lambda index: index == 1

    calls: list[dict] = []

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name, trust_remote_code=False: object()
    )

    def fake_auto_model(
        name, device_map=None, torch_dtype=None, trust_remote_code=False, quantization_config=None
    ):
        model = types.SimpleNamespace(device="cuda:1", to=lambda t: model)
        calls.append({"device_map": device_map})
        return model

    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=fake_auto_model)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    config = ModelConfig(
        name="m", repo_id="org/m", backend="transformers", device="cuda:1", dtype="float16"
    )
    local = loader._get_model_path("org/m", None)
    (local / "config.json").write_text("{}")

    loader._load_transformers(local, config)
    assert calls[0]["device_map"] == {"": "cuda:1"}


def test_loader_mlx_backend_loads_local(monkeypatch, tmp_path):
    loader = _loader(tmp_path)
    calls: list[dict] = []

    fake_mlx_lm = types.ModuleType("mlx_lm")

    def fake_load(path):
        calls.append({"path": path})
        return object(), object()

    fake_mlx_lm.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    config = ModelConfig(name="m", repo_id="org/m", backend="mlx", device="cpu")
    local = loader._get_model_path("org/m", None)
    (local / "config.json").write_text("{}")
    handle = loader._load_llm_mlx(local, config)
    assert isinstance(handle, tuple)
    assert calls[0]["path"] == str(local)


def test_loader_mlx_backend_without_library_raises(monkeypatch, tmp_path):
    loader = _loader(tmp_path)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    config = ModelConfig(name="m", repo_id="org/m", backend="mlx", device="cpu")
    local = loader._get_model_path("org/m", None)
    with pytest.raises(Exception, match="mlx_lm not available"):
        loader._load_llm_mlx(local, config)
