# signal-local-llm

Local LLM inference strategies for the Signal Switchboard voice-agent stack.

Extracted from `signal-switchboard` following the same pattern as
[`signal-asr-strategies`](https://github.com/Lunary-Lab-OSS/signal-asr-strategies).

## Contents

- `signal_llm.config` — `ModelConfig`, `SpeculativeDecodingConfig`, `LLMConfig` dataclasses
- `signal_llm.loader` — `LocalLLMModelLoader` / `LLMLoader`: downloads and loads LLM models (ExLlamaV2, vLLM, Transformers, MLX)
- `signal_llm.component` — `LLMComponent`: router/agent/semantic inference with platform-specific settings
- `signal_llm.intent_router` — `IntentRouter`: cascading router (reflex → estimator → thinker)
- `signal_llm.router` — RouteLLM strategy pattern (Matrix Factorization, SOTA)

## Install

```bash
uv pip install -e ../signal-local-llm
# With ExLlamaV2 (Windows CUDA):
uv pip install -e "../signal-local-llm[exllamav2]"
# With vLLM:
uv pip install -e "../signal-local-llm[vllm]"
# With RouteLLM routing:
uv pip install -e "../signal-local-llm[routellm]"
```

> **Note on CUDA wheels:** the `exllamav2` PyPI package is CPU-only. For the
> CUDA 12.1 build used on Windows/RTX, fetch the official wheel from the
> [exllamav2 releases page](https://github.com/turboderp/exllamav2/releases) or
> from this repository's CI artifacts — binary wheels are never committed.

## Usage

```python
from signal_llm import LocalLLMModelLoader, LLMComponent, LLMConfig, ModelConfig

config = LLMConfig(
    router_priority=[ModelConfig(name="qwen3-0.6b", repo_id="Qwen/Qwen3-0.6B")],
    semantic_priority=[ModelConfig(name="qwen3-8b", repo_id="Qwen/Qwen3-8B")],
    agent_priority=[ModelConfig(name="qwen3-8b", repo_id="Qwen/Qwen3-8B")],
)
loader = LocalLLMModelLoader(models_dir="./models", cache_dir="./cache", device="cuda")
component = LLMComponent(config=config, model_loader=loader, device="cuda", platform="windows")
component.load_all_models()
response = component.generate_agent("What is the capital of France?")
```

Switchboard imports `signal_engine.strategies.llm.LLMComponent`; that module is a
compatibility shim that re-exports this package.
