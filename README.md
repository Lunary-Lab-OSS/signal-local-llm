# signal-local-llm

[![ci](https://github.com/Lunary-Lab-OSS/signal-local-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Lunary-Lab-OSS/signal-local-llm/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Local LLM inference for voice agents. It loads open-weight models from Hugging Face onto the best backend for the host and serves three roles (router, semantic, and agent). A cascading intent router sends each request to the cheapest model that can answer it.

Part of the [Signal Switchboard](https://github.com/Lunary-Lab-OSS) voice-agent stack, alongside [`signal-asr-strategies`](https://github.com/Lunary-Lab-OSS/signal-asr-strategies).

## Features

- **Multiple backends, one API.** Hugging Face Transformers (CPU and CUDA), ExLlamaV2 (CUDA), and MLX (Apple Silicon), chosen per model or automatically.
- **Role-based models.** Separate router, semantic, and agent roles, each with an ordered fallback list. Roles that point to the same model share one loaded instance.
- **Cascading intent routing.** Instant reflex commands, then a difficulty estimator, then a thinking-token budget. Only the hard requests pay for reasoning.
- **Voice-safe output.** `<think>…</think>` reasoning is always stripped, so partial chain-of-thought never reaches text-to-speech.
- **Reproducible and safe by default.** Revision-pinned, content-addressed model cache with resumable downloads; `trust_remote_code` is off unless you pin a revision; prompts never appear in logs.
- **Speculative decoding** with a small draft model on ExLlamaV2.

## Backends

| Backend | `backend=` | Platforms | Devices |
|---|---|---|---|
| [Transformers](https://github.com/huggingface/transformers) | `transformers` | Linux, Windows, macOS | `cpu`, `cuda`, `cuda:<n>`, `mps` |
| [ExLlamaV2](https://github.com/turboderp-org/exllamav2) | `exllamav2` | Linux, Windows | `cuda` (choose the GPU with `CUDA_VISIBLE_DEVICES`) |
| [MLX](https://github.com/ml-explore/mlx-lm) | `mlx` | macOS (Apple Silicon) | Apple GPU |
| Automatic | `auto` | Any | MLX on Apple Silicon, ExLlamaV2 on Windows, Transformers elsewhere |

The backend and device are checked against the host before anything downloads, so a mismatch fails immediately with a clear error.

## Installation

Requires Python 3.12+. Install from GitHub with the extra for your backend:

```bash
pip install "signal-local-llm[transformers] @ git+https://github.com/Lunary-Lab-OSS/signal-local-llm.git"
```

| Extra | Adds |
|---|---|
| `transformers` | PyTorch, Transformers, Accelerate, and bitsandbytes |
| `cuda` | Everything in `transformers`, plus ExLlamaV2 |
| `exllamav2` | ExLlamaV2 only |
| `mlx` | `mlx-lm` for Apple Silicon |
| `routellm` | The neural difficulty router (sentence-transformers, textstat) |
| `cpu` | PyTorch only |

For a reproducible environment from source, use [uv](https://docs.astral.sh/uv/). Pick either the CPU profile or the GPU profile, not both:

```bash
git clone https://github.com/Lunary-Lab-OSS/signal-local-llm.git
cd signal-local-llm
uv sync --locked --extra cpu --extra transformers   # CPU
uv sync --locked --extra cuda                       # NVIDIA GPU
```

All profiles use the standard PyPI PyTorch wheels. On Linux these include the CUDA runtime libraries, so the first install downloads several GB.

## Quick start

```python
from pathlib import Path

from signal_llm import LLMComponent, LLMConfig, LocalLLMModelLoader, ModelConfig

cache = Path.home() / ".cache" / "signal-llm"
loader = LocalLLMModelLoader(models_dir=cache / "models", cache_dir=cache / "hub", device="cpu")

qwen = ModelConfig(
    name="qwen2.5-0.5b",
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    backend="transformers",
    device="cpu",
    dtype="float32",
)
config = LLMConfig(
    router_priority=[qwen],
    semantic_priority=[qwen],
    agent_priority=[qwen],
    max_tokens=64,
)

llm = LLMComponent(config=config, model_loader=loader, device="cpu")
llm.load_all_models()            # identical roles share one loaded model

print(llm.generate_agent("What is the capital of France? Answer in one word."))
llm.close()
```

Each `*_priority` list is tried in order, so you can list a GPU model first and a CPU fallback after it. For CUDA, set `device="cuda"` on the loader, the `ModelConfig`, and the `LLMComponent`.

### Generation

| Method | Use |
|---|---|
| `generate_router(prompt)` | Fast classification or short outputs from the router model |
| `generate_semantic(prompt, system_prompt=None, max_tokens=None)` | Low-latency answers, no reasoning |
| `generate_agent(prompt, thinking_tokens=None)` | Full answers with an optional reasoning budget |

`max_tokens` caps the total number of generated tokens, reasoning included. `temperature` and `top_p` from `LLMConfig` apply to every backend.

### Intent routing

```python
from signal_llm import IntentRouter

router = IntentRouter(router_type="llm", max_thinking_tokens=2500)
intent, difficulty, thinking_tokens = router.route(
    "scroll down", router_generate_func=llm.generate_router
)
# -> ("reflex", None, None)
```

`route()` returns one of three intents:

| Intent | Meaning | Typical handler |
|---|---|---|
| `reflex` | Exact-match UI command (`scroll down`, `close window`, …), resolved without a model | Execute directly |
| `semantic` | Code-navigation command or simple request | `generate_semantic` |
| `agentic` | Open-ended request; includes a difficulty score and thinking-token budget | `generate_agent(prompt, thinking_tokens=...)` |

Router labels are parsed strictly: an output that is not exactly one valid label raises an error rather than being guessed.

With `router_type="routellm"`, a local neural router (`routellm_router_name="sota"`, the default) scores difficulty, and the budget scales from `min_thinking_tokens` to `max_thinking_tokens`.

## Configuration reference

**`ModelConfig`**

| Field | Default | Notes |
|---|---|---|
| `name`, `repo_id` | required | Display name and Hugging Face repository |
| `revision` | `None` | Commit SHA to pin (recommended) |
| `backend` | `exllamav2` | `transformers`, `exllamav2`, `mlx`, or `auto` |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:<n>`, or `mps` |
| `dtype` | `float16` | `float16`, `bfloat16`, or `float32` |
| `quantized` | `False` | 8-bit bitsandbytes loading for Transformers on CUDA |
| `local_path` | `None` | Use a model directory on disk instead of downloading |
| `trust_remote_code` | `False` | Requires a pinned `revision` |

**`LLMConfig`** (main fields)

| Field | Default | Notes |
|---|---|---|
| `router_priority`, `semantic_priority`, `agent_priority` | required | Ordered model lists per role |
| `semantic_enabled`, `agent_enabled` | `True` | Skip loading a role entirely |
| `max_tokens` | `2048` | Total generation cap |
| `temperature`, `top_p` | `0.7`, `0.9` | Sampling parameters |
| `agent_thinking_tokens` | `512` | Default reasoning budget for the agent |
| `speculative_decoding` | `None` | `SpeculativeDecodingConfig(enabled=True, num_speculative_tokens=5)` |
| `router_type` | `llm` | `llm`, `routellm`, or `coreml` |

## Neural router checkpoints

The `routellm` router (SOTA single-tower model) loads a trained checkpoint from `routellm_checkpoint_path`. A checkpoint is a dictionary with:

- `schema_version`: `2`
- `kind`: `"head"` (head tensors and `model_features`) or `"full"` (complete model state)
- `state_dict`: the tensors
- `artifact_metadata`: `embedding_model_id`, `embedding_revision`, `embedding_dimension`, `normalize_embeddings`, `linguistic_feature_version` (`1`), `architecture` (`"single-tower-scalar-regression-v2"`), and `model_pair_values`

The metadata must match the deployed configuration and the checkpoint's own tensors, so a checkpoint trained against a different embedding backbone can't be loaded by mistake. The embedding backbone (`routellm_embedding_model`, pinned with `routellm_embedding_revision`) is loaded from the local Hugging Face cache.

## Architecture

```
signal_llm/
├── component.py       LLMComponent: role loading, handle sharing, generation
├── config.py          Validated ModelConfig / LLMConfig / SpeculativeDecodingConfig
├── loader.py          Download, cache, and backend dispatch
├── intent_router.py   Reflex → estimator → thinking-budget cascade
└── router/
    ├── factory.py     Router selection
    ├── sota.py        SOTA single-tower router and checkpoint validation
    ├── model.py       Router network (embedding backbone + regression head)
    ├── features.py    Linguistic features
    └── coreml_complexity.py   Core ML complexity classifier (macOS)
```

`signal_local_llm` is an import alias for `signal_llm`.

## Development

```bash
uv sync --locked --group dev --extra cpu
uv run --locked --group dev --extra cpu pytest tests/ -m "not integration" --cov=signal_llm
uv run --locked --group dev --extra cpu mypy signal_llm
uvx ruff@0.15.14 check . && uvx ruff@0.15.14 format --check .
```

The unit suite runs on CPU with no model downloads. The end-to-end test runs real inference with a pinned Qwen2.5-0.5B-Instruct snapshot:

```bash
SIGNAL_LLM_INTEGRATION=1 uv run --locked --group dev --extra cpu --extra transformers pytest tests/test_integration.py
```

CI runs linting, type checking, unit tests on Python 3.12 and 3.13 with a coverage gate, a package build, a dependency audit of every install profile, and the real-model test on each pull request.

## Contributing

Issues and pull requests are welcome. Please run the checks above before you open a PR. To report a security issue, follow the [security policy](https://github.com/Lunary-Lab-OSS/.github/blob/main/.github/SECURITY.md).

## License

[GPL-3.0-or-later](LICENSE). Model weights keep their own upstream licenses.
