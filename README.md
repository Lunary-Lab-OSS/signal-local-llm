# signal-local-llm

Local LLM inference strategies for the Signal Switchboard voice-agent stack.

Extracted from `signal-switchboard` following the same pattern as
[`signal-asr-strategies`](https://github.com/Lunary-Lab-OSS/signal-asr-strategies).

## Contents

- `signal_llm.config` — validated `ModelConfig`, `SpeculativeDecodingConfig`, `LLMConfig`
- `signal_llm.loader` — `LocalLLMModelLoader` / `LLMLoader`: downloads and loads LLM models (ExLlamaV2, vLLM, Transformers, MLX)
- `signal_llm.component` — `LLMComponent`: router/agent/semantic inference with role-scoped loading and a documented token-budget contract
- `signal_llm.intent_router` — `IntentRouter`: cascading router (reflex → estimator → thinker) with strict label parsing
- `signal_llm.router` — SOTA single-tower routing; local Matrix Factorization is disabled

## Install

```bash
uv pip install -e .                    # core (huggingface-hub only)
uv sync --locked --no-dev --extra cpu --extra transformers # locked CPU inference
uv sync --locked --no-dev --extra cuda # separate GPU resolution; hardware unverified
uv pip install -e ".[mlx]"            # Apple Silicon MLX
uv pip install -e ".[routellm]"       # neural routing (sentence-transformers, textstat)
# Do not combine cpu with cuda or exllamav2.
```

Use uv 0.10.12. All profiles now use official PyPI Torch 2.13.0 artifacts;
`cpu` means CPU execution, **not CPU-only wheels**. Select `device="cpu"`.
On Linux the PyPI wheels also pull large NVIDIA runtime dependencies, even for
CPU execution. No driver or GPU is required to select CPU execution.
CUDA/ExLlamaV2 still conflict with `cpu` as a profile guardrail.
Do not use `--all-extras`. Platform-compatible GPU wheels, drivers and kernel
builds still need hardware validation; replacing wheels manually leaves the
locked profile. No binary wheels are committed here.

## Usage

SOTA routing defaults to unquantized inference, including on CUDA. Opt in with
`LLMConfig.routellm_use_int4=True` only with CUDA and a compatible `torchao`
installation; the `routellm` extra does not install this optional quantizer.

```python
from signal_llm import LLMComponent, LLMConfig, ModelConfig

config = LLMConfig(
    router_priority=[
        ModelConfig(name="qwen3-0.6b", repo_id="Qwen/Qwen3-0.6B", backend="transformers")
    ],
    semantic_priority=[
        ModelConfig(name="qwen3-8b", repo_id="Qwen/Qwen3-8B", backend="transformers")
    ],
    agent_priority=[ModelConfig(name="qwen3-8b", repo_id="Qwen/Qwen3-8B", backend="transformers")],
)
loader = None  # or LocalLLMModelLoader(models_dir, cache_dir, device="cuda")
component = LLMComponent(config=config, model_loader=loader, device="cuda", platform="linux")
component.load_all_models()
response = component.generate_agent("What is the capital of France?")
```

### Contracts (tested)

- **Backend dispatch:** the configured backend wins; platform/device support
  is validated *before* any download. Linux CPU transformers has a real-model
  integration track; vLLM is currently unsupported (see dependency migration). MLX is
  Apple-Silicon-only; unknown backends raise.
- **Devices:** `cuda:1` keeps its index everywhere; unsupported names raise
  instead of silently mapping to CPU.
- **Remote code:** `trust_remote_code` defaults to **off**. Opting in
  requires a pinned revision (`ModelConfig(trust_remote_code=True,
  revision=...)`); checkpoints advertising a different embedding model than
  the configured trusted identity are rejected.
- **Model cache:** directories are keyed by a collision-resistant hash of
  `repo@revision`; interrupted downloads resume (a completion manifest is
  only written after a successful snapshot).
- **Role loading:** semantic/agent/router load independently behind their
  enablement gates; sharing happens only between actually-loaded handles
  with identical repo+revision+backend.
- **Token budgets:** `max_tokens` caps total generated tokens, not input plus
  output context. Thinking and answer allowances are not independently enforced
  by every backend; they share the total generation cap. `temperature`/`top_p`
  from config reach every backend.
- **Reasoning:** complete `<think>…</think>` blocks are always stripped; an
  unterminated opener means "still thinking" and yields an empty answer —
  partial chain-of-thought never reaches voice output.
- **Routing:** classifier labels are parsed strictly (one exact label;
  negations/echoes/multi-label are errors); anchored code commands skip the
  estimator; budgets apply to agentic routes only; non-finite scores raise.
- **Logging:** prompt content never appears in logs.

## Development

```bash
uv sync --locked --group dev --extra cpu
uv run --locked --group dev --extra cpu pytest tests/ -q -m "not integration" --cov=signal_llm --cov-fail-under=80
uv run --locked --group dev --extra cpu mypy signal_llm
uvx ruff@0.15.14 check . && uvx ruff@0.15.14 format --check .
```

The unit suite runs on CPU torch with no model downloads; heavy backends
are exercised through injectable seams with strict fakes. `signal_engine`
compatibility: the Switchboard shim re-exports this package unchanged.

### Breaking router checkpoint migration

SOTA inference now requires a trained checkpoint; missing checkpoints fail
instead of publishing a random initialized router. The serialized dictionary
must contain exactly `schema_version` (integer `1`), `kind` (`head` or `full`),
and `state_dict`. A head artifact needs all `head.*` tensors and
`model_features`; a full artifact needs the complete model state. Missing or
unexpected keys, non-tensors, non-finite values, incompatible shapes and dtypes
are rejected. Bare legacy state dictionaries no longer load: migrate from the
trusted training artifact with the matching architecture, retaining trained
weights and explicitly identifying head versus full state. Renaming keys or
inventing missing tensors is not a valid migration.

SOTA's SentenceTransformer backbone must already be cached locally; loading
uses `local_files_only=True` and `trust_remote_code=False`. The router does
not download a missing backbone. Arrange trusted backbone provisioning
separately from the checkpoint. Direct `SingleTowerStudent` construction is
the training path, not permission to serve an untrained inference router.

Matrix Factorization fails explicitly: RouteLLM 0.2.0 does not accept the
local `embedding_model` interface and its scorer calls an external OpenAI
embedding API. Substituting local embeddings is incompatible with its trained
projection, even if dimensions match. Use SOTA with a trained local artifact;
there is no silent fallback or provider call in the disabled strategy.

### Verification scope

Linux CPU Transformers CI runs the existing Qwen/Qwen2.5-0.5B-Instruct test
under a 25-minute job limit, caches downloaded model files, and rejects skips.
The model revision is currently unpinned, so this is real inference evidence,
not bit-reproducible model validation. No new model API is introduced.
CUDA, native Windows, macOS MLX/MPS, ExLlamaV2, and vLLM hardware execution
remain unverified by that test; mocked unit coverage is not hardware evidence.

Dependency gates audit exported locked runtime profiles (core, CPU Transformers,
CPU routing, standalone CPU/Transformers/routing, CUDA, ExLlamaV2, MLX), not the pip-audit tool environment.
Platform markers are evaluated on the audit runner: a Linux scan does not
validate macOS/Windows-only dependencies. Unknown packages/advisory failures
remain blocking. Lock resolution is not a vulnerability-free or hardware-support
claim, and model weights are outside the Python dependency audit.

### Dependency migration (2026-09-17)

- `setuptools>=83.0.0` is required (lock: 84.0.0).
  [PYSEC-2026-3447](https://osv.dev/vulnerability/PYSEC-2026-3447) affects
  versions below 83.0.0 (Unicode MANIFEST exclusion bypass), fixed in 83.0.0.
  vLLM 0.29.0 requires `setuptools>=77.0.3,<81.0.0` on Python >3.11.
  Therefore the `vllm` extra is removed and `cuda` no longer installs it;
  retained loader code is unsupported, not an audited integration. Migrate
  to `transformers` and set `backend="transformers"` with a compatible model;
  do not assume vLLM scheduling, throughput or quantization compatibility.
- The custom CPU index was removed, not bypassed in auditing. `2.13.0+cpu`
  is absent from PyPI's audit endpoint. No source-equivalence proof was
  established, so no suffix stripping or advisory suppression is performed.
  The PyPI 2.13.0 artifact is audited under its actual published version.
- Routing and dev pin `textstat==0.7.8`, whose declared dependency is `cmudict`,
  not NLTK. The previously resolved 0.7.13 introduces NLTK 3.10.3;
  [GHSA-8mgp-746c-j5xp](https://osv.dev/vulnerability/GHSA-8mgp-746c-j5xp)
  affects NLTK through 3.10.3 (latest checked), with no fixed version recorded.
  This is a dependency-path removal, not a claim that NLTK is fixed.
  The used `flesch_kincaid_grade` and `polysyllabcount` API remains available;
  syllable/readability results can differ between releases. Revalidate router
  thresholds/checkpoints against representative prompts before deployment.

Restoring vLLM requires compatible upstream constraints, clean audits and
backend validation. Audits query advisory metadata; they do not download model
weights or prove hardware support. PyPI Torch cold installs can require several
GB of downloads and disk space; the former CPU-only download budget no longer
applies. Model inference tests add the separately cached Qwen model download.

Local verification of the final migration: all nine CI runtime profile exports
passed `uv lock --check` and the exact strict pip-audit 2.9.0 CI command on Linux.
Python 3.13 unit suite: 322 passed, 1 deselected; coverage 88.03%. Mypy passed.
An attempted textstat 0.7.5 downgrade failed three tests because it imported
removed `pkg_resources`; 0.7.8 passes those real feature-extraction tests with
setuptools 84.0.0. Python 3.12, real-model inference and non-Linux/GPU hardware
were not rerun for this migration. Passing audits and unit tests are not release
approval, nor evidence that removed NeMo/vLLM integrations are safe.

## License

GPL-3.0-or-later. Model weights keep their own upstream licenses.
