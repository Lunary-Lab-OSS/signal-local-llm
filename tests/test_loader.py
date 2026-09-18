"""Tests for the LLM loader contract (L01-L05, L16)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from signal_llm.config import ModelConfig
from signal_llm.loader import (
    LocalLLMModelLoader,
    ModelLoadError,
    model_cache_key,
    parse_device,
)

# --------------------------------------------------------------------- #
# Device parsing (L02)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ("auto", None)),
        ("", ("auto", None)),
        ("auto", ("auto", None)),
        ("AUTO", ("auto", None)),
        ("cpu", ("cpu", None)),
        ("cuda", ("cuda", None)),
        ("cuda:0", ("cuda", 0)),
        ("cuda:3", ("cuda", 3)),
        ("mps", ("mps", None)),
    ],
)
def test_parse_device_valid(raw, expected) -> None:
    assert parse_device(raw) == expected


@pytest.mark.parametrize("bad", ["tpu", "cuda:x", "cuda:", "gpu", "cuda:-1"])
def test_parse_device_invalid(bad) -> None:
    with pytest.raises(ValueError):
        parse_device(bad)


# --------------------------------------------------------------------- #
# Cache-key collision resistance (L04)
# --------------------------------------------------------------------- #


def test_cache_keys_never_collide_for_underscore_identities() -> None:
    assert model_cache_key("a_b/c", None) != model_cache_key("a/b_c", None)


def test_cache_keys_separate_revisions_of_same_repo() -> None:
    assert model_cache_key("org/m", "v1") != model_cache_key("org/m", "v2")
    # None and "main" are intentionally the same identity (main is the
    # default revision), matching the previous single-directory behaviour.
    assert model_cache_key("org/m", None) == model_cache_key("org/m", "main")
    assert model_cache_key("org/m", None) != model_cache_key("org/m", "v2")


def test_cache_key_contains_no_raw_revision_component() -> None:
    key = model_cache_key("org/m", "x/../../outside")
    assert ".." not in key
    assert "/" not in key


def test_cache_directory_stays_inside_models_root(tmp_path: Path) -> None:
    loader = LocalLLMModelLoader(
        models_dir=tmp_path / "models", cache_dir=tmp_path / "cache", device="cpu"
    )
    path = loader._get_model_path("org/m", "x/../../escape")
    assert tmp_path.resolve() in path.resolve().parents
    assert path.parent == (tmp_path / "models").resolve()


# --------------------------------------------------------------------- #
# Fixture: recording loader
# --------------------------------------------------------------------- #


class RecordingLoader(LocalLLMModelLoader):
    """Loader with stubbed downloads/backends recording every call."""

    def __init__(self, tmp_path: Path, *, platform="linux", machine="x86_64", device="cpu"):
        super().__init__(
            models_dir=tmp_path / "models", cache_dir=tmp_path / "cache", device=device
        )
        self.platform = platform
        self.machine = machine
        self.download_calls: list[tuple[str, str | None]] = []
        self.backend_loads: list[str] = []

    def _download_model(self, repo_id, local_path, revision=None):
        self.download_calls.append((repo_id, revision))
        (local_path / "config.json").write_text("{}", encoding="utf-8")

    def _load_llm_mlx(self, model_path, config):
        self.backend_loads.append("mlx")
        return object(), object()

    def _load_exllamav2(self, model_path, config, draft_model_path=None, speculative=None):
        self.backend_loads.append("exllamav2")
        return object(), None

    def _load_vllm(self, model_path, config, draft_model_path=None, speculative=None):
        self.backend_loads.append("vllm")
        return object()

    def _load_transformers(self, model_path, config):
        self.backend_loads.append("transformers")
        return object(), object()

    def _has_cuda(self):
        return getattr(self, "_fake_cuda", False)

    def _cuda_index_available(self, index):
        return getattr(self, "_fake_cuda", False) and index in (0, 1)


# --------------------------------------------------------------------- #
# Backend dispatch (L01)
# --------------------------------------------------------------------- #


def test_linux_transformers_backend_is_dispatched(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    config = ModelConfig(name="m", repo_id="org/m", backend="transformers", device="cpu")
    handle, name = loader.load_llm_model([config])
    assert name == "m"
    assert loader.backend_loads == ["transformers"]
    assert handle is not None


def test_linux_vllm_backend_is_dispatched(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    loader._fake_cuda = True
    config = ModelConfig(name="m", repo_id="org/m", backend="vllm", device="cuda")
    loader.load_llm_model([config])
    assert loader.backend_loads == ["vllm"]


def test_darwin_arm64_does_not_override_explicit_backend(tmp_path) -> None:
    loader = RecordingLoader(tmp_path, platform="darwin", machine="arm64")
    config = ModelConfig(name="m", repo_id="org/m", backend="transformers", device="cpu")
    loader.load_llm_model([config])
    assert loader.backend_loads == ["transformers"]


def test_darwin_arm64_auto_backend_selects_mlx(tmp_path) -> None:
    loader = RecordingLoader(tmp_path, platform="darwin", machine="arm64")
    config = ModelConfig(name="m", repo_id="org/m", backend="auto", device="cpu")
    loader.load_llm_model([config])
    assert loader.backend_loads == ["mlx"]


def test_unknown_backend_rejected_before_download(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    config = ModelConfig(name="m", repo_id="org/m", backend="auto", device="cpu")
    object.__setattr__(config, "backend", "bogus")
    with pytest.raises(ValueError, match="unknown backend"):
        loader.load_llm_model([config])
    assert loader.download_calls == []


@pytest.mark.parametrize("backend", ["exllamav2", "vllm"])
def test_cuda_backends_rejected_without_cuda_before_download(tmp_path, backend) -> None:
    loader = RecordingLoader(tmp_path)  # no CUDA
    config = ModelConfig(name="m", repo_id="org/m", backend=backend, device="cuda")
    with pytest.raises(ModelLoadError, match="requires CUDA"):
        loader.load_llm_model([config])
    assert loader.download_calls == [], "must not download for unsupported hosts (L01)"


def test_mlx_rejected_off_apple_silicon_before_download(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    config = ModelConfig(name="m", repo_id="org/m", backend="mlx", device="cpu")
    with pytest.raises(ModelLoadError, match="Apple Silicon"):
        loader.load_llm_model([config])
    assert loader.download_calls == []


def test_cuda_index_validated_before_download(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    loader._fake_cuda = True  # devices 0 and 1 only
    config = ModelConfig(name="m", repo_id="org/m", backend="transformers", device="cuda:7")
    with pytest.raises(ModelLoadError, match="index 7"):
        loader.load_llm_model([config])
    assert loader.download_calls == []


def test_failure_falls_back_to_next_candidate(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    bad = ModelConfig(name="bad", repo_id="org/bad", backend="vllm", device="cuda")
    good = ModelConfig(name="good", repo_id="org/good", backend="transformers", device="cpu")
    _handle, name = loader.load_llm_model([bad, good])
    assert name == "good"
    assert loader.backend_loads == ["transformers"]


def test_all_candidates_failing_reports_every_attempt(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    bad1 = ModelConfig(name="b1", repo_id="org/b1", backend="vllm", device="cuda")
    bad2 = ModelConfig(name="b2", repo_id="org/b2", backend="mlx", device="cpu")
    with pytest.raises(ModelLoadError) as excinfo:
        loader.load_llm_model([bad1, bad2])
    message = str(excinfo.value)
    assert "b1" in message and "b2" in message


def test_empty_priority_list_is_an_error(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    with pytest.raises(ModelLoadError, match="empty"):
        loader.load_llm_model([])


# --------------------------------------------------------------------- #
# Download completion manifest (L05)
# --------------------------------------------------------------------- #


class _FakeHub:
    """Fake huggingface_hub.snapshot_download writing config.json."""

    def __init__(self):
        self.calls: list[dict] = []

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        (Path(kwargs["local_dir"]) / "config.json").write_text("{}", encoding="utf-8")
        return "resolved-commit-sha"


def _install_fake_hub(monkeypatch, hub: _FakeHub) -> None:
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = hub.snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_manifest_roundtrip_and_resume(monkeypatch, tmp_path) -> None:
    hub = _FakeHub()
    _install_fake_hub(monkeypatch, hub)
    loader = LocalLLMModelLoader(
        models_dir=tmp_path / "models", cache_dir=tmp_path / "cache", device="cpu"
    )

    local = loader._get_model_path("org/m", None)
    loader._ensure_downloaded("org/m", local, None)
    assert len(hub.calls) == 1
    manifest = local / ".signal-llm-manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["complete"] is True
    assert data["repo_id"] == "org/m"

    # Complete manifest: no re-download.
    loader._ensure_downloaded("org/m", local, None)
    assert len(hub.calls) == 1


def test_interrupted_download_resumes(monkeypatch, tmp_path) -> None:
    class _FlakyHub(_FakeHub):
        def __init__(self):
            super().__init__()
            self.fail_first = True

        def snapshot_download(self, **kwargs):
            self.calls.append(kwargs)
            (Path(kwargs["local_dir"]) / "config.json").write_text("{}", encoding="utf-8")
            if self.fail_first:
                self.fail_first = False
                raise OSError("connection reset mid-download")
            return "sha"

    hub = _FlakyHub()
    _install_fake_hub(monkeypatch, hub)
    loader = LocalLLMModelLoader(
        models_dir=tmp_path / "models", cache_dir=tmp_path / "cache", device="cpu"
    )
    local = loader._get_model_path("org/m", None)
    with pytest.raises(OSError, match="connection reset"):
        loader._ensure_downloaded("org/m", local, None)

    # config.json exists but no manifest: partial snapshot must resume.
    assert (local / "config.json").exists()
    assert not loader._is_complete(local)

    loader._ensure_downloaded("org/m", local, None)
    assert loader._is_complete(local)
    assert len(hub.calls) == 2


def test_download_validates_config_json_presence(monkeypatch, tmp_path) -> None:
    class _EmptyHub(_FakeHub):
        def snapshot_download(self, **kwargs):
            self.calls.append(kwargs)
            return "sha"  # writes nothing

    hub = _EmptyHub()
    _install_fake_hub(monkeypatch, hub)
    loader = LocalLLMModelLoader(
        models_dir=tmp_path / "models", cache_dir=tmp_path / "cache", device="cpu"
    )
    local = loader._get_model_path("org/m", None)
    with pytest.raises(ModelLoadError, match=r"config\.json"):
        loader._download_model("org/m", local, None)


# --------------------------------------------------------------------- #
# Local path bypass (L16)
# --------------------------------------------------------------------- #


def test_explicit_local_path_skips_hub(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    local = tmp_path / "local-model"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")

    config = ModelConfig(
        name="m", repo_id="org/m", backend="transformers", device="cpu", local_path=str(local)
    )
    loader.load_llm_model([config])
    assert loader.download_calls == [], "explicit local_path must not hit the hub (L16)"
    assert loader.backend_loads == ["transformers"]


def test_explicit_local_path_missing_config_raises(tmp_path) -> None:
    loader = RecordingLoader(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    config = ModelConfig(
        name="m", repo_id="org/m", backend="transformers", device="cpu", local_path=str(empty)
    )
    with pytest.raises(ModelLoadError, match="local_path"):
        loader.load_llm_model([config])


@pytest.mark.parametrize("backend", ["exllamav2", "vllm"])
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0", "cuda:1"])
def test_cuda_only_backends_never_silently_ignore_device(tmp_path, backend, device):
    loader = RecordingLoader(tmp_path)
    loader._fake_cuda = True
    config = ModelConfig(name="m", repo_id="org/m", backend=backend, device=device)
    with pytest.raises(ModelLoadError, match=r"requires CUDA|explicit CUDA index"):
        loader.load_llm_model([config])
    assert loader.download_calls == []


def test_backend_value_error_falls_back_with_immutable_handle_metadata(tmp_path):
    class ValueErrorLoader(RecordingLoader):
        def _load_transformers(self, model_path, config):
            if config.name == "bad":
                raise ValueError("architecture unsupported by this backend")
            return object(), object()

    loader = ValueErrorLoader(tmp_path)
    bad = ModelConfig(name="bad", repo_id="org/bad", backend="transformers")
    good = ModelConfig(name="good", repo_id="org/good", backend="auto")
    result = loader.load_llm_model([bad, good])
    handle, name = result
    assert name == "good"
    assert type(handle[0]) is object
    assert result.metadata.config == good
    assert result.metadata.config is not good
    assert result.metadata.backend == "transformers"
    assert result.metadata.model_path == loader._get_model_path("org/good").resolve()
    assert result.metadata.draft_model_path is None


def test_default_model_device_does_not_override_loader_cpu_request(tmp_path):
    loader = RecordingLoader(tmp_path, device="cpu")
    loader._fake_cuda = True
    config = ModelConfig(name="m", repo_id="org/m", backend="vllm")
    with pytest.raises(ModelLoadError, match="requires CUDA"):
        loader.load_llm_model([config])
    assert loader.download_calls == []


def test_all_caller_config_prevalidated_before_any_download(tmp_path):
    loader = RecordingLoader(tmp_path)
    good = ModelConfig(name="good", repo_id="org/good", backend="transformers")
    bad = ModelConfig(name="bad", repo_id="org/bad", backend="transformers")
    bad.device = "cudafoo"
    with pytest.raises(ValueError, match="device"):
        loader.load_llm_model([good, bad])
    assert loader.download_calls == []


@pytest.mark.parametrize("manifest", [[], None, True, "text", 42, {"version": 2}])
def test_malformed_manifest_is_incomplete(tmp_path, manifest):
    loader = RecordingLoader(tmp_path)
    local = loader._get_model_path("org/m")
    loader._manifest_path(local).write_text(json.dumps(manifest))
    assert not loader._is_complete(local, "org/m", None)


def test_manifest_checks_identity_and_every_downloaded_file(monkeypatch, tmp_path):
    loader = RecordingLoader(tmp_path)
    local = loader._get_model_path("org/m")
    (local / "config.json").write_text("{}")
    weights = local / "model.safetensors"
    weights.write_bytes(b"weights")
    loader._write_manifest(local, "org/m", "rev", None)
    assert loader._is_complete(local, "org/m", "rev")
    assert not loader._is_complete(local, "org/other", "rev")
    assert not loader._is_complete(local, "org/m", "other")
    weights.write_bytes(b"truncated")
    assert not loader._is_complete(local, "org/m", "rev")
    weights.unlink()
    assert not loader._is_complete(local, "org/m", "rev")


@pytest.mark.parametrize("name,size", [("../outside", 1), ("/absolute", 1), ("config.json", True)])
def test_manifest_rejects_unsafe_file_entries(tmp_path, name, size):
    loader = RecordingLoader(tmp_path)
    local = loader._get_model_path("org/m")
    (local / "config.json").write_text("{}")
    loader._write_manifest(local, "org/m", None, None)
    manifest = loader._manifest_path(local)
    data = json.loads(manifest.read_text())
    data["files"][name] = size
    manifest.write_text(json.dumps(data))
    assert not loader._is_complete(local, "org/m", None)


def test_failed_manifest_replace_preserves_prior_complete_manifest(monkeypatch, tmp_path):
    loader = RecordingLoader(tmp_path)
    local = loader._get_model_path("org/m")
    (local / "config.json").write_text("{}")
    loader._write_manifest(local, "org/m", None, None)
    before = loader._manifest_path(local).read_bytes()

    def fail_replace(self, target):
        assert json.loads(self.read_text())["complete"] is True
        assert target.read_bytes() == before
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        loader._write_manifest(local, "org/other", None, None)
    assert loader._manifest_path(local).read_bytes() == before
    assert list(local.glob(".signal-llm-manifest.json.*")) == []
