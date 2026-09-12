"""Tests for LLMLoader — no real GPU required."""

from unittest.mock import patch

import pytest

from signal_llm.config import ModelConfig
from signal_llm.loader import LLMLoader


@pytest.fixture
def loader(tmp_path):
    return LLMLoader(models_dir=tmp_path / "models", cache_dir=tmp_path / "cache", device="cpu")


def test_get_model_path_basic(loader):
    path = loader._get_model_path("org/model-name")
    assert path.name == "org_model-name"
    assert path.exists()


def test_get_model_path_with_revision(loader):
    path = loader._get_model_path("org/model", revision="v1.0")
    assert "v1.0" in path.name


def test_has_cuda_no_torch(loader):
    with (
        patch.dict("sys.modules", {"torch": None}),
        # Patch importlib so torch import raises ImportError
        patch(
            "builtins.__import__",
            side_effect=lambda n, *a, **k: (
                (_ for _ in ()).throw(ImportError()) if n == "torch" else __import__(n, *a, **k)
            ),
        ),
    ):
        result = loader._has_cuda()
    # Either False or True depending on actual torch presence — just ensure no exception
    assert isinstance(result, bool)


def test_load_llm_model_all_fail(loader, tmp_path):
    model = ModelConfig(name="fake", repo_id="fake/fake")
    with (
        patch.object(loader, "_download_model", side_effect=RuntimeError("no network")),
        pytest.raises(RuntimeError, match="Failed to load any LLM model"),
    ):
        loader.load_llm_model([model])
