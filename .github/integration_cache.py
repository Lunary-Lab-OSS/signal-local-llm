"""Reuse model files while requiring a fresh loader completion manifest."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def persistent_download_cache(tmp_path):
    cache = Path(os.environ["SIGNAL_LLM_TEST_CACHE"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for manifest in cache.glob("*/.signal-llm-manifest.json"):
        manifest.unlink()
    (tmp_path / "models").symlink_to(cache, target_is_directory=True)
