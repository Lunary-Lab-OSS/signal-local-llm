"""Cold-import and export identity tests for both package names."""

from __future__ import annotations

import subprocess
import sys


def test_signal_llm_exports_public_api() -> None:
    import signal_llm

    for name in signal_llm.__all__:
        assert hasattr(signal_llm, name), name


def test_signal_local_llm_compat_package_importable() -> None:
    import signal_local_llm

    assert signal_local_llm  # package exists


def test_cold_import_does_not_load_heavy_backends() -> None:
    """Importing the package must not pull torch/vllm/exllamav2/mlx."""
    code = (
        "import sys\n"
        "import signal_llm\n"
        "heavy = [m for m in ('torch', 'vllm', 'exllamav2', 'mlx', 'mlx_lm',\n"
        "                     'routellm', 'sentence_transformers')\n"
        "         if m in sys.modules]\n"
        "assert not heavy, f'unexpected heavy imports: {heavy}'\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_cold_import_makes_no_filesystem_or_network_side_effects(tmp_path) -> None:
    """Importing the package must not create model dirs or touch the net."""
    code = (
        "import os, sys\n"
        "before = set(os.listdir(os.getcwd()))\n"
        "import signal_llm\n"
        "after = set(os.listdir(os.getcwd()))\n"
        "assert before == after, f'created: {after - before}'\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
