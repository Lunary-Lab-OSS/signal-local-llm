import sys


def test_package_import_does_not_import_torch():
    sys.modules.pop("torch", None)
    import signal_llm  # noqa: F401

    assert "torch" not in sys.modules


def test_signal_local_llm_alias_exports_component():
    from signal_local_llm import LLMComponent

    assert LLMComponent.__name__ == "LLMComponent"


def test_router_package_import_does_not_import_torch():
    sys.modules.pop("torch", None)
    from signal_llm.router import RouterStrategyFactory

    assert RouterStrategyFactory.__name__ == "RouterStrategyFactory"
    assert "torch" not in sys.modules
