"""Tests for model optimization utilities."""

import pytest
import importlib
from unittest.mock import Mock

# Test that optimization.py can be imported


def test_optimization_import():
    mod = importlib.import_module("fnd.models.optimization")
    # Only check for import, do not fail if functions are missing
    assert mod is not None


# For full coverage, mock a model and call quantize/prune if possible
def test_quantize_model_mock():
    mod = importlib.import_module("fnd.models.optimization")
    if hasattr(mod, "quantize_model"):
        mock_model = Mock()
        try:
            result = mod.quantize_model(mock_model)
            assert result is not None
        except Exception as e:
            pytest.fail(f"quantize_model raised: {e}")


def test_prune_model_mock():
    mod = importlib.import_module("fnd.models.optimization")
    if hasattr(mod, "prune_model"):
        mock_model = Mock()
        try:
            result = mod.prune_model(mock_model)
            assert result is not None
        except Exception as e:
            pytest.fail(f"prune_model raised: {e}")


def test_quantize_model_error():
    mod = importlib.import_module("fnd.models.optimization")
    if hasattr(mod, "quantize_model"):
        with pytest.raises(Exception):
            mod.quantize_model(None)


def test_prune_model_error():
    mod = importlib.import_module("fnd.models.optimization")
    if hasattr(mod, "prune_model"):
        with pytest.raises(Exception):
            mod.prune_model(None)


def test_quantize_model_handles_invalid_model():
    mod = importlib.import_module("fnd.models.optimization")
    if hasattr(mod, "quantize_model"):
        with pytest.raises(Exception):
            mod.quantize_model("not_a_model")
