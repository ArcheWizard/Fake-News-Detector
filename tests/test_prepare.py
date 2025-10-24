"""Tests for data preprocessing module."""

import pytest
import importlib
from unittest.mock import Mock

# Test that prepare.py can be imported and main functions exist


def test_prepare_import():
    mod = importlib.import_module("fnd.data.prepare")
    assert hasattr(mod, "main") or hasattr(mod, "preprocess_data")


# For full coverage, mock input data and test preprocess_data if possible
def test_preprocess_data_mock():
    mod = importlib.import_module("fnd.data.prepare")
    if hasattr(mod, "preprocess_data"):
        mock_df = Mock()
        try:
            result = mod.preprocess_data(mock_df)
            assert result is not None
        except Exception as e:
            pytest.fail(f"preprocess_data raised: {e}")


def test_preprocess_data_error():
    mod = importlib.import_module("fnd.data.prepare")
    if hasattr(mod, "preprocess_data"):
        with pytest.raises(Exception):
            mod.preprocess_data(None)
