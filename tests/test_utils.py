"""Tests for model utility functions."""

import pytest
import importlib
from unittest.mock import patch, Mock

# Test that utils.py can be imported and main functions exist


def test_utils_import():
    mod = importlib.import_module("fnd.models.utils")
    assert hasattr(mod, "get_model_info")
    assert hasattr(mod, "load_model_and_tokenizer_from_dir")


def test_get_model_info_mock():
    import json

    mod = importlib.import_module("fnd.models.utils")
    fake_json = json.dumps(
        {
            "model_type": "bert",
            "num_labels": 2,
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "id2label": {"0": "real", "1": "fake"},
        }
    )
    with (
        patch("fnd.models.utils.os.path.exists", return_value=True),
        patch("fnd.models.utils.open", create=True) as mock_open,
        patch("fnd.models.utils.os.path.isdir", return_value=True),
    ):
        mock_open.return_value.__enter__.return_value.read.return_value = fake_json
        mock_open.return_value.__enter__.return_value.__iter__.return_value = (
            fake_json.splitlines()
        )
        mock_open.return_value.__enter__.return_value.read = lambda: fake_json
        try:
            result = mod.get_model_info("/fake/dir")
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"get_model_info raised: {e}")


def test_load_model_and_tokenizer_from_dir_mock():
    mod = importlib.import_module("fnd.models.utils")
    with (
        patch("fnd.models.utils.os.path.exists", return_value=True),
        patch("fnd.models.utils.os.path.isdir", return_value=True),
        patch("fnd.models.utils.AutoModelForSequenceClassification") as mock_model,
        patch("fnd.models.utils.AutoTokenizer") as mock_tokenizer,
    ):
        mock_model.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        try:
            model, tokenizer = mod.load_model_and_tokenizer_from_dir("/fake/dir")
            assert model is not None
            assert tokenizer is not None
        except Exception as e:
            pytest.fail(f"load_model_and_tokenizer_from_dir raised: {e}")


def test_get_model_info_error():
    mod = importlib.import_module("fnd.models.utils")
    with patch("fnd.models.utils.os.path.exists", return_value=False):
        with pytest.raises(Exception):
            mod.get_model_info("/nonexistent/dir")
