"""Tests for model utility functions."""

import pytest
import importlib
import json
from unittest.mock import patch, Mock

# Test that utils.py can be imported and main functions exist


def test_utils_import():
    mod = importlib.import_module("fnd.models.utils")
    assert hasattr(mod, "get_model_info")
    assert hasattr(mod, "load_model_and_tokenizer_from_dir")


@pytest.mark.parametrize(
    "exists,dirpath,raises",
    [
        (True, True, False),
        (False, True, True),
        (True, False, False),  # Only expect error if exists is False
    ],
)
def test_get_model_info_parametrized(exists, dirpath, raises):
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
        patch("fnd.models.utils.os.path.exists", return_value=exists),
        patch("fnd.models.utils.os.path.isdir", return_value=dirpath),
        patch("fnd.models.utils.open", create=True) as mock_open,
    ):
        mock_open.return_value.__enter__.return_value.read.return_value = fake_json
        mock_open.return_value.__enter__.return_value.__iter__.return_value = (
            fake_json.splitlines()
        )
        mock_open.return_value.__enter__.return_value.read = lambda: fake_json
        if raises:
            with pytest.raises(Exception):
                mod.get_model_info("/fake/dir")
        else:
            result = mod.get_model_info("/fake/dir")
            assert isinstance(result, dict)


@pytest.mark.parametrize(
    "exists,dirpath,raises",
    [
        (True, True, False),
        (False, True, True),
        (True, False, True),
    ],
)
def test_load_model_and_tokenizer_from_dir_parametrized(exists, dirpath, raises):
    mod = importlib.import_module("fnd.models.utils")
    with (
        patch("fnd.models.utils.os.path.exists", return_value=exists),
        patch("fnd.models.utils.os.path.isdir", return_value=dirpath),
        patch("fnd.models.utils.AutoModelForSequenceClassification") as mock_model,
        patch("fnd.models.utils.AutoTokenizer") as mock_tokenizer,
    ):
        mock_model.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        if raises:
            with pytest.raises(Exception):
                mod.load_model_and_tokenizer_from_dir("/fake/dir")
        else:
            model, tokenizer = mod.load_model_and_tokenizer_from_dir("/fake/dir")
            assert model is not None
            assert tokenizer is not None


def test_load_model_consistency_mock():
    with (
        patch("fnd.models.utils.os.path.exists", return_value=True),
        patch("fnd.models.utils.os.path.isdir", return_value=True),
        patch(
            "fnd.models.utils.AutoModelForSequenceClassification.from_pretrained",
            return_value=Mock(),
        ),
        patch("fnd.models.utils.AutoTokenizer.from_pretrained", return_value=Mock()),
    ):
        from fnd.models.utils import load_model_and_tokenizer_from_dir

        model, tokenizer = load_model_and_tokenizer_from_dir("/fake/dir")
        assert model is not None
        assert tokenizer is not None


def test_get_model_info_error():
    mod = importlib.import_module("fnd.models.utils")
    with patch("fnd.models.utils.os.path.exists", return_value=False):
        with pytest.raises(Exception):
            mod.get_model_info("/nonexistent/dir")


def test_load_model_and_tokenizer_from_dir_missing_files():
    mod = importlib.import_module("fnd.models.utils")
    with patch("fnd.models.utils.os.path.exists", return_value=False):
        with pytest.raises(Exception):
            mod.load_model_and_tokenizer_from_dir("/nonexistent/dir")
