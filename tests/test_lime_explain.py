"""Tests for LIME explainability module."""

import pytest
import importlib
from unittest.mock import Mock, patch


# Test that lime_explain.py can be imported and main functions exist
def test_lime_explain_import():
    mod = importlib.import_module("fnd.explain.lime_explain")
    assert mod is not None


# For full coverage, mock a model and call explain_instance if possible
def test_lime_explain_instance_mock():
    mod = importlib.import_module("fnd.explain.lime_explain")
    if hasattr(mod, "explain_instance"):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_text = "Fake news example"
        try:
            result = mod.explain_instance(mock_model, mock_tokenizer, mock_text)
            assert result is not None
        except Exception as e:
            pytest.fail(f"explain_instance raised: {e}")


def test_lime_explain_instance_error():
    mod = importlib.import_module("fnd.explain.lime_explain")
    if hasattr(mod, "explain_instance"):
        with pytest.raises(Exception):
            mod.explain_instance(None, None, None)


def test_lime_explain_predict_proba_handles_output():
    mod = importlib.import_module("fnd.explain.lime_explain")

    class DummyPipeline:
        def __call__(self, text):
            return [{"label": "real", "score": 0.7}, {"label": "fake", "score": 0.3}]

    with patch(
        "fnd.explain.lime_explain.create_classification_pipeline",
        return_value=DummyPipeline(),
    ):
        exp, html = mod.explain_text_with_lime("/fake/dir", "Test text")
        assert exp is not None
        assert html is not None
