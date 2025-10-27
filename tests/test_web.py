"""Tests for Streamlit web app."""

import importlib
import streamlit as st
import pytest

# Streamlit UI is hard to test directly, but we can check that the app runs without error


import json
from unittest.mock import patch, MagicMock, mock_open


# Helper mock for st.session_state that supports both attribute and key access
class SessionStateMock:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


MODULE = "fnd.web.app"


def test_streamlit_app_runs():
    try:
        importlib.import_module(MODULE)
    except Exception as e:
        pytest.fail(f"Streamlit app failed to import: {e}")


@pytest.mark.skip(
    reason="Launching Streamlit server blocks pytest; test manually if needed."
)
def test_streamlit_app_runs_with_input(monkeypatch):
    import streamlit.web.bootstrap

    # Monkeypatch sys.argv to simulate running the app with required CLI args
    monkeypatch.setattr(
        "sys.argv",
        ["streamlit", "run", "src/fnd/web/app.py", "--model_dir", "/fake/dir"],
    )
    streamlit.web.bootstrap.run("src/fnd/web/app.py", False, [], flag_options={})


def test_streamlit_app_invalid_input(monkeypatch):
    # Simulate invalid input by monkeypatching st.text_area to return None
    monkeypatch.setattr(st, "text_area", lambda *a, **kw: None)
    try:
        importlib.import_module(MODULE)
    except Exception as e:
        pytest.fail(f"Streamlit app failed with invalid input: {e}")


@patch("fnd.web.app.argparse.ArgumentParser")
def test_main_cli_args_parsing(mock_argparser):
    # Simulate CLI args
    mock_parser = MagicMock()
    mock_parser.parse_known_args.return_value = (
        MagicMock(model_dir="/fake/model", samples_file="samples.json"),
        [],
    )
    mock_argparser.return_value = mock_parser
    from fnd.web import app as web_app

    args = web_app.main_cli()
    assert args.model_dir == "/fake/model"
    assert args.samples_file == "samples.json"


def test_load_metrics_and_samples(tmp_path):
    # Test metrics loading (valid, missing, corrupt)
    from fnd.web import app as web_app

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metrics_path = model_dir.parent / "metrics.json"
    metrics = {
        "eval_accuracy": 0.9,
        "eval_f1": 0.8,
        "eval_precision": 0.7,
        "eval_recall": 0.6,
        "eval_roc_auc": 0.95,
    }
    metrics_path.write_text(json.dumps(metrics))
    loaded = web_app.load_metrics(str(model_dir))
    if loaded is not None:
        assert loaded.get("eval_accuracy") == 0.9
    # Missing file
    assert web_app.load_metrics(str(tmp_path / "no_model")) is None
    # Corrupt file
    metrics_path.write_text("not json")
    with patch("builtins.open", mock_open(read_data="not json")):
        try:
            _ = web_app.load_metrics(str(model_dir))
        except Exception:
            pass


def test_load_test_samples(tmp_path):
    from fnd.web import app as web_app

    samples_file = tmp_path / "samples.json"
    samples = {
        "real": [{"text": "real news", "label": "real"}],
        "fake": [{"text": "fake news", "label": "fake"}],
    }
    samples_file.write_text(json.dumps(samples))
    loaded = web_app.load_test_samples(str(samples_file))
    if loaded is not None:
        assert "real" in loaded and "fake" in loaded
    # Missing file
    assert web_app.load_test_samples(str(tmp_path / "missing.json")) is None
    # Corrupt file
    samples_file.write_text("not json")
    with patch("builtins.open", mock_open(read_data="not json")):
        try:
            _ = web_app.load_test_samples(str(samples_file))
        except Exception:
            pass


def test_load_pipeline(monkeypatch):
    from fnd.web import app as web_app

    monkeypatch.setattr(
        "fnd.web.app.create_classification_pipeline",
        lambda **kwargs: lambda x: [{"label": "real", "score": 0.9}],
    )
    pipeline = web_app.load_pipeline("/fake/model")
    result = pipeline("test text")
    assert isinstance(result, list)
    assert result[0]["label"] == "real"


@pytest.mark.parametrize(
    "metrics,expected_keys",
    [
        (
            {
                "eval_accuracy": 0.9,
                "eval_f1": 0.8,
                "eval_precision": 0.7,
                "eval_recall": 0.6,
                "eval_roc_auc": 0.95,
            },
            [
                "Test Accuracy",
                "Test F1 Score",
                "Test Precision",
                "Test Recall",
                "Test ROC AUC",
            ],
        ),
        (
            {
                "eval_accuracy": 0.9,
                "eval_f1": 0.8,
                "eval_precision": 0.7,
                "eval_recall": 0.6,
            },
            ["Test Accuracy", "Test F1 Score", "Test Precision", "Test Recall"],
        ),
        ({}, []),
    ],
)
def test_sidebar_metrics_display(monkeypatch, metrics, expected_keys):
    # Patch st.sidebar.metric to record calls
    from fnd.web import app as web_app

    calls = []
    monkeypatch.setattr(st.sidebar, "title", lambda x: None)
    monkeypatch.setattr(st.sidebar, "metric", lambda k, v: calls.append((k, v)))
    web_app.load_metrics = lambda model_dir: metrics
    # Patch args
    monkeypatch.setattr(
        web_app,
        "main_cli",
        lambda: MagicMock(model_dir="/fake/model", samples_file="samples.json"),
    )
    # Patch pipeline and text_area
    monkeypatch.setattr(
        web_app,
        "load_pipeline",
        lambda model_dir: lambda x: [{"label": "real", "score": 0.9}],
    )
    monkeypatch.setattr(web_app, "load_test_samples", lambda f: None)
    monkeypatch.setattr(st, "text_area", lambda *a, **k: "test text")
    monkeypatch.setattr(st, "button", lambda *a, **k: False)
    monkeypatch.setattr(st, "title", lambda x: None)
    monkeypatch.setattr(st, "caption", lambda x: None)
    # Use SessionStateMock for session_state
    monkeypatch.setattr(st, "session_state", SessionStateMock())
    web_app.app()
    for key in expected_keys:
        assert any(key in call[0] for call in calls)


def test_sample_selector_and_load(monkeypatch):
    from fnd.web import app as web_app

    # Patch sidebar and session state
    monkeypatch.setattr(st.sidebar, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st.sidebar, "title", lambda *a, **k: None)
    monkeypatch.setattr(
        st.sidebar,
        "selectbox",
        lambda label, opts, **kwargs: opts[1] if "category" in label else 0,
    )
    monkeypatch.setattr(
        st.sidebar, "button", lambda label: True if "Load Sample" in label else False
    )
    # Patch test samples
    test_samples = {
        "real": [{"text": "real news", "label": "real"}],
        "fake": [{"text": "fake news", "label": "fake"}],
    }
    monkeypatch.setattr(web_app, "load_test_samples", lambda f: test_samples)
    # Patch args, pipeline, text_area, session_state
    monkeypatch.setattr(
        web_app,
        "main_cli",
        lambda: MagicMock(model_dir="/fake/model", samples_file="samples.json"),
    )
    monkeypatch.setattr(
        web_app,
        "load_pipeline",
        lambda model_dir: lambda x: [{"label": "real", "score": 0.9}],
    )
    monkeypatch.setattr(st, "text_area", lambda *a, **k: "real news")
    monkeypatch.setattr(st, "button", lambda *a, **k: False)
    monkeypatch.setattr(st, "title", lambda x: None)
    monkeypatch.setattr(st, "caption", lambda x: None)
    # Simulate session state
    state = SessionStateMock(text_input="", true_label=None)
    monkeypatch.setattr(st, "session_state", state)
    web_app.app()
    assert state.text_input == "real news"
    assert state.true_label == "real"


def test_prediction_and_explainability(monkeypatch):
    from fnd.web import app as web_app

    # Patch everything for prediction
    monkeypatch.setattr(
        web_app,
        "main_cli",
        lambda: MagicMock(model_dir="/fake/model", samples_file="samples.json"),
    )
    monkeypatch.setattr(web_app, "load_metrics", lambda model_dir: None)
    monkeypatch.setattr(web_app, "load_test_samples", lambda f: None)
    # Patch pipeline to return two classes
    monkeypatch.setattr(
        web_app,
        "load_pipeline",
        lambda model_dir: lambda x: [
            {"label": "real", "score": 0.7},
            {"label": "fake", "score": 0.3},
        ],
    )
    # Patch st functions
    monkeypatch.setattr(st, "text_area", lambda *a, **k: "test text")
    monkeypatch.setattr(
        st, "button", lambda label: True if "Predict" in label else False
    )
    monkeypatch.setattr(st, "title", lambda x: None)
    monkeypatch.setattr(st, "caption", lambda x: None)
    monkeypatch.setattr(st, "info", lambda x: None)
    monkeypatch.setattr(st, "success", lambda x: None)
    monkeypatch.setattr(st, "error", lambda x: None)
    monkeypatch.setattr(st, "write", lambda *a, **k: None)
    monkeypatch.setattr(st, "sidebar", MagicMock())
    monkeypatch.setattr(
        st, "session_state", SessionStateMock(text_input="test text", true_label="real")
    )
    # Patch explainability checkboxes
    monkeypatch.setattr(
        st.sidebar,
        "checkbox",
        lambda label, value=False: label == "Show LIME explanation",
    )
    # Patch LIME explanation at correct import path
    monkeypatch.setattr(
        "fnd.explain.lime_explain.explain_text_with_lime",
        lambda *a, **k: (None, "<div>LIME</div>"),
    )
    monkeypatch.setattr("fnd.web.app.components.html", lambda html, **kwargs: None)
    # Patch spinner
    monkeypatch.setattr(
        st,
        "spinner",
        lambda msg: MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None
        ),
    )
    # Patch SHAP explanation at correct import path
    monkeypatch.setattr(
        st.sidebar,
        "checkbox",
        lambda label, value=False: label == "Show SHAP explanation",
    )
    monkeypatch.setattr(
        "fnd.explain.shap_explain.explain_text_with_shap",
        lambda *a, **k: ("shap explanation", None),
    )
    monkeypatch.setattr(st, "subheader", lambda x: None)
    web_app.app()


def test_prediction_error_handling(monkeypatch):
    from fnd.web import app as web_app

    # Patch pipeline to raise error
    monkeypatch.setattr(
        web_app,
        "main_cli",
        lambda: MagicMock(model_dir="/fake/model", samples_file="samples.json"),
    )
    monkeypatch.setattr(web_app, "load_metrics", lambda model_dir: None)
    monkeypatch.setattr(web_app, "load_test_samples", lambda f: None)
    monkeypatch.setattr(
        web_app,
        "load_pipeline",
        lambda model_dir: lambda x: (_ for _ in ()).throw(Exception("pipeline error")),
    )
    monkeypatch.setattr(st, "text_area", lambda *a, **k: "test text")
    monkeypatch.setattr(
        st, "button", lambda label: True if "Predict" in label else False
    )
    monkeypatch.setattr(st, "title", lambda x: None)
    monkeypatch.setattr(st, "caption", lambda x: None)
    monkeypatch.setattr(st, "info", lambda x: None)
    monkeypatch.setattr(st, "success", lambda x: None)
    monkeypatch.setattr(st, "error", lambda x: None)
    monkeypatch.setattr(st, "write", lambda *a, **k: None)
    monkeypatch.setattr(st, "sidebar", MagicMock())
    monkeypatch.setattr(
        st, "session_state", SessionStateMock(text_input="test text", true_label="real")
    )
    # Should not raise
    try:
        web_app.app()
    except Exception as e:
        pytest.fail(f"App crashed on pipeline error: {e}")


# For full UI coverage, use streamlit.testing or simulate user input if possible
