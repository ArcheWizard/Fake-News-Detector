"""Tests for Streamlit web app."""

import importlib
import pytest

# Streamlit UI is hard to test directly, but we can check that the app runs without error


def test_streamlit_app_runs():
    try:
        importlib.import_module("fnd.web.app")
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
    import streamlit as st

    monkeypatch.setattr(st, "text_area", lambda *a, **kw: None)
    try:
        importlib.import_module("fnd.web.app")
    except Exception as e:
        pytest.fail(f"Streamlit app failed with invalid input: {e}")


# For full UI coverage, use streamlit.testing or simulate user input if possible
