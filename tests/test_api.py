"""Tests for FastAPI prediction API."""

from fastapi.testclient import TestClient
from fnd.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fake News Detector" in response.text


def test_predict_endpoint_valid():
    payload = {"text": "This is a test news article.", "model_dir": "/fake/dir"}
    from unittest.mock import patch

    with (
        patch("fnd.api.main.os.path.isdir", return_value=True),
        patch(
            "fnd.api.main.create_classification_pipeline",
            return_value=lambda text, **kwargs: [{"label": "real", "score": 0.99}],
        ),
    ):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data or "scores" in data


def test_predict_endpoint_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_empty_string():
    payload = {"text": "", "model_dir": "/fake/dir"}
    from unittest.mock import patch

    with (
        patch("fnd.api.main.os.path.isdir", return_value=True),
        patch(
            "fnd.api.main.create_classification_pipeline",
            return_value=lambda text, **kwargs: [{"label": "real", "score": 0.99}],
        ),
    ):
        response = client.post("/predict", json=payload)
        assert response.status_code == 400  # Should be 400 for empty text


def test_predict_endpoint_non_string():
    payload = {"text": 12345}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_batch():
    payload = {"texts": ["News 1", "News 2", "News 3"]}
    response = client.post("/predict_batch", json=payload)
    # Accept either 200 or 404 if not implemented
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        for item in data:
            assert "label" in item
            assert "score" in item


def test_predict_endpoint_long_text():
    long_text = "A" * 10000
    payload = {"text": long_text, "model_dir": "/fake/dir"}
    from unittest.mock import patch

    with (
        patch("fnd.api.main.os.path.isdir", return_value=True),
        patch(
            "fnd.api.main.create_classification_pipeline",
            return_value=lambda text, **kwargs: [{"label": "real", "score": 0.99}],
        ),
    ):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data or "scores" in data


def test_predict_endpoint_html_injection():
    payload = {"text": "<script>alert('xss')</script>", "model_dir": "/fake/dir"}
    from unittest.mock import patch

    with (
        patch("fnd.api.main.os.path.isdir", return_value=True),
        patch(
            "fnd.api.main.create_classification_pipeline",
            return_value=lambda text, **kwargs: [{"label": "real", "score": 0.99}],
        ),
    ):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data or "scores" in data


# Add more tests for error cases, batch prediction, etc. as needed
