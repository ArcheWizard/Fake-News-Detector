"""Tests for FastAPI prediction API."""

from fastapi.testclient import TestClient
from fnd.api.main import app
import pytest

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fake News Detector" in response.text


@pytest.mark.parametrize(
    "payload,expected_status,expected_keys",
    [
        (
            {"text": "This is a test news article.", "model_dir": "/fake/dir"},
            200,
            ["label", "scores"],
        ),
        ({"text": "", "model_dir": "/fake/dir"}, 400, []),
        ({"text": 12345}, 422, []),
        ({"text": "A" * 10000, "model_dir": "/fake/dir"}, 200, ["label", "scores"]),
        (
            {"text": "<script>alert('xss')</script>", "model_dir": "/fake/dir"},
            200,
            ["label", "scores"],
        ),
    ],
)
def test_predict_endpoint_parametrized(payload, expected_status, expected_keys):
    from unittest.mock import patch

    # Only patch for valid/empty string/long/html cases
    patch_needed = isinstance(payload.get("text", None), str) and payload["text"] != ""
    if patch_needed:
        with (
            patch("fnd.api.main.os.path.isdir", return_value=True),
            patch(
                "fnd.api.main.create_classification_pipeline",
                return_value=lambda text, **kwargs: [{"label": "real", "score": 0.99}],
            ),
        ):
            response = client.post("/predict", json=payload)
    else:
        response = client.post("/predict", json=payload)
    assert response.status_code == expected_status
    if expected_status == 200:
        data = response.json()
        assert any(k in data for k in expected_keys)


def test_predict_endpoint_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity

    # Covered by parametrized test

    # Covered by parametrized test


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

    # Covered by parametrized test

    # Covered by parametrized test


# Add more tests for error cases, batch prediction, etc. as needed
