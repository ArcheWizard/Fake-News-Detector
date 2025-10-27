"""Comprehensive tests for training metrics computation."""

import numpy as np
import pytest

from fnd.training.metrics import _softmax, compute_metrics


class TestSoftmax:
    """Tests for softmax normalization function."""

    def test_softmax_normalization(self):
        """Test that softmax produces probabilities that sum to 1."""
        logits = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]])
        probs = _softmax(logits)

        assert probs.shape == logits.shape
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_softmax_output_range(self):
        """Test that softmax outputs are in valid probability range [0, 1]."""
        logits = np.array([[1.0, 2.0], [0.5, 1.5], [-1.0, 3.0]])
        probs = _softmax(logits)

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_softmax_stability_with_large_values(self):
        """Test that softmax is numerically stable with large values."""
        logits = np.array([[1000.0, 1001.0], [500.0, 499.0]])
        probs = _softmax(logits)

        # Should not overflow or produce NaN
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_softmax_with_negative_values(self):
        """Test softmax with negative logits."""
        logits = np.array([[-2.0, -1.0], [-5.0, -3.0]])
        probs = _softmax(logits)

        assert probs.shape == logits.shape
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_softmax_equal_logits(self):
        """Test that equal logits produce equal probabilities."""
        logits = np.array([[1.0, 1.0], [0.0, 0.0]])
        probs = _softmax(logits)

        # Equal logits should give equal probabilities (0.5 each)
        assert np.allclose(probs, 0.5)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    @pytest.fixture
    def mock_eval_pred(self):
        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

        return MockEvalPred

    @pytest.mark.parametrize(
        "logits,labels,expected_accuracy",
        [
            (
                np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]]),
                np.array([0, 1, 0, 1]),
                1.0,
            ),
            (
                np.array([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 10.0]]),
                np.array([1, 1, 0, 0]),
                0.0,
            ),
        ],
    )
    def test_prediction_scenarios(
        self, mock_eval_pred, logits, labels, expected_accuracy
    ):
        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)
        assert metrics["accuracy"] == expected_accuracy

    def test_balanced_predictions(self, mock_eval_pred):
        """Test metrics with balanced 50% accuracy."""
        logits = np.array(
            [
                [10.0, 0.0],  # Correct
                [0.0, 10.0],  # Correct
                [10.0, 0.0],  # Wrong
                [0.0, 10.0],  # Wrong
            ]
        )
        labels = np.array([0, 1, 1, 0])

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        assert metrics["accuracy"] == 0.5
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_handles_dict_format_eval_pred(self):
        """Test that compute_metrics handles dict-format EvalPrediction."""
        logits = np.array([[10.0, 0.0], [0.0, 10.0]])
        labels = np.array([0, 1])

        eval_pred = {"predictions": logits, "label_ids": labels}

        metrics = compute_metrics(eval_pred)

        assert metrics["accuracy"] == 1.0
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_metrics_keys_present(self, mock_eval_pred):
        """Test that all expected metric keys are present."""
        logits = np.array([[10.0, 0.0], [0.0, 10.0]])
        labels = np.array([0, 1])

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        expected_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for key in expected_keys:
            assert key in metrics, f"Missing expected metric: {key}"

    def test_metrics_are_floats(self, mock_eval_pred):
        """Test that all metrics are returned as float type."""
        logits = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
        labels = np.array([0, 1, 0])

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        for key, value in metrics.items():
            assert isinstance(value, float) or np.isnan(value), (
                f"{key} is not a float: {type(value)}"
            )

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_roc_auc_with_single_class_labels(self, mock_eval_pred):
        """Test that ROC AUC handles case with only one class present."""
        logits = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
        labels = np.array([0, 0, 0])  # Only class 0

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        # ROC AUC should be NaN or handled gracefully
        assert "roc_auc" in metrics
        assert isinstance(metrics["roc_auc"], float)

    def test_probabilistic_predictions(self, mock_eval_pred):
        """Test metrics with soft probability predictions."""
        # Probabilities that need softmax normalization
        logits = np.array(
            [
                [2.0, 1.0],  # Slightly favor class 0
                [1.0, 2.5],  # Favor class 1
                [3.0, 0.5],  # Strongly favor class 0
                [0.5, 3.0],  # Strongly favor class 1
            ]
        )
        labels = np.array([0, 1, 0, 1])

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        assert metrics["accuracy"] == 1.0
        assert 0.9 < metrics["roc_auc"] <= 1.0  # High AUC for perfect separation
        assert metrics["f1"] == 1.0

    def test_large_batch_size(self, mock_eval_pred):
        """Test metrics computation with large batch size."""
        np.random.seed(42)
        n_samples = 1000

        # Create random logits and labels
        logits = np.random.randn(n_samples, 2)
        labels = np.random.randint(0, 2, n_samples)

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        # All metrics should be valid numbers
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert not np.isnan(metrics["roc_auc"])

    def test_metrics_large_batch_performance(self, mock_eval_pred):
        import time

        n_samples = 100_000
        logits = np.random.randn(n_samples, 2)
        labels = np.random.randint(0, 2, n_samples)
        eval_pred = mock_eval_pred(logits, labels)
        start = time.time()
        metrics = compute_metrics(eval_pred)
        duration = time.time() - start
        assert duration < 5  # Should compute in under 5 seconds
        assert 0 <= metrics["accuracy"] <= 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def mock_eval_pred(self):
        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

        return MockEvalPred

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_single_sample(self, mock_eval_pred):
        """Test metrics with only one sample."""
        logits = np.array([[10.0, 0.0]])
        labels = np.array([0])

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        assert metrics["accuracy"] == 1.0

    def test_zero_logits(self, mock_eval_pred):
        """Test metrics with all-zero logits."""
        logits = np.array([[0.0, 0.0], [0.0, 0.0]])  # Equal probs for both classes
        labels = np.array([0, 1])

        eval_pred = mock_eval_pred(logits, labels)
        metrics = compute_metrics(eval_pred)

        # Accuracy: Random guess ~0.5
        assert 0.0 <= metrics["accuracy"] <= 1.0
        # Other metrics should be valid floats
        assert isinstance(metrics["f1"], float)
        assert isinstance(metrics["roc_auc"], float)
