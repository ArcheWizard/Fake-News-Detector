"""Integration tests for training and evaluation workflows.

These tests verify that the refactored scripts work correctly end-to-end.
"""

import os

import pytest

from fnd.config import FNDConfig
from fnd.data.datasets import load_dataset
from fnd.models.utils import get_model_info, load_model_and_tokenizer_from_dir


class TestConfigIntegration:
    """Test configuration system integration."""

    def test_config_round_trip(self, tmp_path):
        """Test saving and loading configuration."""
        config_path = tmp_path / "test_config.yaml"

        # Create config
        config = FNDConfig(seed=123, model_name="bert-base-uncased", max_seq_length=128)

        # Save
        config.to_yaml(str(config_path))

        # Load
        loaded_config = FNDConfig.from_yaml(str(config_path))

        # Verify
        assert loaded_config.seed == 123
        assert loaded_config.model_name == "bert-base-uncased"
        assert loaded_config.max_seq_length == 128

    def test_config_with_overrides(self, tmp_path):
        """Test CLI override system."""
        config_path = tmp_path / "test_config.yaml"

        # Create base config
        config = FNDConfig()
        config.to_yaml(str(config_path))

        # Load with overrides
        loaded = FNDConfig.from_yaml_with_overrides(
            str(config_path), seed=999, train_epochs=10, train_batch_size=64
        )

        # Verify overrides applied
        assert loaded.seed == 999
        assert loaded.train.epochs == 10
        assert loaded.train.batch_size == 64

        # Verify non-overridden values preserved
        assert loaded.model_name == "roberta-base"


class TestDataLoadingIntegration:
    """Test data loading with configuration."""

    def test_load_dataset_with_config(self, tmp_path):
        """Test loading dataset using config values."""
        # Create test data
        import pandas as pd

        # Create minimal dataset
        texts = [f"Sample text {i}" for i in range(100)]
        labels = [i % 2 for i in range(100)]
        df = pd.DataFrame({"text": texts, "label": labels})
        df.to_csv(tmp_path / "dataset.csv", index=False)

        # Create config
        config = FNDConfig()
        config.paths.data_dir = str(tmp_path)
        config.data.val_size = 0.2
        config.data.test_size = 0.2
        config.seed = 42

        # Load dataset
        bundle = load_dataset(
            config.data.dataset,
            str(tmp_path),
            seed=config.seed,
            val_size=config.data.val_size,
            test_size=config.data.test_size,
        )

        # Verify splits
        assert len(bundle.train_df) > 0
        assert len(bundle.validation_df) > 0
        assert len(bundle.test_df) > 0

        total = len(bundle.train_df) + len(bundle.validation_df) + len(bundle.test_df)
        assert total == 100


class TestModelUtilitiesIntegration:
    """Test model utilities integration."""

    def test_model_info_structure(self):
        """Test get_model_info returns expected structure."""
        # This test uses a real trained model if available
        model_dir = "runs/roberta-kfr/model"

        if not os.path.exists(model_dir):
            pytest.skip(f"Model directory {model_dir} not found")

        info = get_model_info(model_dir)

        # Verify structure
        assert "model_type" in info
        assert "num_labels" in info
        assert "vocab_size" in info
        assert "hidden_size" in info
        assert "num_layers" in info
        assert "id2label" in info
        assert "label2id" in info

        # Verify values
        assert info["model_type"] == "roberta"
        assert info["num_labels"] == 2  # Binary classification
        assert isinstance(info["id2label"], dict)
        assert isinstance(info["label2id"], dict)

    def test_load_model_consistency(self):
        """Test that centralized loading is consistent."""
        model_dir = "runs/roberta-kfr/model"

        if not os.path.exists(model_dir):
            pytest.skip(f"Model directory {model_dir} not found")

        # Load model
        model, tokenizer = load_model_and_tokenizer_from_dir(model_dir)

        # Verify types
        assert model is not None
        assert tokenizer is not None

        # Verify model has expected attributes
        assert hasattr(model, "config")
        assert hasattr(model, "forward")

        # Verify tokenizer has expected attributes
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")


class TestEndToEndWorkflow:
    """Test complete training/evaluation workflow."""

    def test_config_creation_and_validation(self):
        """Test that config can be created and validated."""
        # Create config with custom values
        config = FNDConfig(seed=42, model_name="bert-base-uncased", max_seq_length=128)

        # Verify validation passes
        assert config.seed == 42
        assert config.max_seq_length == 128

        # Verify nested configs created
        assert config.train is not None
        assert config.eval is not None
        assert config.data is not None
        assert config.paths is not None

    def test_invalid_config_raises_error(self):
        """Test that invalid config values raise errors."""
        from fnd.exceptions import ConfigurationError

        # Test negative seed
        with pytest.raises(ConfigurationError):
            FNDConfig(seed=-1)

        # Test invalid max_seq_length
        with pytest.raises(ConfigurationError):
            FNDConfig(max_seq_length=0)

        # Test empty model_name
        with pytest.raises(ConfigurationError):
            FNDConfig(model_name="")
