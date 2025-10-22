"""Comprehensive tests for configuration module."""
import pytest
import tempfile
from pathlib import Path

from fnd.config import (
    FNDConfig,
    TrainConfig,
    EvalConfig,
    DataConfig,
    PathsConfig,
)
from fnd.exceptions import ConfigurationError


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainConfig()
        assert config.epochs == 3
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.warmup_ratio == 0.1

    def test_validation_negative_epochs(self):
        """Test that negative epochs raise error."""
        with pytest.raises(ConfigurationError, match="epochs must be positive"):
            TrainConfig(epochs=-1)

    def test_validation_zero_batch_size(self):
        """Test that zero batch size raises error."""
        with pytest.raises(ConfigurationError, match="batch_size must be positive"):
            TrainConfig(batch_size=0)

    def test_validation_invalid_warmup_ratio(self):
        """Test that warmup_ratio outside [0, 1] raises error."""
        with pytest.raises(ConfigurationError, match="warmup_ratio must be in"):
            TrainConfig(warmup_ratio=1.5)

    def test_validation_invalid_save_strategy(self):
        """Test that invalid save_strategy raises error."""
        with pytest.raises(ConfigurationError, match="save_strategy must be"):
            TrainConfig(save_strategy="invalid")


class TestEvalConfig:
    """Tests for EvalConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = EvalConfig()
        assert "accuracy" in config.metrics
        assert "f1" in config.metrics
        assert config.batch_size == 32
        assert config.save_plots is True

    def test_validation_invalid_metric(self):
        """Test that invalid metrics raise error."""
        with pytest.raises(ConfigurationError, match="Invalid metrics"):
            EvalConfig(metrics=["accuracy", "invalid_metric"])

    def test_validation_negative_batch_size(self):
        """Test that negative batch size raises error."""
        with pytest.raises(ConfigurationError, match="batch_size must be positive"):
            EvalConfig(batch_size=-1)


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DataConfig()
        assert config.dataset == "kaggle_fake_real"
        assert config.val_size == 0.1
        assert config.test_size == 0.1
        assert config.max_samples is None
        assert config.shuffle is True

    def test_validation_invalid_val_size(self):
        """Test that val_size outside [0, 1) raises error."""
        with pytest.raises(ConfigurationError, match="val_size must be in"):
            DataConfig(val_size=1.5)

    def test_validation_sum_exceeds_one(self):
        """Test that val_size + test_size >= 1 raises error."""
        with pytest.raises(ConfigurationError, match="val_size \\+ test_size must be"):
            DataConfig(val_size=0.6, test_size=0.5)

    def test_validation_negative_max_samples(self):
        """Test that negative max_samples raises error."""
        with pytest.raises(ConfigurationError, match="max_samples must be positive"):
            DataConfig(max_samples=-10)


class TestPathsConfig:
    """Tests for PathsConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = PathsConfig()
        assert "data" in config.data_dir or "kaggle_fake_real" in config.data_dir
        assert "runs" in config.runs_dir
        assert "models" in config.models_dir

    def test_paths_are_absolute(self):
        """Test that paths are converted to absolute."""
        config = PathsConfig(data_dir="./data")
        assert Path(config.data_dir).is_absolute()


class TestFNDConfig:
    """Tests for main FNDConfig class."""

    def test_default_values(self):
        """Test that default configuration is valid."""
        config = FNDConfig()
        assert config.seed == 42
        assert config.model_name == "roberta-base"
        assert config.max_seq_length == 256
        assert isinstance(config.train, TrainConfig)
        assert isinstance(config.eval, EvalConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.paths, PathsConfig)

    def test_validation_negative_seed(self):
        """Test that negative seed raises error."""
        with pytest.raises(ConfigurationError, match="seed must be non-negative"):
            FNDConfig(seed=-1)

    def test_validation_empty_model_name(self):
        """Test that empty model_name raises error."""
        with pytest.raises(ConfigurationError, match="model_name cannot be empty"):
            FNDConfig(model_name="")

    def test_validation_invalid_max_seq_length(self):
        """Test that zero/negative max_seq_length raises error."""
        with pytest.raises(ConfigurationError, match="max_seq_length must be positive"):
            FNDConfig(max_seq_length=0)


class TestYAMLLoading:
    """Tests for YAML loading functionality."""

    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        yaml_content = """
seed: 123
model_name: bert-base-uncased
max_seq_length: 512
train:
  epochs: 5
  batch_size: 32
eval:
  batch_size: 64
data:
  val_size: 0.15
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = FNDConfig.from_yaml(str(yaml_file))

        assert config.seed == 123
        assert config.model_name == "bert-base-uncased"
        assert config.max_seq_length == 512
        assert config.train.epochs == 5
        assert config.train.batch_size == 32
        assert config.eval.batch_size == 64
        assert config.data.val_size == 0.15

    def test_load_missing_file(self, tmp_path):
        """Test that loading missing file raises error."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            FNDConfig.from_yaml(str(tmp_path / "nonexistent.yaml"))

    def test_load_empty_yaml(self, tmp_path):
        """Test loading empty YAML file uses defaults."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        config = FNDConfig.from_yaml(str(yaml_file))

        # Should use defaults
        assert config.seed == 42
        assert config.model_name == "roberta-base"

    def test_load_partial_yaml(self, tmp_path):
        """Test loading partial YAML uses defaults for missing values."""
        yaml_content = """
seed: 999
train:
  epochs: 10
"""
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text(yaml_content)

        config = FNDConfig.from_yaml(str(yaml_file))

        assert config.seed == 999
        assert config.train.epochs == 10
        # Other values should be defaults
        assert config.model_name == "roberta-base"
        assert config.train.batch_size == 16


class TestCLIOverrides:
    """Tests for CLI override functionality."""

    def test_override_top_level(self, tmp_path):
        """Test overriding top-level configuration."""
        yaml_content = "seed: 42"
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = FNDConfig.from_yaml_with_overrides(
            str(yaml_file),
            seed=999
        )

        assert config.seed == 999

    def test_override_nested(self, tmp_path):
        """Test overriding nested configuration."""
        yaml_content = """
train:
  epochs: 3
  batch_size: 16
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = FNDConfig.from_yaml_with_overrides(
            str(yaml_file),
            train_epochs=10,
            train_batch_size=64
        )

        assert config.train.epochs == 10
        assert config.train.batch_size == 64

    def test_override_multiple_nested(self, tmp_path):
        """Test overriding multiple nested configurations."""
        yaml_content = "seed: 42"
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = FNDConfig.from_yaml_with_overrides(
            str(yaml_file),
            train_epochs=5,
            eval_batch_size=128,
            data_val_size=0.2
        )

        assert config.train.epochs == 5
        assert config.eval.batch_size == 128
        assert config.data.val_size == 0.2

    def test_override_invalid_key(self, tmp_path):
        """Test that invalid override key raises error."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("seed: 42")

        with pytest.raises(ConfigurationError, match="Invalid override key"):
            FNDConfig.from_yaml_with_overrides(
                str(yaml_file),
                invalid_key=123
            )

    def test_override_none_values_ignored(self, tmp_path):
        """Test that None override values are ignored."""
        yaml_content = "seed: 42"
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = FNDConfig.from_yaml_with_overrides(
            str(yaml_file),
            seed=None,  # Should be ignored
            train_epochs=5
        )

        assert config.seed == 42  # Not overridden
        assert config.train.epochs == 5  # Overridden


class TestYAMLSaving:
    """Tests for YAML saving functionality."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = FNDConfig(seed=123, model_name="bert-base-uncased")
        config_dict = config.to_dict()

        assert config_dict["seed"] == 123
        assert config_dict["model_name"] == "bert-base-uncased"
        assert "train" in config_dict
        assert "eval" in config_dict

    def test_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file."""
        config = FNDConfig(seed=456, model_name="distilbert-base-uncased")
        yaml_file = tmp_path / "output.yaml"

        config.to_yaml(str(yaml_file))

        assert yaml_file.exists()

        # Load it back and verify
        loaded_config = FNDConfig.from_yaml(str(yaml_file))
        assert loaded_config.seed == 456
        assert loaded_config.model_name == "distilbert-base-uncased"

    def test_round_trip(self, tmp_path):
        """Test that save and load preserves configuration."""
        original = FNDConfig(
            seed=789,
            model_name="roberta-large",
            max_seq_length=512,
            train=TrainConfig(epochs=10, batch_size=8),
            data=DataConfig(val_size=0.2, test_size=0.2)
        )

        yaml_file = tmp_path / "roundtrip.yaml"
        original.to_yaml(str(yaml_file))
        loaded = FNDConfig.from_yaml(str(yaml_file))

        assert loaded.seed == original.seed
        assert loaded.model_name == original.model_name
        assert loaded.max_seq_length == original.max_seq_length
        assert loaded.train.epochs == original.train.epochs
        assert loaded.train.batch_size == original.train.batch_size
        assert loaded.data.val_size == original.data.val_size
