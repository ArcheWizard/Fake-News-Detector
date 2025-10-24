"""Configuration management with YAML and CLI override support.

This module provides a hierarchical configuration system using dataclasses,
supporting both YAML file loading and command-line argument overrides.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fnd.exceptions import ConfigurationError


@dataclass
class TrainConfig:
    """Training hyperparameters and settings.

    Attributes:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization coefficient
        warmup_ratio: Proportion of training for learning rate warmup
        gradient_accumulation_steps: Steps to accumulate gradients before update
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Whether to use mixed precision training (FP16)
        save_strategy: When to save checkpoints ('epoch', 'steps', or 'no')
        save_steps: Save checkpoint every N steps (if save_strategy='steps')
        evaluation_strategy: When to run evaluation ('epoch', 'steps', or 'no')
        eval_steps: Run evaluation every N steps (if evaluation_strategy='steps')
        logging_steps: Log metrics every N steps
        load_best_model_at_end: Load best model at end of training
        metric_for_best_model: Metric to use for best model selection
    """

    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = False
    save_strategy: str = "epoch"
    save_steps: int = 500
    evaluation_strategy: str = "epoch"
    eval_steps: int = 500
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    # Advanced knobs
    lr_scheduler_type: str = "linear"  # linear | cosine | cosine_with_restarts | polynomial | constant | constant_with_warmup
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = False
    bf16: bool = False
    torch_compile: bool = False
    optim: str = "adamw_torch"  # common fast default

    def __post_init__(self):
        """Validate training configuration."""
        if self.epochs <= 0:
            raise ConfigurationError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ConfigurationError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        if self.learning_rate <= 0:
            raise ConfigurationError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if not 0 <= self.warmup_ratio <= 1:
            raise ConfigurationError(
                f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}"
            )
        if self.save_strategy not in {"epoch", "steps", "no"}:
            raise ConfigurationError(
                f"save_strategy must be 'epoch', 'steps', or 'no', got {self.save_strategy}"
            )
        if self.evaluation_strategy not in {"epoch", "steps", "no"}:
            raise ConfigurationError(
                f"evaluation_strategy must be 'epoch', 'steps', or 'no', got {self.evaluation_strategy}"
            )
        valid_schedulers = {
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        }
        if self.lr_scheduler_type not in valid_schedulers:
            raise ConfigurationError(
                f"lr_scheduler_type must be one of {valid_schedulers}, got {self.lr_scheduler_type}"
            )
        if self.dataloader_num_workers < 0:
            raise ConfigurationError(
                f"dataloader_num_workers must be >= 0, got {self.dataloader_num_workers}"
            )
        valid_optims = {"adamw_torch", "adamw_hf", "adamw_torch_fused", "adafactor"}
        if self.optim not in valid_optims:
            raise ConfigurationError(
                f"optim must be one of {valid_optims}, got {self.optim}"
            )


@dataclass
class EvalConfig:
    """Evaluation settings and metrics.

    Attributes:
        metrics: List of metrics to compute during evaluation
        batch_size: Evaluation batch size
        save_plots: Whether to save visualization plots (confusion matrix, ROC curve)
    """

    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "f1", "precision", "recall", "roc_auc"]
    )
    batch_size: int = 32
    save_plots: bool = True

    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.batch_size <= 0:
            raise ConfigurationError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        valid_metrics = {"accuracy", "f1", "precision", "recall", "roc_auc"}
        invalid = set(self.metrics) - valid_metrics
        if invalid:
            raise ConfigurationError(
                f"Invalid metrics: {invalid}. Valid options: {valid_metrics}"
            )


@dataclass
class DataConfig:
    """Data loading and preprocessing settings.

    Attributes:
        dataset: Dataset name/identifier
        text_field: Name of text column in dataset
        label_field: Name of label column in dataset
        val_size: Fraction of data for validation set
        test_size: Fraction of data for test set
        max_samples: Optional limit on total samples (None = all)
        shuffle: Whether to shuffle data before splitting
    """

    dataset: str = "kaggle_fake_real"
    text_field: str = "text"
    label_field: str = "label"
    val_size: float = 0.1
    test_size: float = 0.1
    max_samples: int | None = None
    shuffle: bool = True

    def __post_init__(self):
        """Validate data configuration."""
        if not 0 <= self.val_size < 1:
            raise ConfigurationError(f"val_size must be in [0, 1), got {self.val_size}")
        if not 0 <= self.test_size < 1:
            raise ConfigurationError(
                f"test_size must be in [0, 1), got {self.test_size}"
            )
        if self.val_size + self.test_size >= 1:
            raise ConfigurationError(
                f"val_size + test_size must be < 1, got {self.val_size + self.test_size}"
            )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ConfigurationError(
                f"max_samples must be positive or None, got {self.max_samples}"
            )


@dataclass
class PathsConfig:
    """File paths and directories.

    Attributes:
        data_dir: Directory containing dataset files
        runs_dir: Directory for training run outputs
        models_dir: Directory for saved models
    """

    data_dir: str = "data/processed/kaggle_fake_real"
    runs_dir: str = "runs"
    models_dir: str = "models"

    def __post_init__(self):
        """Expand paths to absolute."""
        self.data_dir = str(Path(self.data_dir).expanduser().absolute())
        self.runs_dir = str(Path(self.runs_dir).expanduser().absolute())
        self.models_dir = str(Path(self.models_dir).expanduser().absolute())


@dataclass
class FNDConfig:
    """Main configuration container for Fake News Detector.

    This is the top-level configuration object that contains all sub-configurations
    for training, evaluation, data, and paths.

    Attributes:
        seed: Random seed for reproducibility
        model_name: HuggingFace model identifier (e.g., 'roberta-base')
        max_seq_length: Maximum sequence length for tokenization
        train: Training configuration
        eval: Evaluation configuration
        data: Data configuration
        paths: Paths configuration

    Examples:
        Load from YAML:
        >>> config = FNDConfig.from_yaml("config/config.yaml")

        Load with CLI overrides:
        >>> config = FNDConfig.from_yaml_with_overrides(
        ...     "config/config.yaml",
        ...     seed=123,
        ...     train_epochs=5,
        ...     train_batch_size=32
        ... )

        Create programmatically:
        >>> config = FNDConfig(
        ...     seed=42,
        ...     model_name="bert-base-uncased",
        ...     train=TrainConfig(epochs=5, batch_size=32)
        ... )
    """

    seed: int = 42
    model_name: str = "roberta-base"
    max_seq_length: int = 256
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self):
        """Validate main configuration."""
        if self.seed < 0:
            raise ConfigurationError(f"seed must be non-negative, got {self.seed}")
        if self.max_seq_length <= 0:
            raise ConfigurationError(
                f"max_seq_length must be positive, got {self.max_seq_length}"
            )
        if not self.model_name:
            raise ConfigurationError("model_name cannot be empty")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FNDConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            FNDConfig instance with values loaded from YAML

        Raises:
            ConfigurationError: If file not found or YAML is invalid

        Example:
            >>> config = FNDConfig.from_yaml("config/config.yaml")
        """
        yaml_path_obj = Path(yaml_path).expanduser()
        if not yaml_path_obj.exists():
            raise ConfigurationError(f"Configuration file not found: {yaml_path_obj}")

        try:
            with open(yaml_path_obj) as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {yaml_path_obj}: {e}") from e

        if config_dict is None:
            config_dict = {}

        return cls(
            seed=config_dict.get("seed", 42),
            model_name=config_dict.get("model_name", "roberta-base"),
            max_seq_length=config_dict.get("max_seq_length", 256),
            train=TrainConfig(**config_dict.get("train", {})),
            eval=EvalConfig(**config_dict.get("eval", {})),
            data=DataConfig(**config_dict.get("data", {})),
            paths=PathsConfig(**config_dict.get("paths", {})),
        )

    @classmethod
    def from_yaml_with_overrides(cls, yaml_path: str, **overrides) -> "FNDConfig":
        """Load from YAML and apply CLI overrides.

        Overrides use dot notation for nested fields, e.g.:
        - 'seed=42' overrides top-level seed
        - 'train.epochs=5' overrides train config epochs
        - 'paths.data_dir=/path/to/data' overrides paths config data_dir

        Args:
            yaml_path: Path to YAML configuration file
            **overrides: Keyword arguments for overrides using underscore notation
                (e.g., train_epochs=5 becomes train.epochs)

        Returns:
            FNDConfig instance with YAML values and CLI overrides applied

        Raises:
            ConfigurationError: If override path is invalid

        Example:
            >>> config = FNDConfig.from_yaml_with_overrides(
            ...     "config/config.yaml",
            ...     seed=123,
            ...     train_epochs=5,
            ...     train_batch_size=32,
            ...     data_val_size=0.15
            ... )
        """
        config = cls.from_yaml(yaml_path)

        for key, value in overrides.items():
            if value is None:
                continue

            # Convert underscore notation to dot notation
            # train_epochs -> train.epochs
            parts = key.split("_", 1)
            if len(parts) == 2 and hasattr(config, parts[0]):
                obj = getattr(config, parts[0])
                attr_name = parts[1]
                if hasattr(obj, attr_name):
                    setattr(obj, attr_name, value)
                else:
                    raise ConfigurationError(
                        f"Invalid override key: {key} (no attribute '{attr_name}' in '{parts[0]}')"
                    )
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ConfigurationError(f"Invalid override key: {key}")

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = FNDConfig()
            >>> config_dict = config.to_dict()
        """
        return asdict(self)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path where YAML file will be saved

        Example:
            >>> config = FNDConfig()
            >>> config.to_yaml("output/config.yaml")
        """
        yaml_path_obj = Path(yaml_path).expanduser()
        yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path_obj, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
