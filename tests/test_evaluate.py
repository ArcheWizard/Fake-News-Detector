"""Tests for evaluation CLI and metrics."""

import importlib
import types
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, mock_open
from conftest import DummyConfig


# --- Shared test helpers ---
class FakeDataset:
    def map(self, *a, **k):
        return self


class DummyTrainer:
    def __init__(self, **kwargs):
        pass

    def evaluate(self, x):
        return {"accuracy": 1.0}

    def predict(self, x):
        # Default: one prediction, one label
        return types.SimpleNamespace(predictions=[[0.0, 1.0]], label_ids=[1])


class DummyTrainerWithPlots(DummyTrainer):
    def predict(self, x):
        # Two predictions, two labels for plot tests
        return types.SimpleNamespace(
            predictions=np.array([[0.0, 1.0], [1.0, 0.0]]), label_ids=np.array([1, 0])
        )


class DummyTrainerEmpty(DummyTrainer):
    def predict(self, x):
        return types.SimpleNamespace(predictions=[], label_ids=[])


class PlotConfig(DummyConfig):
    class eval:
        save_plots = True


# --- Scenario-based and file output tests ---
@pytest.mark.parametrize(
    "cli_args,expect_metrics,expect_plots",
    [
        (["--model_dir", "/fake/model", "--out_dir", "tmp"], True, False),
        (
            ["--model_dir", "/fake/model", "--out_dir", "tmp", "--config", "fake.yaml"],
            True,
            False,
        ),
        (
            [
                "--model_dir",
                "/fake/model",
                "--out_dir",
                "tmp",
                "--config",
                "fake.yaml",
                "--max_seq_length",
                "64",
            ],
            True,
            False,
        ),
    ],
)
def test_cli_arg_combinations(
    monkeypatch, tmp_path, cli_args, expect_metrics, expect_plots
):
    evaluate_mod = importlib.import_module("fnd.eval.evaluate")
    # Patch config, dataset, model, os, open, Trainer, compute_metrics, argparse
    from fnd.config import FNDConfig as RealFNDConfig

    monkeypatch.setattr(evaluate_mod, "FNDConfig", lambda: DummyConfig())
    monkeypatch.setattr(
        RealFNDConfig,
        "from_yaml_with_overrides",
        classmethod(lambda cls, path, **kwargs: DummyConfig()),
    )
    monkeypatch.setattr(
        evaluate_mod, "load_dataset", lambda *a, **k: types.SimpleNamespace(test_df=[])
    )
    monkeypatch.setattr(
        evaluate_mod,
        "Dataset",
        types.SimpleNamespace(from_pandas=lambda df: FakeDataset()),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "load_model_and_tokenizer_from_dir",
        lambda d: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "os",
        types.SimpleNamespace(makedirs=lambda *a, **k: None, path=os.path),
    )
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setattr(evaluate_mod, "Trainer", DummyTrainer)
    monkeypatch.setattr(evaluate_mod, "compute_metrics", lambda x: {"accuracy": 1.0})

    # Patch argparse to simulate CLI args
    class FakeArgumentParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            class Args:
                def __init__(self):
                    self.config = None
                    self.model_dir = "/fake/model"
                    self.out_dir = str(tmp_path)
                    self.data_dataset = None
                    self.paths_data_dir = None
                    self.max_seq_length = None

            return Args()

    monkeypatch.setattr(
        evaluate_mod,
        "argparse",
        types.SimpleNamespace(ArgumentParser=FakeArgumentParser),
    )
    evaluate_mod.main()


def test_cli_with_plots(monkeypatch, tmp_path):
    evaluate_mod = importlib.import_module("fnd.eval.evaluate")
    from fnd.config import FNDConfig as RealFNDConfig

    monkeypatch.setattr(evaluate_mod, "FNDConfig", lambda: PlotConfig())
    monkeypatch.setattr(
        RealFNDConfig,
        "from_yaml_with_overrides",
        classmethod(lambda cls, path, **kwargs: PlotConfig()),
    )
    monkeypatch.setattr(
        evaluate_mod, "load_dataset", lambda *a, **k: types.SimpleNamespace(test_df=[])
    )
    monkeypatch.setattr(
        evaluate_mod,
        "Dataset",
        types.SimpleNamespace(from_pandas=lambda df: FakeDataset()),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "load_model_and_tokenizer_from_dir",
        lambda d: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "os",
        types.SimpleNamespace(makedirs=lambda *a, **k: None, path=os.path),
    )
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setattr(evaluate_mod, "Trainer", DummyTrainerWithPlots)
    monkeypatch.setattr(evaluate_mod, "compute_metrics", lambda x: {"accuracy": 1.0})

    class FakeArgumentParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            class Args:
                def __init__(self):
                    self.config = None
                    self.model_dir = "/fake/model"
                    self.out_dir = str(tmp_path)
                    self.data_dataset = None
                    self.paths_data_dir = None
                    self.max_seq_length = None

            return Args()

    monkeypatch.setattr(
        evaluate_mod,
        "argparse",
        types.SimpleNamespace(ArgumentParser=FakeArgumentParser),
    )
    evaluate_mod.main()


def test_out_dir_permission_error(monkeypatch, tmp_path):
    evaluate_mod = importlib.import_module("fnd.eval.evaluate")
    from fnd.config import FNDConfig as RealFNDConfig

    monkeypatch.setattr(evaluate_mod, "FNDConfig", lambda: DummyConfig())
    monkeypatch.setattr(
        RealFNDConfig,
        "from_yaml_with_overrides",
        classmethod(lambda cls, path, **kwargs: DummyConfig()),
    )
    monkeypatch.setattr(
        evaluate_mod, "load_dataset", lambda *a, **k: types.SimpleNamespace(test_df=[])
    )
    monkeypatch.setattr(
        evaluate_mod,
        "Dataset",
        types.SimpleNamespace(from_pandas=lambda df: FakeDataset()),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "load_model_and_tokenizer_from_dir",
        lambda d: (MagicMock(), MagicMock()),
    )

    def raise_makedirs(*a, **k):
        raise PermissionError("Cannot create out_dir")

    monkeypatch.setattr(
        evaluate_mod, "os", types.SimpleNamespace(makedirs=raise_makedirs, path=os.path)
    )
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setattr(evaluate_mod, "Trainer", DummyTrainer)
    monkeypatch.setattr(evaluate_mod, "compute_metrics", lambda x: {"accuracy": 1.0})

    class FakeArgumentParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            class Args:
                def __init__(self):
                    self.config = None
                    self.model_dir = "/fake/model"
                    self.out_dir = "/root/forbidden"
                    self.data_dataset = None
                    self.paths_data_dir = None
                    self.max_seq_length = None

            return Args()

    monkeypatch.setattr(
        evaluate_mod,
        "argparse",
        types.SimpleNamespace(ArgumentParser=FakeArgumentParser),
    )
    with pytest.raises(PermissionError):
        evaluate_mod.main()


def test_empty_test_set(monkeypatch, tmp_path):
    evaluate_mod = importlib.import_module("fnd.eval.evaluate")
    from fnd.config import FNDConfig as RealFNDConfig

    monkeypatch.setattr(evaluate_mod, "FNDConfig", lambda: DummyConfig())
    monkeypatch.setattr(
        RealFNDConfig,
        "from_yaml_with_overrides",
        classmethod(lambda cls, path, **kwargs: DummyConfig()),
    )
    monkeypatch.setattr(
        evaluate_mod, "load_dataset", lambda *a, **k: types.SimpleNamespace(test_df=[])
    )
    monkeypatch.setattr(
        evaluate_mod,
        "Dataset",
        types.SimpleNamespace(from_pandas=lambda df: FakeDataset()),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "load_model_and_tokenizer_from_dir",
        lambda d: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "os",
        types.SimpleNamespace(makedirs=lambda *a, **k: None, path=os.path),
    )
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setattr(evaluate_mod, "Trainer", DummyTrainerEmpty)
    monkeypatch.setattr(evaluate_mod, "compute_metrics", lambda x: {"accuracy": 1.0})

    class FakeArgumentParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            class Args:
                def __init__(self):
                    self.config = None
                    self.model_dir = "/fake/model"
                    self.out_dir = str(tmp_path)
                    self.data_dataset = None
                    self.paths_data_dir = None
                    self.max_seq_length = None

            return Args()

    monkeypatch.setattr(
        evaluate_mod,
        "argparse",
        types.SimpleNamespace(ArgumentParser=FakeArgumentParser),
    )
    # Should run without error (should not crash on empty test set)
    evaluate_mod.main()
