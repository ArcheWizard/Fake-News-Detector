"""Shared fixtures and mocks for tests."""

import pytest
from unittest.mock import Mock


class DummyConfig:
    def __init__(self):
        self.data = Mock(dataset="kaggle_fake_real", val_size=0.1, test_size=0.1)
        self.paths = Mock(data_dir="/tmp")
        self.seed = 42
        self.max_seq_length = 128
        self.eval = Mock(save_plots=False)


class DummyArgs:
    def __init__(self, tmp_path):
        self.config = None
        self.model_dir = "/fake/model"
        self.out_dir = str(tmp_path)
        self.data_dataset = None
        self.paths_data_dir = None
        self.max_seq_length = None


@pytest.fixture
def dummy_config():
    class DummyConfig:
        def __init__(self):
            self.data = Mock(dataset="kaggle_fake_real", val_size=0.1, test_size=0.1)
            self.paths = Mock(data_dir="/tmp")
            self.seed = 42
            self.max_seq_length = 128
            self.eval = Mock(save_plots=False)

    return DummyConfig()


@pytest.fixture
def fake_model_and_tokenizer():
    return (Mock(name="FakeModel"), Mock(name="FakeTokenizer"))


@pytest.fixture
def dummy_args(tmp_path):
    class DummyArgs:
        config = None
        model_dir = "/fake/model"
        out_dir = str(tmp_path)
        data_dataset = None
        paths_data_dir = None
        max_seq_length = None

    return DummyArgs()
