import os

from fnd.data.datasets import load_dataset
from fnd.exceptions import DataLoadError
import pytest


def test_loader_raises_for_missing_dir(tmp_path):
    with pytest.raises(DataLoadError):
        load_dataset("kaggle_fake_real", str(tmp_path / "does_not_exist"))
