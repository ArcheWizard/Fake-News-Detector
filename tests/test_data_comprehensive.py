"""Comprehensive tests for data loading and preprocessing."""

import pandas as pd
import pytest

from fnd.data.datasets import (
    ID2LABEL,
    LABEL2ID,
    DatasetBundle,
    _read_kfr_raw_dir,
    _read_processed_csv,
    load_dataset,
    load_kaggle_fake_real,
)
from fnd.exceptions import DataLoadError


class TestDatasetBundle:
    """Tests for DatasetBundle dataclass."""

    def test_dataset_bundle_structure(self):
        """Test that DatasetBundle has expected attributes."""
        train_df = pd.DataFrame({"text": ["sample"], "label": [0]})
        val_df = pd.DataFrame({"text": ["sample"], "label": [1]})
        test_df = pd.DataFrame({"text": ["sample"], "label": [0]})

        bundle = DatasetBundle(
            train_df=train_df,
            validation_df=val_df,
            test_df=test_df,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )

        assert bundle.train_df is not None
        assert bundle.validation_df is not None
        assert bundle.test_df is not None
        assert bundle.label2id == {"real": 0, "fake": 1}
        assert bundle.id2label == {0: "real", 1: "fake"}


class TestLoadKaggleFakeReal:
    """Tests for loading Kaggle Fake/Real dataset."""

    def test_load_from_raw_csv_files(self, tmp_path):
        """Test loading from raw True.csv and Fake.csv files."""
        # Create mock raw CSV files (need at least 100 samples)
        true_df = pd.DataFrame(
            {
                "title": [f"True Title {i}" for i in range(60)],
                "text": [f"True text {i}" for i in range(60)],
                "subject": ["news"] * 60,
                "date": ["2020-01-01"] * 60,
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": [f"Fake Title {i}" for i in range(60)],
                "text": [f"Fake text {i}" for i in range(60)],
                "subject": ["politics"] * 60,
                "date": ["2020-01-01"] * 60,
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        bundle = load_kaggle_fake_real(str(tmp_path), seed=42)

        assert isinstance(bundle, DatasetBundle)
        assert len(bundle.train_df) > 0
        assert len(bundle.validation_df) > 0
        assert len(bundle.test_df) > 0
        assert bundle.label2id == {"real": 0, "fake": 1}
        assert set(bundle.train_df["label"].unique()).issubset({0, 1})

    def test_load_from_processed_csv(self, tmp_path):
        """Test loading from pre-processed dataset.csv file."""
        # Create mock processed dataset
        df = pd.DataFrame(
            {"text": [f"Sample text {i}" for i in range(100)], "label": [0, 1] * 50}
        )

        df.to_csv(tmp_path / "dataset.csv", index=False)

        bundle = load_kaggle_fake_real(str(tmp_path), seed=42)

        assert isinstance(bundle, DatasetBundle)
        total_samples = (
            len(bundle.train_df) + len(bundle.validation_df) + len(bundle.test_df)
        )
        assert total_samples == 100

    def test_stratified_split_maintains_class_balance(self, tmp_path):
        """Test that splits maintain approximate class balance."""
        # Create balanced dataset
        true_df = pd.DataFrame(
            {
                "title": [f"True {i}" for i in range(50)],
                "text": [f"True text {i}" for i in range(50)],
                "subject": ["news"] * 50,
                "date": ["2020-01-01"] * 50,
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": [f"Fake {i}" for i in range(50)],
                "text": [f"Fake text {i}" for i in range(50)],
                "subject": ["politics"] * 50,
                "date": ["2020-01-01"] * 50,
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        bundle = load_kaggle_fake_real(
            str(tmp_path), seed=42, val_size=0.2, test_size=0.2
        )

        # Check approximate balance in each split
        for df_name, df in [
            ("train", bundle.train_df),
            ("validation", bundle.validation_df),
            ("test", bundle.test_df),
        ]:
            if len(df) > 0:
                real_count = (df["label"] == 0).sum()
                fake_count = (df["label"] == 1).sum()
                if fake_count > 0:
                    ratio = real_count / fake_count
                    assert (
                        0.5 < ratio < 2.0
                    ), f"{df_name} split has severe class imbalance: {ratio}"

    def test_max_samples_limits_dataset_size(self, tmp_path):
        """Test that max_samples parameter correctly limits dataset size."""
        # Create larger dataset
        true_df = pd.DataFrame(
            {
                "title": [f"True {i}" for i in range(100)],
                "text": [f"True text {i}" for i in range(100)],
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": [f"Fake {i}" for i in range(100)],
                "text": [f"Fake text {i}" for i in range(100)],
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        bundle = load_kaggle_fake_real(str(tmp_path), seed=42, max_samples=50)

        total_samples = (
            len(bundle.train_df) + len(bundle.validation_df) + len(bundle.test_df)
        )
        assert total_samples == 50

    def test_raises_error_for_missing_directory(self, tmp_path):
        """Test that DataLoadError is raised for missing directory."""
        with pytest.raises(DataLoadError):
            load_kaggle_fake_real(str(tmp_path / "does_not_exist"))

    def test_raises_error_for_missing_csv_files(self, tmp_path):
        """Test that DataLoadError is raised when CSV files are missing."""
        # Create directory but no CSV files
        data_dir = tmp_path / "empty_dir"
        data_dir.mkdir()

        with pytest.raises(DataLoadError):
            load_kaggle_fake_real(str(data_dir))

    def test_empty_text_entries_filtered_out(self, tmp_path):
        """Test that entries with empty text are filtered out."""
        # Create enough samples (need 100+), with some empty entries
        true_titles = [f"Title {i}" if i % 10 != 0 else "" for i in range(60)]
        true_texts = [f"Text {i}" if i % 10 != 0 else "" for i in range(60)]
        fake_titles = [f"Fake {i}" if i % 10 != 0 else "" for i in range(60)]
        fake_texts = [f"Fake text {i}" if i % 10 != 0 else "" for i in range(60)]

        true_df = pd.DataFrame(
            {
                "title": true_titles,
                "text": true_texts,
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": fake_titles,
                "text": fake_texts,
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        bundle = load_kaggle_fake_real(str(tmp_path), seed=42)

        # Should have filtered out empty entries (every 10th entry)
        total_samples = (
            len(bundle.train_df) + len(bundle.validation_df) + len(bundle.test_df)
        )
        assert (
            total_samples == 108
        )  # 120 original - 12 empty (every 10th)    def test_reproducible_splits_with_same_seed(self, tmp_path):
        """Test that same seed produces identical splits."""
        true_df = pd.DataFrame(
            {
                "title": [f"True {i}" for i in range(50)],
                "text": [f"True text {i}" for i in range(50)],
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": [f"Fake {i}" for i in range(50)],
                "text": [f"Fake text {i}" for i in range(50)],
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        bundle1 = load_kaggle_fake_real(str(tmp_path), seed=42)
        bundle2 = load_kaggle_fake_real(str(tmp_path), seed=42)

        # Check that splits are identical
        pd.testing.assert_frame_equal(bundle1.train_df, bundle2.train_df)
        pd.testing.assert_frame_equal(bundle1.validation_df, bundle2.validation_df)
        pd.testing.assert_frame_equal(bundle1.test_df, bundle2.test_df)

    def test_different_seeds_produce_different_splits(self, tmp_path):
        """Test that different seeds produce different splits."""
        true_df = pd.DataFrame(
            {
                "title": [f"True {i}" for i in range(50)],
                "text": [f"True text {i}" for i in range(50)],
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": [f"Fake {i}" for i in range(50)],
                "text": [f"Fake text {i}" for i in range(50)],
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        bundle1 = load_kaggle_fake_real(str(tmp_path), seed=42)
        bundle2 = load_kaggle_fake_real(str(tmp_path), seed=123)

        # Check that at least one split is different
        train_equal = bundle1.train_df.equals(bundle2.train_df)
        val_equal = bundle1.validation_df.equals(bundle2.validation_df)
        test_equal = bundle1.test_df.equals(bundle2.test_df)

        assert not (
            train_equal and val_equal and test_equal
        ), "Different seeds should produce different splits"


class TestReadKFRRawDir:
    """Tests for _read_kfr_raw_dir helper function."""

    def test_combines_title_and_text(self, tmp_path):
        """Test that title and text are combined with double newline."""
        true_df = pd.DataFrame(
            {
                "title": ["Title 1"],
                "text": ["Text 1"],
                "subject": ["news"],
                "date": ["2020-01-01"],
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": ["Title 2"],
                "text": ["Text 2"],
                "subject": ["politics"],
                "date": ["2020-01-02"],
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        df = _read_kfr_raw_dir(str(tmp_path))

        assert "Title 1\n\nText 1" in df["text"].values
        assert "Title 2\n\nText 2" in df["text"].values

    def test_assigns_correct_labels(self, tmp_path):
        """Test that True.csv gets label 0 and Fake.csv gets label 1."""
        true_df = pd.DataFrame(
            {
                "title": ["True"],
                "text": ["Text"],
            }
        )
        fake_df = pd.DataFrame(
            {
                "title": ["Fake"],
                "text": ["Text"],
            }
        )

        true_df.to_csv(tmp_path / "True.csv", index=False)
        fake_df.to_csv(tmp_path / "Fake.csv", index=False)

        df = _read_kfr_raw_dir(str(tmp_path))

        # Find the true and fake entries
        true_row = df[df["text"].str.contains("True")].iloc[0]
        fake_row = df[df["text"].str.contains("Fake")].iloc[0]

        assert true_row["label"] == 0
        assert fake_row["label"] == 1


class TestReadProcessedCSV:
    """Tests for _read_processed_csv helper function."""

    def test_reads_existing_processed_csv(self, tmp_path):
        """Test reading an existing dataset.csv file."""
        df = pd.DataFrame({"text": ["Sample 1", "Sample 2"], "label": [0, 1]})

        df.to_csv(tmp_path / "dataset.csv", index=False)

        result = _read_processed_csv(str(tmp_path))

        assert result is not None
        assert len(result) == 2
        assert list(result.columns) == ["text", "label"]

    def test_returns_none_for_missing_csv(self, tmp_path):
        """Test returns None when dataset.csv doesn't exist."""
        result = _read_processed_csv(str(tmp_path))
        assert result is None


class TestLoadDataset:
    """Tests for the generic load_dataset wrapper."""

    def test_loads_kaggle_fake_real_dataset(self, tmp_path):
        """Test that load_dataset correctly dispatches to kaggle_fake_real loader."""
        df = pd.DataFrame(
            {"text": [f"Sample {i}" for i in range(120)], "label": [0, 1] * 60}
        )

        df.to_csv(tmp_path / "dataset.csv", index=False)

        bundle = load_dataset("kaggle_fake_real", str(tmp_path), seed=42)

        assert isinstance(bundle, DatasetBundle)
        assert len(bundle.train_df) > 0

    def test_raises_value_error_for_unknown_dataset(self, tmp_path):
        """Test that ValueError is raised for unknown dataset names."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            load_dataset("unknown_dataset", str(tmp_path))
