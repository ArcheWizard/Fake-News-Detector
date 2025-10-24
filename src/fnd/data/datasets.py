from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from fnd.exceptions import DataLoadError, InsufficientDataError

LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def _read_kfr_raw_dir(in_dir: str) -> pd.DataFrame:
    """Read and normalize Kaggle Fake/Real News raw CSV files.

    Reads True.csv and Fake.csv from the specified directory, combines title
    and text fields, assigns appropriate labels, and filters out empty entries.

    Args:
        in_dir: Path to directory containing True.csv and Fake.csv

    Returns:
        DataFrame with columns:
            - text: Combined title + "\\n\\n" + article text (whitespace trimmed)
            - label: Integer label (0=real, 1=fake)

    Raises:
        DataLoadError: If True.csv or Fake.csv are missing from in_dir

    Note:
        - Expected columns in raw CSVs: title, text, subject, date
        - Only title and text columns are used
        - Missing values are filled with empty strings
        - Entries with empty text (after combining) are filtered out
    """
    true_csv = os.path.join(in_dir, "True.csv")
    fake_csv = os.path.join(in_dir, "Fake.csv")
    if not (os.path.isfile(true_csv) and os.path.isfile(fake_csv)):
        raise DataLoadError(
            f"Missing required CSV files in {in_dir}\n"
            f"Expected: True.csv and Fake.csv\n"
            f"Found: True.csv={os.path.isfile(true_csv)}, Fake.csv={os.path.isfile(fake_csv)}\n\n"
            f"Please download the Kaggle Fake and Real News dataset and place the files in this directory."
        )

    df_true_raw = pd.read_csv(true_csv)
    df_fake_raw = pd.read_csv(fake_csv)

    def normalize(df_raw: pd.DataFrame, label_value: int) -> pd.DataFrame:
        title_series = (
            df_raw["title"].fillna("")
            if "title" in df_raw.columns
            else pd.Series([""] * len(df_raw))
        )
        text_series = (
            df_raw["text"].fillna("")
            if "text" in df_raw.columns
            else pd.Series([""] * len(df_raw))
        )
        combined = (
            title_series.astype(str) + "\n\n" + text_series.astype(str)
        ).str.strip()
        return pd.DataFrame({"text": combined, "label": label_value})

    df_true = normalize(df_true_raw, 0)
    df_fake = normalize(df_fake_raw, 1)

    df = pd.concat([df_true, df_fake], ignore_index=True)
    # Drop empties
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    return df


def _read_processed_csv(in_dir: str) -> Optional[pd.DataFrame]:
    """Attempt to read a pre-processed dataset.csv file.

    Args:
        in_dir: Path to directory that may contain dataset.csv

    Returns:
        DataFrame with [text, label] columns if dataset.csv exists and is valid,
        otherwise None to indicate raw files should be read instead.

    Note:
        - Looks for file named "dataset.csv" in in_dir
        - Validates that required columns [text, label] are present
        - Returns only the text and label columns even if others exist
    """
    processed_csv = os.path.join(in_dir, "dataset.csv")
    if os.path.isfile(processed_csv):
        df = pd.read_csv(processed_csv)
        # Ensure required columns
        if {"text", "label"}.issubset(df.columns):
            return df[["text", "label"]].copy()
    return None


def load_kaggle_fake_real(
    data_dir: str,
    *,
    seed: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.1,
    max_samples: Optional[int] = None,
) -> DatasetBundle:
    """Load Kaggle Fake/Real News dataset with stratified train/validation/test splits.

    This function handles both raw CSV files (True.csv, Fake.csv) and pre-processed
    dataset.csv files. It performs stratified splitting to maintain class balance
    across train/validation/test sets and includes comprehensive validation checks.

    Args:
        data_dir: Path to directory containing either:
            - Processed file: dataset.csv with columns [text, label]
            - Raw files: True.csv and Fake.csv with columns [title, text, subject, date]
        seed: Random seed for reproducible splits. Default: 42
        val_size: Fraction of data for validation set (0.0-1.0). Default: 0.1
        test_size: Fraction of data for test set (0.0-1.0). Default: 0.1
        max_samples: Optional limit on total samples to load. If None, loads all.
            Useful for quick experiments or testing. Default: None

    Returns:
        DatasetBundle containing:
            - train_df: Training DataFrame with columns [text, label]
            - validation_df: Validation DataFrame with columns [text, label]
            - test_df: Test DataFrame with columns [text, label]
            - label2id: Dict mapping {"real": 0, "fake": 1}
            - id2label: Dict mapping {0: "real", 1: "fake"}

    Raises:
        DataLoadError: If data_dir doesn't exist, required files are missing,
            or data files are malformed/corrupted
        InsufficientDataError: If:
            - Dataset is empty after loading
            - Dataset has fewer than 100 total samples
            - Dataset has fewer than 10 samples per class
            - Dataset has only one class present
            - Dataset is too small for requested split sizes
            - Stratification fails due to class imbalance

    Examples:
        Basic usage with default splits (80/10/10):
        >>> bundle = load_kaggle_fake_real("data/processed/kaggle_fake_real")
        >>> print(f"Train: {len(bundle.train_df)}, Val: {len(bundle.validation_df)}")
        Train: 36000, Val: 4000

        Custom split sizes:
        >>> bundle = load_kaggle_fake_real(
        ...     "data/processed/kaggle_fake_real",
        ...     val_size=0.15,
        ...     test_size=0.15,
        ...     seed=123
        ... )

        Quick testing with limited samples:
        >>> bundle = load_kaggle_fake_real(
        ...     "data/processed/kaggle_fake_real",
        ...     max_samples=1000
        ... )

    Note:
        - Labels are binary: 0 = real news, 1 = fake news
        - For raw files, text combines: title + "\\n\\n" + article body
        - Empty text entries are automatically filtered out
        - Splits are stratified to maintain ~50/50 class balance
        - Minimum 100 total samples and 10 per class required
    """
    if not os.path.isdir(data_dir):
        raise DataLoadError(
            f"Data directory not found: {data_dir}\n"
            f"Please run: python -m fnd.data.prepare --dataset kaggle_fake_real "
            f"--in_dir <raw_data_dir> --out_dir {data_dir}"
        )

    df = _read_processed_csv(data_dir)
    if df is None:
        df = _read_kfr_raw_dir(data_dir)

    if len(df) == 0:
        raise InsufficientDataError(
            f"Dataset is empty after loading from {data_dir}\n"
            f"Please check that your data files contain valid entries."
        )

    if len(df) < 100:
        raise InsufficientDataError(
            f"Dataset too small: {len(df)} samples (minimum: 100)\n"
            f"Please provide more data for reliable train/validation/test splits."
        )

    if max_samples is not None and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(
            drop=True
        )

    # Validate split sizes
    min_samples_needed = int(len(df) * (val_size + test_size)) + 10
    if len(df) < min_samples_needed:
        raise InsufficientDataError(
            f"Dataset ({len(df)} samples) too small for requested split sizes\n"
            f"Minimum needed: {min_samples_needed} samples\n"
            f"Current split configuration: val_size={val_size}, test_size={test_size}\n"
            f"Try reducing split sizes or providing more data."
        )

    # Check class balance
    label_counts = df["label"].value_counts()
    if len(label_counts) < 2:
        raise InsufficientDataError(
            f"Dataset contains only one class: {label_counts.to_dict()}\n"
            f"Binary classification requires samples from both classes (0=real, 1=fake)."
        )

    min_class_count = label_counts.min()
    if min_class_count < 10:
        raise InsufficientDataError(
            f"Insufficient samples for minority class: {label_counts.to_dict()}\n"
            f"Need at least 10 samples per class for reliable stratified splitting."
        )

    # First split off test, then split train into train/val to maintain stratification
    try:
        df_trainval, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["label"],
            shuffle=True,
        )
        relative_val = val_size / (1.0 - test_size)
        df_train, df_val = train_test_split(
            df_trainval,
            test_size=relative_val,
            random_state=seed,
            stratify=df_trainval["label"],
            shuffle=True,
        )
    except ValueError as e:
        raise InsufficientDataError(
            f"Failed to create stratified splits: {str(e)}\n"
            f"Dataset size: {len(df)}, Label distribution: {label_counts.to_dict()}\n"
            f"Try adjusting split sizes or providing more balanced data."
        ) from e

    return DatasetBundle(
        train_df=df_train[["text", "label"]].reset_index(drop=True),
        validation_df=df_val[["text", "label"]].reset_index(drop=True),
        test_df=df_test[["text", "label"]].reset_index(drop=True),
        label2id=LABEL2ID.copy(),
        id2label=ID2LABEL.copy(),
    )


def load_dataset(
    dataset: str,
    data_dir: str,
    *,
    seed: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.1,
    max_samples: Optional[int] = None,
) -> DatasetBundle:
    """Load a fake news detection dataset by name with automatic dispatcher.

    This is a convenience wrapper that routes to the appropriate dataset loader
    based on the dataset name. Currently supports Kaggle Fake/Real News dataset.

    Args:
        dataset: Dataset identifier. Supported values:
            - "kaggle_fake_real", "kfr", "kaggle": Kaggle Fake and Real News
        data_dir: Path to directory containing dataset files
        seed: Random seed for reproducible splits. Default: 42
        val_size: Fraction of data for validation set. Default: 0.1
        test_size: Fraction of data for test set. Default: 0.1
        max_samples: Optional limit on total samples. Default: None (all samples)

    Returns:
        DatasetBundle with train/validation/test splits and label mappings.
        See load_kaggle_fake_real() for detailed structure.

    Raises:
        ValueError: If dataset name is not recognized
        DataLoadError: If data loading fails (see specific loader for details)
        InsufficientDataError: If dataset doesn't meet minimum requirements

    Examples:
        >>> bundle = load_dataset("kaggle_fake_real", "data/processed/kaggle_fake_real")
        >>> bundle = load_dataset("kfr", "data/processed/kaggle_fake_real", max_samples=500)

    Note:
        Future versions will support additional datasets like FakeNewsNet and LIAR.
    """
    dataset = (dataset or "").lower()
    if dataset in {"kaggle_fake_real", "kfr", "kaggle"}:
        return load_kaggle_fake_real(
            data_dir,
            seed=seed,
            val_size=val_size,
            test_size=test_size,
            max_samples=max_samples,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
