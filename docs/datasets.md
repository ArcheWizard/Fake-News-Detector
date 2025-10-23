# Datasets

This project currently supports the Kaggle Fake and Real News dataset. Two layouts are recognized:

1. Processed CSV (preferred)

- Directory contains `dataset.csv` with columns `text` and `label` (0=real, 1=fake)
- Example: `data/processed/kaggle_fake_real/dataset.csv`

1. Raw Kaggle CSVs

- Directory contains `True.csv` and `Fake.csv` as provided by Kaggle
- Example: `data/raw/kaggle_fake_real/True.csv`, `data/raw/kaggle_fake_real/Fake.csv`

## Preparing the processed dataset

Normalize the raw files into a single processed CSV via the prepare script:

```bash
mkdir -p data/raw/kaggle_fake_real data/processed/kaggle_fake_real
# Place raw files:
# data/raw/kaggle_fake_real/True.csv
# data/raw/kaggle_fake_real/Fake.csv

python -m fnd.data.prepare \
 --dataset kaggle_fake_real \
 --in_dir data/raw/kaggle_fake_real \
 --out_dir data/processed/kaggle_fake_real
```

Then point your config (`paths.data_dir`) or CLI (`--paths_data_dir`) to `data/processed/kaggle_fake_real`.

## Training with the processed dataset

```bash
python -m fnd.training.train \
 --config config/config.yaml \
 --run_name my-experiment \
 --paths_data_dir data/processed/kaggle_fake_real
```

## Common errors

- DataLoadError: Missing required CSV files
- Ensure your `paths.data_dir` points to a directory that contains either:
- `dataset.csv` (processed), or
- `True.csv` and `Fake.csv` (raw) and you have run the prepare step
