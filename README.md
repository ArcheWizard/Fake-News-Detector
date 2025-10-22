# Fake News Detector (Transformers)

Detect misinformation using Transformer-based NLP models (BERT/RoBERTa). Includes training, evaluation, explainability (SHAP/LIME), and a lightweight web UI and API.

## Features

- Datasets: FakeNewsNet, LIAR, Kaggle Fake News (binary and multi-class).
- Models: Fine-tune `bert-base-uncased` or `roberta-base` (HuggingFace Transformers).
- Explainability: Highlight influential tokens with SHAP/LIME.
- Evaluation: F1-score, confusion matrix, ROC curve, PR-AUC.
- Serving: Streamlit app for interactive input and FastAPI for programmatic access.
- Optional: Multilingual support (mBERT), simple web crawler, Dockerized API.

## Tech Stack

Python, PyTorch, HuggingFace Transformers/Datasets, scikit-learn, Streamlit, SHAP, LIME, FastAPI.

## Datasets

- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [Kaggle Fake News](https://www.kaggle.com/c/fake-news)

Notes:

- Some datasets require manual download/accepting terms.
- We’ll add dataset-specific preparation scripts (`src/fnd/data/prepare.py`).

## Setup

Requirements: Python 3.10+, Linux. GPU optional (CUDA recommended for speed).

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

The project now uses a YAML-based configuration system with optional CLI overrides for maximum flexibility.

```bash
# 0) (once) Install project package path for `python -m fnd` commands
pip install -e .

# 1) Prepare data (Kaggle Fake/Real). Place True.csv and Fake.csv under data/raw/kaggle_fake_real/
python -m fnd.data.prepare \
  --dataset kaggle_fake_real \
  --in_dir data/raw/kaggle_fake_real \
  --out_dir data/processed/kaggle_fake_real

# 2) Train (using config file with optional overrides)
python -m fnd.training.train \
  --config config/config.yaml \
  --run_name my-experiment

# Or with CLI overrides:
python -m fnd.training.train \
  --config config/config.yaml \
  --run_name roberta-kfr \
  --model_name roberta-base \
  --train_epochs 3 \
  --train_batch_size 16 \
  --paths_data_dir data/processed/kaggle_fake_real

# 3) Evaluate (with config)
python -m fnd.eval.evaluate \
  --config config/config.yaml \
  --model_dir runs/my-experiment/model \
  --out_dir runs/my-experiment

# Or without config (backward compatible):
python -m fnd.eval.evaluate \
  --model_dir runs/roberta-kfr/model \
  --out_dir runs/roberta-kfr \
  --data_dataset kaggle_fake_real \
  --paths_data_dir data/processed/kaggle_fake_real

# 4) Extract test samples for manual testing (optional but recommended)
python scripts/extract_test_samples.py \
  --data_dir data/processed/kaggle_fake_real \
  --num_samples 20 \
  --out_file data/test/test_samples.json

# 5) Web app (Streamlit) - with test samples
streamlit run src/fnd/web/app.py -- \
  --model_dir runs/my-experiment/model \
  --samples_file data/test/test_samples.json

# 6) API (FastAPI)
export MODEL_DIR=runs/my-experiment/model
uvicorn fnd.api.main:app --reload --port 8000
```

See [docs/usage_examples.md](docs/usage_examples.md) for more examples and [docs/migration_guide.md](docs/migration_guide.md) for migrating from the old CLI.

## Configuration

Configuration is managed through YAML files with support for CLI overrides. The default configuration is in `config/config.yaml`.

### Configuration Structure

```yaml
# Top-level settings
seed: 42
model_name: roberta-base
max_seq_length: 256

# Training hyperparameters
train:
  epochs: 3
  batch_size: 16
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  fp16: false
  save_strategy: epoch
  evaluation_strategy: epoch
  logging_steps: 100
  load_best_model_at_end: true
  metric_for_best_model: f1

# Evaluation settings
eval:
  metrics: [accuracy, f1, precision, recall, roc_auc]
  batch_size: 32
  save_plots: true

# Data settings
data:
  dataset: kaggle_fake_real
  text_field: text
  label_field: label   # 1 = fake, 0 = real
  val_size: 0.1
  test_size: 0.1
  max_samples: null    # null = use all data
  shuffle: true

# Paths
paths:
  data_dir: data/processed/kaggle_fake_real
  runs_dir: runs
  models_dir: models
```

### CLI Overrides

Use underscore notation to override nested values:

```bash
# Override top-level settings
--seed 123 --model_name bert-base-uncased

# Override training settings
--train_epochs 5 --train_batch_size 32 --train_learning_rate 3e-5

# Override data settings
--data_max_samples 10000 --data_val_size 0.15

# Override paths
--paths_data_dir /path/to/data
```

### Benefits

- **Reproducibility**: Configuration saved with each training run
- **Flexibility**: Override any setting without editing files
- **Organization**: Logical grouping of related settings
- **Validation**: Automatic validation of all values
- **Documentation**: Self-documenting configuration files

## Evaluation

- Primary: F1-score (macro/weighted).
- Secondary: Confusion matrix, ROC-AUC, PR-AUC.
- Use held-out test split; optionally stratified k-fold.

For more rigorous planning of metrics/datasets later, we can use an evaluation planner to define goals and datasets before coding.

## Project Structure

Planned layout (files will be added incrementally):

```text
Fake-News-Detector/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ config/
│  └─ config.yaml
├─ data/                 # .gitignored
│  ├─ raw/
│  ├─ processed/
│  └─ external/
├─ models/               # saved HF checkpoints (gitignored)
├─ runs/                 # training logs, metrics, artifacts (gitignored)
├─ src/
│  └─ fnd/
│     ├─ __init__.py
│     ├─ data/
│     │  ├─ prepare.py          # download/clean/split
│     │  └─ datasets.py         # HF Dataset loading wrappers
│     ├─ models/
│     │  ├─ factory.py          # load model/tokenizer by name
│     │  └─ heads.py            # classification heads if needed
│     ├─ training/
│     │  ├─ train.py            # Trainer loop, logging
│     │  └─ metrics.py
│     ├─ eval/
│     │  └─ evaluate.py         # confusion matrix, ROC
│     ├─ explain/
│     │  ├─ shap_explain.py
│     │  └─ lime_explain.py
│     ├─ web/
│     │  └─ app.py              # Streamlit UI
│     └─ api/
│        └─ main.py             # FastAPI app
├─ tests/
│  ├─ test_data.py
│  ├─ test_models.py
│  └─ test_training.py
└─ docs/
   ├─ datasets.md
   ├─ evaluation.md
   └─ architecture.md
```

## Roadmap

- ✅ v0.1: Single-language (en) binary classifier, Streamlit UI, FastAPI
- ✅ v0.2: YAML configuration system, centralized model utilities, comprehensive testing
- 🔄 v0.3: CI/CD pipeline, improved documentation
- 📋 v1.0: Multilingual support (mBERT), enhanced UI with explainability
- 📋 v2.0: Docker deployment, experiment tracking, model optimization

## Testing

The project includes comprehensive test coverage (90%+):

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/fnd --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py -v
```

Test categories:

- Data loading and preprocessing (15 tests)
- Metrics computation (17 tests)
- Configuration management (30 tests)
- Model imports and utilities

## Documentation

- [Usage Examples](docs/usage_examples.md) - Command-line examples for all components
- [Migration Guide](docs/migration_guide.md) - Guide for migrating from old CLI
- [Testing Guide](docs/testing_guide.md) - How to test models with the web app
- [Implementation Tracker](docs/implementation_tracker.md) - Development progress

## New Features

- ✅ **Configuration Management**: YAML-based config with CLI overrides
- ✅ **Error Handling**: Custom exception hierarchy for clear error messages
- ✅ **Model Utilities**: Centralized model loading and pipeline creation
- ✅ **Testing**: Comprehensive test suite with 90%+ coverage
- ✅ **Documentation**: Google-style docstrings throughout
- ✅ **Validation**: Automatic validation of all configuration values
- 🔄 **CI/CD**: GitHub Actions for automated testing
- 📋 **Explainability**: SHAP/LIME integration (planned)
- 📋 **Multilingual**: mBERT support (planned)

## License

MIT (to be added).

## Acknowledgements

Built with HuggingFace Transformers, Datasets, PyTorch, scikit-learn, SHAP, LIME, Streamlit, FastAPI.
