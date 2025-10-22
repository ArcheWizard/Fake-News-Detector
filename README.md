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

These commands will be available as the code is scaffolded. Paths and options are configurable.

```bash
# 0) (once) Install project package path for `python -m fnd` commands
pip install -e .

# 1) Prepare data (Kaggle Fake/Real). Place True.csv and Fake.csv under data/raw/kaggle_fake_real/
#    If you already have them, the prepare script will standardize columns.
python -m fnd.data.prepare \
  --dataset kaggle_fake_real \
  --in_dir data/raw/kaggle_fake_real \
  --out_dir data/processed/kaggle_fake_real

# 2) Train (binary: label 1 = fake, 0 = real)
python -m fnd.training.train \
  --model roberta-base \
  --dataset kaggle_fake_real \
  --data_dir data/processed/kaggle_fake_real \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --out_dir runs/roberta-kfr

# 3) Evaluate
python -m fnd.eval.evaluate \
  --dataset kaggle_fake_real \
  --data_dir data/processed/kaggle_fake_real \
  --model_dir runs/roberta-kfr/model \
  --out_dir runs/roberta-kfr

# 4) Extract test samples for manual testing (optional but recommended)
python scripts/extract_test_samples.py \
  --data_dir data/processed/kaggle_fake_real \
  --num_samples 20 \
  --out_file test_samples.json

# 5) Web app (Streamlit) - with test samples
streamlit run src/fnd/web/app.py -- \
  --model_dir runs/roberta-kfr/model \
  --samples_file test_samples.json

# 6) Web app (Streamlit) - without test samples
streamlit run src/fnd/web/app.py -- \
  --model_dir runs/roberta-kfr/model

# 7) API (FastAPI)
uvicorn fnd.api.main:app --reload --port 8000
```

## Configuration

Central config lives in `config/config.yaml` (model, tokenizer, training/eval params, paths). Override via CLI flags or env vars.

Example defaults (to be created):

```yaml
seed: 42
model_name: roberta-base
max_seq_length: 256
train:
  epochs: 3
  batch_size: 16
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
eval:
  metrics: ["f1", "precision", "recall", "roc_auc"]
data:
  dataset: kaggle_fake_real
  text_field: "text"
  label_field: "label"   # 1 = fake, 0 = real
  val_size: 0.1
  test_size: 0.1
paths:
  data_dir: "data"
  runs_dir: "runs"
  models_dir: "models"
```

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

- v0: Single-language (en) binary classifier, Streamlit UI, FastAPI.
- v1: Multilingual (mBERT), improved crawler integration.
- v2: Docker + CI/CD; experiment tracking.

## License

MIT (to be added).

## Acknowledgements

Built with HuggingFace Transformers, Datasets, PyTorch, scikit-learn, SHAP, LIME, Streamlit, FastAPI.
