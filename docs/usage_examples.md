# Usage Examples

This document provides examples of how to use the refactored Fake News Detector commands with the new configuration system.

## Training

### Basic Training (using config file)

```bash
python -m fnd.training.train \
 --config config/config.yaml \
 --run_name my-experiment
```

### Training with CLI Overrides

Override specific configuration values from the command line:

```bash
python -m fnd.training.train \
 --config config/config.yaml \
 --run_name roberta-large-experiment \
 --model_name roberta-large \
 --train_epochs 5 \
 --train_batch_size 32 \
 --train_learning_rate 3e-5 \
 --data_max_samples 10000
```

### Override Data Paths

```bash
python -m fnd.training.train \
 --config config/config.yaml \
 --run_name quick-test \
 --paths_data_dir data/processed/kaggle_fake_real \
 --data_max_samples 1000 \
 --train_epochs 1
```

### Use Profiles (presets)

Apply a profile overlay from `config/profiles/` to quickly adapt settings:

```bash
# Fast iteration (1 epoch, shorter seq length)
python -m fnd.training.train \
 --config config/config.yaml \
 --profile fast \
 --run_name roberta-fast \
 --paths_data_dir data/processed/kaggle_fake_real

# Memory-friendly (small batch, checkpointing)
python -m fnd.training.train \
 --config config/config.yaml \
 --profile memory \
 --run_name roberta-mem \
 --paths_data_dir data/processed/kaggle_fake_real

# Distilled backbone (distilbert)
python -m fnd.training.train \
 --config config/config.yaml \
 --profile distil \
 --run_name distil-fast \
 --paths_data_dir data/processed/kaggle_fake_real
```

CLI flags still override any profile values.

## Evaluation

### Basic Evaluation (with config)

```bash
python -m fnd.eval.evaluate \
 --config config/config.yaml \
 --model_dir runs/my-experiment/model \
 --out_dir runs/my-experiment/evaluation
```

### Evaluation without Config (backward compatible)

```bash
python -m fnd.eval.evaluate \
 --model_dir runs/roberta-kfr/model \
 --out_dir runs/roberta-kfr \
 --data_dataset kaggle_fake_real \
 --paths_data_dir data/processed/kaggle_fake_real
```

### Disable Plot Saving

Edit your config.yaml or create a custom one:

```yaml
eval:
 save_plots: false # Skip confusion matrix and ROC curve generation
```

## Web Application

The web app now uses centralized model utilities for consistent loading:

```bash
streamlit run src/fnd/web/app.py -- \
 --model_dir runs/my-experiment/model \
 --samples_file data/test/test_samples.json
```

Enable optional explainers (SHAP/LIME) via the sidebar toggles. Install `shap` and `lime` to activate.

## Optimization Utilities

Post-training speed-ups for CPU inference:

```bash
# Quantize (dynamic) to qint8
python -m fnd.models.optimization --model_dir runs/my-experiment/model --out_dir runs/my-experiment/model-quant --mode quantize

# Prune Linear layers by 20%
python -m fnd.models.optimization --model_dir runs/my-experiment/model --out_dir runs/my-experiment/model-pruned --mode prune --amount 0.2
```

## Experiment Tracking

The project supports optional experiment tracking through HuggingFace Trainer's built-in integrations. This allows you to automatically log metrics, hyperparameters, and model artifacts to popular tracking platforms.

### Supported Platforms

- **Weights & Biases (wandb)**: Most comprehensive tracking with visualization
- **MLflow**: Enterprise-grade tracking and model registry
- **TensorBoard**: Simple visualization of metrics
- **Azure ML**: Microsoft Azure integration

### Setup

**Step 1: Install tracking library** (choose one or more):

```bash
# For Weights & Biases
pip install wandb
# or use: pip install -r requirements-tracking.txt

# For MLflow
pip install mlflow

# For TensorBoard
pip install tensorboard
```

**Step 2: Authenticate** (if required):

```bash
# Weights & Biases
wandb login

# MLflow (if using remote server)
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

**Step 3: Enable tracking via environment variable**:

```bash
# Single platform
export FND_REPORT_TO=wandb

# Multiple platforms (comma-separated)
export FND_REPORT_TO=wandb,mlflow,tensorboard

# Then run training as usual
python -m fnd.training.train --config config/config.yaml --run_name my-exp
```

### What Gets Tracked

The integration automatically logs:

- **Metrics**: Training/validation loss, accuracy, F1, precision, recall, ROC-AUC
- **Hyperparameters**: Learning rate, batch size, epochs, model name, etc.
- **System Info**: GPU usage, CPU usage, memory
- **Model Checkpoints**: Best model based on validation metrics
- **Configuration**: Full YAML config saved with run

### Example: Training with W&B

```bash
# Set environment variable
export FND_REPORT_TO=wandb

# Optional: Set W&B project name
export WANDB_PROJECT=fake-news-detection

# Train with tracking
python -m fnd.training.train \
 --config config/config.yaml \
 --run_name roberta-wandb-exp \
 --train_epochs 3 \
 --paths_data_dir data/processed/kaggle_fake_real

# View results at https://wandb.ai/<username>/fake-news-detection
```

### Example: Training with MLflow

```bash
# Set environment variable
export FND_REPORT_TO=mlflow

# Optional: Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Train with tracking
python -m fnd.training.train \
 --config config/config.yaml \
 --run_name roberta-mlflow-exp \
 --train_epochs 3 \
 --paths_data_dir data/processed/kaggle_fake_real

# View results in MLflow UI (if running locally)
# mlflow ui --backend-store-uri ./mlruns
```

### Disabling Tracking

To disable tracking completely:

```bash
# Unset the environment variable
unset FND_REPORT_TO

# Or explicitly disable
export FND_REPORT_TO=none
```

### Viewing Results

- **W&B**: Visit [wandb.ai](https://wandb.ai) and navigate to your project
- **MLflow**: Run `mlflow ui` and open <http://localhost:5000>
- **TensorBoard**: Run `tensorboard --logdir runs/` and open <http://localhost:6006>

### Notes

- Tracking is completely optional; the project works fine without it
- Environment variables are checked at training time
- Tracking libraries are NOT included in base requirements.txt (install separately)
- Configuration and metrics are always saved to disk regardless of tracking

## Multilingual Models

Use a multilingual backbone by setting `model_name` in your config, e.g.:

```yaml
model_name: bert-base-multilingual-cased
```

No additional code changes required.

## API Server

The FastAPI server now uses the same centralized utilities:

```bash
# Set environment variable
export MODEL_DIR=runs/my-experiment/model

# Start server
uvicorn fnd.api.main:app --host 0.0.0.0 --port 8000
```

Or provide model_dir in each request:

```bash
curl -X POST "http://localhost:8000/predict" \
 -H "Content-Type: application/json" \
 -d '{
 "text": "Breaking news: Scientists discover...",
 "model_dir": "runs/my-experiment/model"
 }'
```

## Configuration File Structure

The `config/config.yaml` file supports these sections:

```yaml
# Top-level settings
seed: 42
model_name: roberta-base
max_seq_length: 256

# Training configuration
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

# Evaluation configuration
eval:
 metrics: [accuracy, f1, precision, recall, roc_auc]
 batch_size: 32
 save_plots: true

# Data configuration
data:
 dataset: kaggle_fake_real
 text_field: text
 label_field: label
 val_size: 0.1
 test_size: 0.1
 max_samples: null # null = use all data
 shuffle: true

# Path configuration
paths:
 data_dir: data/processed/kaggle_fake_real
 runs_dir: runs
 models_dir: models
```

## CLI Override Notation

Use underscore notation to override nested config values:

- `--seed 123` → overrides top-level `seed`
- `--train_epochs 5` → overrides `train.epochs`
- `--train_batch_size 32` → overrides `train.batch_size`
- `--data_val_size 0.15` → overrides `data.val_size`
- `--paths_data_dir /path/to/data` → overrides `paths.data_dir`

## Benefits of the New System

1. **Reproducibility**: Save configuration alongside models
2. **Flexibility**: Override any setting via CLI without editing files
3. **Organization**: Group related settings logically
4. **Validation**: Automatic validation of all configuration values
5. **Consistency**: Same model loading code across web app, API, and scripts
6. **Documentation**: Self-documenting configuration files
