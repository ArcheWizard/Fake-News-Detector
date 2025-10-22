# Migration Guide: Old CLI to New Config System

This guide helps migrate from the old command-line argument system to the new configuration-based system.

## Training Script Migration

### Old Command Format

```bash
python -m fnd.training.train \
  --model roberta-base \
  --dataset kaggle_fake_real \
  --data_dir data/processed/kaggle_fake_real \
  --out_dir runs/my-run \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --max_seq_length 256 \
  --seed 42 \
  --max_samples 10000
```

### New Command Format

```bash
python -m fnd.training.train \
  --config config/config.yaml \
  --run_name my-run \
  --model_name roberta-base \
  --data_dataset kaggle_fake_real \
  --paths_data_dir data/processed/kaggle_fake_real \
  --train_epochs 3 \
  --train_batch_size 16 \
  --train_learning_rate 2e-5 \
  --train_weight_decay 0.01 \
  --max_seq_length 256 \
  --seed 42 \
  --data_max_samples 10000
```

### Argument Name Mapping

| Old Argument | New Argument | Config Path |
|-------------|--------------|-------------|
| `--model` | `--model_name` | `model_name` |
| `--dataset` | `--data_dataset` | `data.dataset` |
| `--data_dir` | `--paths_data_dir` | `paths.data_dir` |
| `--out_dir` | `--run_name` | N/A (combined with `paths.runs_dir`) |
| `--epochs` | `--train_epochs` | `train.epochs` |
| `--batch_size` | `--train_batch_size` | `train.batch_size` |
| `--lr` | `--train_learning_rate` | `train.learning_rate` |
| `--weight_decay` | `--train_weight_decay` | `train.weight_decay` |
| `--max_seq_length` | `--max_seq_length` | `max_seq_length` |
| `--seed` | `--seed` | `seed` |
| `--max_samples` | `--data_max_samples` | `data.max_samples` |

### Key Changes

1. **Output Directory**: Instead of `--out_dir runs/my-run`, use `--run_name my-run`. The full path is constructed as `{paths.runs_dir}/{run_name}`.

2. **Config File**: You can now set defaults in `config/config.yaml` and only override what changes between runs.

3. **Nested Settings**: Use underscore notation (e.g., `train_epochs`) for nested config values.

## Evaluation Script Migration

### Old Command Format

```bash
python -m fnd.eval.evaluate \
  --dataset kaggle_fake_real \
  --data_dir data/processed/kaggle_fake_real \
  --model_dir runs/my-run/model \
  --out_dir runs/my-run \
  --max_seq_length 256
```

### New Command Format (Recommended)

```bash
python -m fnd.eval.evaluate \
  --config config/config.yaml \
  --model_dir runs/my-run/model \
  --out_dir runs/my-run
```

### New Command Format (Without Config)

For backward compatibility, you can still use:

```bash
python -m fnd.eval.evaluate \
  --model_dir runs/my-run/model \
  --out_dir runs/my-run \
  --data_dataset kaggle_fake_real \
  --paths_data_dir data/processed/kaggle_fake_real \
  --max_seq_length 256
```

## Web App & API

No migration needed! The web app and API maintain the same external interface:

```bash
# Web app - unchanged
streamlit run src/fnd/web/app.py -- \
  --model_dir runs/my-run/model \
  --samples_file data/test/test_samples.json

# API - unchanged
export MODEL_DIR=runs/my-run/model
uvicorn fnd.api.main:app --host 0.0.0.0 --port 8000
```

However, they now use centralized model loading utilities internally for better consistency and error handling.

## Benefits of Migration

1. **Less Typing**: Set common values once in config.yaml
2. **Better Organization**: Logically grouped settings
3. **Reproducibility**: Config saved with each training run
4. **Validation**: Automatic validation catches errors early
5. **Documentation**: Configuration files are self-documenting
6. **Flexibility**: Still override anything via CLI when needed

## Recommended Workflow

1. **Create a base config** (`config/config.yaml`) with your typical settings
2. **Run experiments** by only specifying what changes:
   ```bash
   # Experiment 1: Different model
   python -m fnd.training.train --config config/config.yaml --run_name exp1-bert --model_name bert-base-uncased
   
   # Experiment 2: Different hyperparameters
   python -m fnd.training.train --config config/config.yaml --run_name exp2-lr --train_learning_rate 5e-5
   
   # Experiment 3: Smaller dataset
   python -m fnd.training.train --config config/config.yaml --run_name exp3-quick --data_max_samples 1000 --train_epochs 1
   ```
3. **Configuration is auto-saved** to `runs/{run_name}/config.yaml` for reproducibility

## Troubleshooting

### "Required argument missing" errors

If you see errors about required arguments, make sure you either:
- Provide `--config` pointing to a valid YAML file, OR
- Set all required values via CLI overrides

### "Invalid override key" errors

Check that your CLI override uses the correct notation:
- Use underscores, not dots: `--train_epochs` not `--train.epochs`
- Match the config structure: see `config/config.yaml` for available paths

### Values not being applied

Remember: CLI overrides take precedence over config file values. If a value isn't changing:
1. Check spelling of the override argument
2. Verify the value is being logged in training output
3. Check for typos in config.yaml
