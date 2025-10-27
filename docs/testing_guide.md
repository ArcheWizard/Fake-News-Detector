# Testing Guide for Streamlit Web App

This guide explains how to test model accuracy using the enhanced Streamlit web app.

## Quick Start

Prerequisite: ensure you've prepared the processed dataset at `data/processed/kaggle_fake_real/dataset.csv` (see README or `python -m fnd.data.prepare ...`).

### 1. Extract Test Samples

First, create a JSON file with sample test examples from your dataset:

```bash
python scripts/extract_test_samples.py \
 --data_dir data/processed/kaggle_fake_real \
 --num_samples 20 \
 --out_file data/test/test_samples.json
```

This will create `test_samples.json` with 20 real news articles and 20 fake news articles from your test set.

### 2. Launch the Enhanced Web App

Run the Streamlit app with both the model directory and samples file:

```bash
streamlit run src/fnd/web/app.py -- \
 --model_dir runs/my-experiment/model \
 --samples_file data/test/test_samples.json
```

## Using the Web App

### Sidebar Features

#### Model Performance Metrics

The sidebar displays test set metrics automatically loaded from `runs/roberta-kfr/metrics.json`:

- **Test Accuracy**: Overall classification accuracy
- **Test F1 Score**: Harmonic mean of precision and recall
- **Test Precision**: True positives / (True positives + False positives)
- **Test Recall**: True positives / (True positives + False negatives)
- **Test ROC AUC**: Area under the ROC curve

#### Test Examples Selector

1. **Select category**: Choose "Real News" or "Fake News"
2. **Select sample**: Pick a numbered sample (1-20)
3. **Load Sample**: Click to load the text into the main text area

### Testing Workflow

1. **Select a category** (Real News or Fake News) from the sidebar
2. **Pick a sample number** from the dropdown
3. **Click "Load Sample"** to populate the text area
4. The true label will be displayed in an info box
5. **Click "Predict"** to see the model's prediction
6. The result will show:

- **CORRECT** if the prediction matches the true label
- **Expected: [true_label]** if the prediction is wrong
- Prediction scores for all classes
- Confidence score (probability)

### Manual Testing

You can also:

- **Type or paste** your own news articles directly into the text area
- **Edit loaded samples** to see how changes affect predictions
- **Test edge cases** like very short text, mixed content, etc.

## Understanding Results

### Prediction Display

```text
Prediction Scores: {'fake': 0.0234, 'real': 0.9766}
Predicted: real (p=0.977) - CORRECT
```

- **Prediction Scores**: Probability for each class (should sum to ~1.0)
- **Predicted**: The class with highest probability
- **p=0.977**: Confidence score (probability of the predicted class)
- **CORRECT/Expected**: Validation against true label (only for loaded samples)

### Performance Indicators

- **High accuracy (>0.95)**: Model performs very well on test data
- **Correct predictions**: Model successfully identifies the article type
- **High confidence (>0.90)**: Model is confident in its prediction
- **Low confidence (<0.60)**: Model is uncertain, borderline case

## Troubleshooting

### No test samples appearing

- Make sure you ran `extract_test_samples.py` first
- Check that `test_samples.json` exists in the current directory
- Verify the `--samples_file` argument matches your file location

### No metrics in sidebar

- Ensure you ran the evaluation script: `python -m fnd.eval.evaluate ...`
- Check that `metrics.json` exists in the parent directory of your model
- The file should be at: `runs/roberta-kfr/metrics.json`

### Model loading errors

- Verify the `--model_dir` path is correct
- Ensure the directory contains: `config.json`, `model.safetensors`, `tokenizer.json`
- Check that you've trained a model first

## Tips for Effective Testing

1. **Test both classes equally**: Try similar numbers of real and fake samples
2. **Look for patterns**: Note which types of articles the model struggles with
3. **Test edge cases**: Very short articles, titles only, very long articles
4. **Compare confidence**: High accuracy with low confidence might indicate overfitting
5. **Document failures**: Keep track of misclassified examples for model improvement

## Current Testing Limitations & Improvement Plan

Recent improvements (2025-10-27):

- Import/smoke tests consolidated into `test_imports.py` (less redundancy)
- Shared fixtures and mocks in `conftest.py` (DRY, easier maintenance)
- Parametrization and robust assertions in key tests

While the project aims for comprehensive coverage, please note:

- Some test methods are stubs or incomplete and will be finished for full edge case coverage.
- Integration tests may require specific directories/files (e.g., model checkpoints) to exist; setup scripts or mocks are recommended for CI/CD.
- Advanced modules (LIME/SHAP explainability, optimization) have basic tests; more robust scenario-based and error tests are planned.
- Performance and stress testing (large datasets, model loading errors) are not yet implemented but are on the roadmap.

Refer to [docs/AI/improvement_plan.md](AI/improvement_plan.md) for details and progress tracking.

## Next Steps

After testing through the UI, you can:

- Run formal evaluation: `python -m fnd.eval.evaluate ...`
- Check confusion matrix and ROC curves in the output directory
- Retrain with different hyperparameters if performance is poor
- Add more training data or try data augmentation
- Experiment with different base models (BERT, RoBERTa, etc.)
