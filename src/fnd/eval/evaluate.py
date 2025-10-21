import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from fnd.data.datasets import load_dataset
from fnd.training.metrics import compute_metrics


def update_metrics_history(metrics: dict):
    """Append the latest evaluation metrics to the test_metrics_history.md file."""
    history_file = "docs/test_metrics_history.md"
    date_str = datetime.now().strftime("%Y-%m-%d")
    epoch = metrics.get("epoch", "N/A")
    eval_loss = metrics.get("eval_loss", "N/A")
    eval_accuracy = metrics.get("eval_accuracy", "N/A")
    eval_precision = metrics.get("eval_precision", "N/A")
    eval_recall = metrics.get("eval_recall", "N/A")
    eval_f1 = metrics.get("eval_f1", "N/A")
    eval_roc_auc = metrics.get("eval_roc_auc", "N/A")

    # Prepare the new entry
    new_entry = f"| {date_str} | {epoch} | {eval_loss} | {eval_accuracy} | {eval_precision} | {eval_recall} | {eval_f1} | {eval_roc_auc} |\n"

    # Append to the history file
    with open(history_file, "a") as f:
        f.write(new_entry)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test split")
    parser.add_argument("--dataset", required=True, choices=["kaggle_fake_real"], help="Dataset identifier")
    parser.add_argument("--data_dir", required=True, help="Directory with processed/raw dataset files")
    parser.add_argument("--model_dir", required=True, help="Directory containing saved model and tokenizer")
    parser.add_argument("--out_dir", required=True, help="Directory to save evaluation artifacts")
    parser.add_argument("--max_seq_length", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_dataset(args.dataset, args.data_dir)
    test_ds = Dataset.from_pandas(bundle.test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    tokenized_test = test_ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=args.max_seq_length), batched=True)

    # Create data collator for proper batching
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, data_collator=data_collator, compute_metrics=compute_metrics)
    metrics = trainer.evaluate(tokenized_test)

    # Confusion matrix and ROC curve
    preds_output = trainer.predict(tokenized_test)

    # Handle predictions - take first element if tuple
    predictions = preds_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    probs = np.exp(predictions - predictions.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    y_true = preds_output.label_ids
    if y_true is None:
        raise ValueError("No labels found in predictions")

    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "fake"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    try:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        roc_auc = roc_auc_score(y_true, probs[:, 1])
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(args.out_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
    except Exception:
        roc_path = None

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Update the metrics history
    update_metrics_history(metrics)

    print(f"Saved metrics to {os.path.join(args.out_dir, 'metrics.json')}")
    print(f"Saved confusion matrix to {cm_path}")
    if roc_path:
        print(f"Saved ROC curve to {roc_path}")


if __name__ == "__main__":
    main()
