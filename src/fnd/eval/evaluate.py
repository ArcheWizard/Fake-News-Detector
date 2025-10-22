import argparse
import json
import os
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from transformers import Trainer

from fnd.config import FNDConfig
from fnd.data.datasets import load_dataset
from fnd.models.utils import load_model_and_tokenizer_from_dir
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
    parser.add_argument("--config", help="Path to YAML config file (optional)")
    parser.add_argument("--model_dir", required=True, help="Directory containing saved model and tokenizer")
    parser.add_argument("--out_dir", required=True, help="Directory to save evaluation artifacts")
    parser.add_argument("--data_dataset", help="Override dataset name")
    parser.add_argument("--paths_data_dir", help="Override data directory")
    parser.add_argument("--max_seq_length", type=int, help="Override max sequence length")
    args = parser.parse_args()

    # Load config if provided, otherwise use defaults
    if args.config:
        overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("config", "model_dir", "out_dir")}
        config = FNDConfig.from_yaml_with_overrides(args.config, **overrides)
    else:
        # Use defaults
        config = FNDConfig()
        if args.data_dataset:
            config.data.dataset = args.data_dataset
        if args.paths_data_dir:
            config.paths.data_dir = args.paths_data_dir
        if args.max_seq_length:
            config.max_seq_length = args.max_seq_length

    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_dataset(
        config.data.dataset,
        config.paths.data_dir,
        seed=config.seed,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
    )
    test_ds = Dataset.from_pandas(bundle.test_df)

    # Load model and tokenizer using centralized utility
    model, tokenizer = load_model_and_tokenizer_from_dir(args.model_dir)

    tokenized_test = test_ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=config.max_seq_length),
        batched=True
    )

    # Create data collator for proper batching
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, data_collator=data_collator, compute_metrics=compute_metrics)
    # Cast to satisfy type checker
    metrics = trainer.evaluate(cast(Any, tokenized_test))

    # Save metrics
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Update the metrics history
    update_metrics_history(metrics)

    # Confusion matrix and ROC curve (if configured to save plots)
    if config.eval.save_plots:
        # Cast to satisfy type checker
        preds_output = trainer.predict(cast(Any, tokenized_test))

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

        print(f"Saved confusion matrix to {cm_path}")
        if roc_path:
            print(f"Saved ROC curve to {roc_path}")

    print(f"Saved metrics to {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
