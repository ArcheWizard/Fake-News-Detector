import argparse
import json
import os
from functools import partial

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from fnd.data.datasets import load_dataset
from fnd.models.factory import load_model_and_tokenizer
from fnd.training.metrics import compute_metrics


def tokenize_function(tokenizer, examples, max_length: int):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model for fake news detection")
    parser.add_argument("--model", required=True, help="HF model name, e.g., roberta-base")
    parser.add_argument("--dataset", required=True, choices=["kaggle_fake_real"], help="Dataset identifier")
    parser.add_argument("--data_dir", required=True, help="Directory with processed/raw dataset files")
    parser.add_argument("--out_dir", required=True, help="Output directory for run artifacts")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_dataset(
        args.dataset,
        args.data_dir,
        seed=args.seed,
        val_size=0.1,
        test_size=0.1,
        max_samples=args.max_samples,
    )

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(bundle.train_df)
    val_ds = Dataset.from_pandas(bundle.validation_df)
    test_ds = Dataset.from_pandas(bundle.test_df)

    tokenizer, model = load_model_and_tokenizer(
        args.model,
        num_labels=len(bundle.id2label),
        id2label=bundle.id2label,
        label2id=bundle.label2id,
    )

    tokenized_train = train_ds.map(lambda x: tokenize_function(tokenizer, x, args.max_seq_length), batched=True)
    tokenized_val = val_ds.map(lambda x: tokenize_function(tokenizer, x, args.max_seq_length), batched=True)
    tokenized_test = test_ds.map(lambda x: tokenize_function(tokenizer, x, args.max_seq_length), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on validation and test
    eval_val = trainer.evaluate(eval_dataset=tokenized_val)
    eval_test = trainer.evaluate(eval_dataset=tokenized_test)

    # Save model and tokenizer
    model_dir = os.path.join(args.out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save metrics and label map
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"eval_val": eval_val, "eval_test": eval_test}, f, indent=2)
    with open(os.path.join(model_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": bundle.label2id, "id2label": bundle.id2label}, f, indent=2)

    print(f"Saved model to {model_dir}")
    print(f"Validation metrics: {eval_val}")
    print(f"Test metrics: {eval_test}")


if __name__ == "__main__":
    main()
