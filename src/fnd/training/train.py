import argparse
import json
import os
from typing import Any, cast

import yaml
from datasets import Dataset
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from fnd.config import FNDConfig
from fnd.data.datasets import load_dataset
from fnd.models.factory import load_model_and_tokenizer
from fnd.training.metrics import compute_metrics


def tokenize_function(tokenizer, examples, max_length: int):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model for fake news detection"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--run_name", default="default-run", help="Name for this training run"
    )
    parser.add_argument(
        "--profile",
        help="Optional training profile to apply (e.g., fast, memory, distil)",
    )
    parser.add_argument("--model_name", help="Override model name")
    parser.add_argument("--data_dataset", help="Override dataset name")
    parser.add_argument("--paths_data_dir", help="Override data directory")
    parser.add_argument("--train_epochs", type=int, help="Override training epochs")
    parser.add_argument("--train_batch_size", type=int, help="Override batch size")
    parser.add_argument(
        "--train_learning_rate", type=float, help="Override learning rate"
    )
    parser.add_argument("--data_max_samples", type=int, help="Override max samples")
    parser.add_argument("--seed", type=int, help="Override random seed")
    args = parser.parse_args()

    # Build overrides from CLI, profile overlay first (so CLI wins)
    def _flatten(prefix: str, obj: Any, out: dict):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}_{k}" if prefix else k, v, out)
        else:
            out[prefix] = obj

    profile_overrides: dict[str, Any] = {}
    if args.profile:
        profile_dir = os.path.join(os.path.dirname(args.config), "profiles")
        profile_path = os.path.join(profile_dir, f"{args.profile}.yaml")
        if os.path.isfile(profile_path):
            with open(profile_path) as pf:
                profile_dict = yaml.safe_load(pf) or {}
            _flatten("", profile_dict, profile_overrides)
        else:
            print(
                f"[warn] Profile '{args.profile}' not found at {profile_path}; continuing without it"
            )

    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ("config", "run_name", "profile")
    }
    merged_overrides = {**profile_overrides, **cli_overrides}

    # Load config from YAML with merged overrides (profile < CLI)
    config = FNDConfig.from_yaml_with_overrides(args.config, **merged_overrides)

    # Create output directory for this run
    output_dir = os.path.join(config.paths.runs_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    bundle = load_dataset(
        config.data.dataset,
        config.paths.data_dir,
        seed=config.seed,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
        max_samples=config.data.max_samples,
    )

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(bundle.train_df)
    val_ds = Dataset.from_pandas(bundle.validation_df)
    test_ds = Dataset.from_pandas(bundle.test_df)

    tokenizer, model = load_model_and_tokenizer(
        config.model_name,
        num_labels=len(bundle.id2label),
        id2label=bundle.id2label,
        label2id=bundle.label2id,
    )

    tokenized_train = train_ds.map(
        lambda x: tokenize_function(tokenizer, x, config.max_seq_length), batched=True
    )
    tokenized_val = val_ds.map(
        lambda x: tokenize_function(tokenizer, x, config.max_seq_length), batched=True
    )
    tokenized_test = test_ds.map(
        lambda x: tokenize_function(tokenizer, x, config.max_seq_length), batched=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Optional experiment tracking integrations via env var
    report_to_env = os.getenv("FND_REPORT_TO")
    if report_to_env:
        report_to = [s.strip() for s in report_to_env.split(",") if s.strip()]
    else:
        report_to = ["none"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.eval.batch_size,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        warmup_ratio=config.train.warmup_ratio,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        max_grad_norm=config.train.max_grad_norm,
        fp16=config.train.fp16,
        bf16=config.train.bf16,
        eval_strategy=config.train.evaluation_strategy,
        save_strategy=config.train.save_strategy,
        logging_steps=config.train.logging_steps,
        load_best_model_at_end=config.train.load_best_model_at_end,
        metric_for_best_model=config.train.metric_for_best_model,
        seed=config.seed,
        report_to=report_to,
        run_name=args.run_name,
        lr_scheduler_type=config.train.lr_scheduler_type,
        dataloader_num_workers=config.train.dataloader_num_workers,
        gradient_checkpointing=config.train.gradient_checkpointing,
        optim=config.train.optim,
        torch_compile=config.train.torch_compile,
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

    # Evaluate on validation and test - cast to satisfy type checker
    eval_val = trainer.evaluate(eval_dataset=cast("Any", tokenized_val))
    eval_test = trainer.evaluate(eval_dataset=cast("Any", tokenized_test))

    # Save model and tokenizer
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save configuration
    config.to_yaml(os.path.join(output_dir, "config.yaml"))

    # Save metrics and label map
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"eval_val": eval_val, "eval_test": eval_test}, f, indent=2)
    with open(os.path.join(model_dir, "label_map.json"), "w") as f:
        json.dump(
            {"label2id": bundle.label2id, "id2label": bundle.id2label}, f, indent=2
        )

    print(f"Saved model to {model_dir}")
    print(f"Validation metrics: {eval_val}")
    print(f"Test metrics: {eval_test}")


if __name__ == "__main__":
    main()
