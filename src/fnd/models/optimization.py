"""Model optimization utilities: quantization and pruning.

These helpers provide optional post-training optimizations for faster inference.
They operate on saved Hugging Face model directories.
"""

from __future__ import annotations

import json
import os

import torch

from fnd.models.utils import load_model_and_tokenizer_from_dir


def quantize_model_dir(
    model_dir: str, out_dir: str, dtype: torch.dtype = torch.qint8
) -> str:
    """Apply dynamic quantization to a saved model directory and write to out_dir.

    Args:
        model_dir: Path to directory containing a saved HF model (config.json, model.safetensors, tokenizer, etc.)
        out_dir: Destination directory for the quantized model
        dtype: Quantized dtype to use (default torch.qint8)

    Returns:
        The output directory path where the quantized model is saved.

    Notes:
        - Dynamic quantization targets Linear layers by default.
        - Works best on CPU inference. GPU performance may not improve.
    """
    model, tokenizer = load_model_and_tokenizer_from_dir(model_dir)
    model.eval()

    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)

    os.makedirs(out_dir, exist_ok=True)
    # Save model and tokenizer in HF format
    qmodel.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Copy label map if present
    label_map_src = os.path.join(model_dir, "label_map.json")
    if os.path.isfile(label_map_src):
        with open(label_map_src, "r") as f:
            label_map = json.load(f)
        with open(os.path.join(out_dir, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)

    return out_dir


def prune_linear_layers(model, amount: float = 0.2):
    """Apply unstructured L1 pruning to Linear layers in-place.

    Args:
        model: A torch.nn.Module (e.g., HF sequence classification model)
        amount: Fraction of connections to prune in each Linear layer

    Returns:
        The pruned model (same instance), with reparam removed.
    """
    import torch.nn as nn
    import torch.nn.utils.prune as prune  # Lazy import to avoid overhead

    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # Make pruning permanent
    return model


def prune_and_save_model_dir(model_dir: str, out_dir: str, amount: float = 0.2) -> str:
    """Load a saved model, apply pruning, and save to out_dir.

    Args:
        model_dir: Input HF model directory
        out_dir: Destination directory for the pruned model
        amount: Fraction to prune for Linear layers

    Returns:
        The output directory path where the pruned model is saved.
    """
    model, tokenizer = load_model_and_tokenizer_from_dir(model_dir)
    model.eval()
    model = prune_linear_layers(model, amount=amount)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Copy label map if present
    label_map_src = os.path.join(model_dir, "label_map.json")
    if os.path.isfile(label_map_src):
        with open(label_map_src, "r") as f:
            label_map = json.load(f)
        with open(os.path.join(out_dir, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a saved HF model (quantize or prune)"
    )
    parser.add_argument(
        "--model_dir", required=True, help="Path to source model directory"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for optimized model"
    )
    parser.add_argument("--mode", choices=["quantize", "prune"], default="quantize")
    parser.add_argument(
        "--amount", type=float, default=0.2, help="Pruning amount for 'prune' mode"
    )
    args = parser.parse_args()

    if args.mode == "quantize":
        path = quantize_model_dir(args.model_dir, args.out_dir)
        print(f"Quantized model saved to {path}")
    else:
        path = prune_and_save_model_dir(
            args.model_dir, args.out_dir, amount=args.amount
        )
        print(f"Pruned model saved to {path}")
