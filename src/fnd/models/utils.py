"""Centralized model loading and utility functions.

This module provides unified functions for loading models, creating pipelines,
and performing common model operations across the codebase.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)

from fnd.exceptions import ModelLoadError


def load_model_and_tokenizer_from_dir(
    model_dir: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a trained model and tokenizer from a directory.

    This function loads a previously fine-tuned model and its tokenizer from
    a directory containing model checkpoint files.

    Args:
        model_dir: Path to directory containing model files
            (config.json, model weights, tokenizer files)

    Returns:
        Tuple of (model, tokenizer):
            - model: Loaded PreTrainedModel for sequence classification
            - tokenizer: Loaded PreTrainedTokenizer

    Raises:
        ModelLoadError: If model directory is invalid, files are missing,
            or loading fails

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer_from_dir("runs/roberta-kfr/model")
        >>> # Use for inference
        >>> inputs = tokenizer("Sample text", return_tensors="pt")
        >>> outputs = model(**inputs)

    Note:
        Directory must contain at minimum:
        - config.json (model configuration)
        - tokenizer_config.json (tokenizer configuration)
        - Model weights file (pytorch_model.bin or model.safetensors)
    """
    model_dir = str(Path(model_dir).expanduser().absolute())

    if not os.path.isdir(model_dir):
        raise ModelLoadError(
            f"Model directory not found: {model_dir}\n"
            f"Please ensure the path exists and contains a trained model."
        )

    # Check for required files
    required_files = ["config.json", "tokenizer_config.json"]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(model_dir, f))
    ]

    if missing_files:
        raise ModelLoadError(
            f"Model directory incomplete: {model_dir}\n"
            f"Missing files: {missing_files}\n"
            f"Please ensure you've trained a model and saved it correctly."
        )

    # Check for model weights
    weight_files = ["pytorch_model.bin", "model.safetensors"]
    has_weights = any(os.path.exists(os.path.join(model_dir, f)) for f in weight_files)

    if not has_weights:
        raise ModelLoadError(
            f"No model weights found in {model_dir}\nExpected one of: {weight_files}"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load model from {model_dir}\n"
            f"Error: {str(e)}\n"
            f"The model files may be corrupted or incompatible."
        ) from e


def create_classification_pipeline(
    model_dir: str,
    max_length: int = 256,
    device: Optional[int] = None,
    return_all_scores: bool = True,
):
    """Create a HuggingFace pipeline for text classification.

    This is a convenient wrapper around the HuggingFace pipeline API
    optimized for fake news classification.

    Args:
        model_dir: Path to directory containing trained model
        max_length: Maximum sequence length for tokenization. Default: 256
        device: Device ID for GPU (-1 for CPU, None for auto-detect). Default: None
        return_all_scores: Whether to return scores for all classes. Default: True

    Returns:
        HuggingFace TextClassificationPipeline configured for inference

    Raises:
        ModelLoadError: If model loading or pipeline creation fails

    Examples:
        Basic usage:
        >>> clf = create_classification_pipeline("runs/roberta-kfr/model")
        >>> result = clf("This is a sample news article")
        >>> print(result)
        [{'label': 'LABEL_0', 'score': 0.95}, {'label': 'LABEL_1', 'score': 0.05}]

        Force CPU:
        >>> clf = create_classification_pipeline("runs/roberta-kfr/model", device=-1)

        Custom max length:
        >>> clf = create_classification_pipeline("runs/roberta-kfr/model", max_length=512)

    Note:
        - First call downloads model if needed
        - Subsequent calls use cached model for faster loading
        - Pipeline handles tokenization, inference, and post-processing
    """
    model_dir = str(Path(model_dir).expanduser().absolute())

    if not os.path.isdir(model_dir):
        raise ModelLoadError(f"Model directory not found: {model_dir}")

    try:
        clf = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            return_all_scores=return_all_scores,
            truncation=True,
            max_length=max_length,
            device=device,
        )
        return clf
    except Exception as e:
        raise ModelLoadError(
            f"Failed to create classification pipeline from {model_dir}\n"
            f"Error: {str(e)}"
        ) from e


def load_label_mapping(model_dir: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load label mappings from a trained model directory.

    Args:
        model_dir: Path to directory containing model and label_map.json

    Returns:
        Tuple of (id2label, label2id) dictionaries

    Raises:
        ModelLoadError: If label mapping file not found or invalid

    Examples:
        >>> id2label, label2id = load_label_mapping("runs/roberta-kfr/model")
        >>> print(id2label)
        {0: 'real', 1: 'fake'}
        >>> print(label2id)
        {'real': 0, 'fake': 1}

    Note:
        Falls back to loading from config.json if label_map.json doesn't exist
    """
    model_dir = str(Path(model_dir).expanduser().absolute())

    # Try label_map.json first
    label_map_path = os.path.join(model_dir, "label_map.json")
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path) as f:
                data = json.load(f)
                return data.get("id2label", {}), data.get("label2id", {})
        except Exception as e:
            raise ModelLoadError(f"Failed to load label_map.json: {e}") from e

    # Fall back to config.json
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
                id2label = config.get("id2label", {})
                # Convert string keys to int if needed
                id2label = {int(k): v for k, v in id2label.items()}
                label2id = {v: k for k, v in id2label.items()}
                return id2label, label2id
        except Exception as e:
            raise ModelLoadError(f"Failed to load config.json: {e}") from e

    raise ModelLoadError(
        f"No label mapping found in {model_dir}\n"
        f"Expected label_map.json or config.json with id2label field"
    )


def get_model_info(model_dir: str) -> Dict[str, Any]:
    """Get information about a trained model.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary containing model information:
            - model_type: Type of model (e.g., 'roberta', 'bert')
            - num_labels: Number of classification labels
            - vocab_size: Size of vocabulary
            - hidden_size: Hidden layer size
            - num_layers: Number of transformer layers
            - id2label: Label ID to name mapping
            - label2id: Label name to ID mapping

    Raises:
        ModelLoadError: If config file cannot be read

    Examples:
        >>> info = get_model_info("runs/roberta-kfr/model")
        >>> print(f"Model type: {info['model_type']}")
        Model type: roberta
        >>> print(f"Labels: {info['id2label']}")
        Labels: {0: 'real', 1: 'fake'}
    """
    model_dir = str(Path(model_dir).expanduser().absolute())
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        raise ModelLoadError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Extract key information
        id2label = config.get("id2label", {})
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

        return {
            "model_type": config.get("model_type", "unknown"),
            "num_labels": config.get("num_labels", len(id2label)),
            "vocab_size": config.get("vocab_size", 0),
            "hidden_size": config.get("hidden_size", 0),
            "num_layers": config.get("num_hidden_layers", 0),
            "id2label": id2label,
            "label2id": label2id,
        }
    except Exception as e:
        raise ModelLoadError(f"Failed to read model info: {e}") from e
