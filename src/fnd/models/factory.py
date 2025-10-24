"""Model factory functions for loading and initializing transformer models."""

from typing import Dict, Tuple

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load a pre-trained transformer model and tokenizer for sequence classification.

    Downloads and initializes a HuggingFace transformer model with a classification
    head suitable for fake news detection. Supports any model available on the
    HuggingFace Hub that has AutoModelForSequenceClassification support.

    Args:
        model_name: HuggingFace model identifier, e.g.:
            - "roberta-base": RoBERTa base model
            - "bert-base-uncased": BERT base uncased
            - "distilbert-base-uncased": DistilBERT base
        num_labels: Number of output classes (typically 2 for binary classification)
        id2label: Mapping from integer label IDs to string names
            Example: {0: "real", 1: "fake"}
        label2id: Reverse mapping from string names to integer IDs
            Example: {"real": 0, "fake": 1}

    Returns:
        Tuple of (tokenizer, model):
            - tokenizer: PreTrainedTokenizer for text tokenization
            - model: PreTrainedModel with classification head initialized

    Raises:
        OSError: If model_name is not found on HuggingFace Hub
        ValueError: If model doesn't support sequence classification

    Examples:
        Load RoBERTa for binary fake news classification:
        >>> tokenizer, model = load_model_and_tokenizer(
        ...     model_name="roberta-base",
        ...     num_labels=2,
        ...     id2label={0: "real", 1: "fake"},
        ...     label2id={"real": 0, "fake": 1}
        ... )
        >>> print(model.config.num_labels)
        2

        Load BERT:
        >>> tokenizer, model = load_model_and_tokenizer(
        ...     model_name="bert-base-uncased",
        ...     num_labels=2,
        ...     id2label={0: "real", 1: "fake"},
        ...     label2id={"real": 0, "fake": 1}
        ... )

    Note:
        - First call will download model weights from HuggingFace Hub
        - Subsequent calls use cached weights for faster loading
        - Model is initialized with random classification head
        - Requires fine-tuning before use for fake news detection
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    return tokenizer, model
