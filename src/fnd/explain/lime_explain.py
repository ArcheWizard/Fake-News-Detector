"""LIME explanation utilities for text classification."""
from typing import List, Optional, Tuple

from fnd.models.utils import create_classification_pipeline


def explain_text_with_lime(
    model_dir: str,
    text: str,
    max_seq_length: int = 256,
    *,
    num_features: int = 10,
    num_samples: int = 500,
    random_state: int = 0,
) -> Tuple[Optional[object], Optional[str]]:
    """Generate a LIME explanation for a single text.

    Returns (exp, html) or (None, None) if LIME is unavailable.

    Notes:
    - Uses the centralized pipeline loader for consistency.
    - Handles varying pipeline output shapes robustly.
    - Returns probabilities aligned to discovered class_names.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")

    # Lazy import to keep LIME optional
    from lime.lime_text import LimeTextExplainer  # type: ignore
    import numpy as np  # type: ignore

    # Create pipeline (return_all_scores=True ensures per-class scores)
    clf = create_classification_pipeline(
        model_dir=model_dir,
        max_length=max_seq_length,
        device=None,
        return_all_scores=True,
    )

    # Discover class names
    class_names = _discover_class_names(clf)
    if not class_names:
        class_names = ["real", "fake"]

    def predict_proba(texts: List[str]):
        """LIME-compatible predict_proba: List[str] -> np.array[n, k]."""
        rows: List[List[float]] = []
        for t in texts:
            out = clf(t)

            # Normalize output to List[Dict[label, score]] for a single sample
            items = _normalize_single_output(out)

            # Map label -> score (lowercased)
            score_map = {str(d.get("label", "")).lower(): float(d.get("score", 0.0)) for d in items}

            # Align to class_names order
            probs = [score_map.get(name, 0.0) for name in class_names]

            # Re-normalize in case of minor numeric drift
            s = float(sum(probs))
            if s > 0.0:
                probs = [p / s for p in probs]
            rows.append(probs)

        return np.asarray(rows, dtype=float)

    explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )
    return exp, exp.as_html()


def _discover_class_names(clf) -> List[str]:
    """Try to infer class names from model config or a probe call."""
    names: List[str] = []

    model = getattr(clf, "model", None)
    config = getattr(model, "config", None)
    id2label = getattr(config, "id2label", None)

    def _to_int_safe(k):
        try:
            return int(k)
        except Exception:
            return k

    # Prefer id2label when available
    if isinstance(id2label, dict) and id2label:
        keys = sorted(id2label.keys(), key=_to_int_safe)
        names = [str(id2label[k]).lower() for k in keys]
    elif isinstance(id2label, (list, tuple)) and len(id2label) > 0:
        names = [str(x).lower() for x in id2label]

    # Fall back to probing the pipeline output
    if not names:
        probe = clf("probe")
        items = _normalize_single_output(probe)
        if items and isinstance(items[0], dict) and "label" in items[0]:
            # Sort labels for deterministic ordering
            names = [str(d["label"]).lower() for d in sorted(items, key=lambda d: str(d.get("label", "")))]

    return names


def _normalize_single_output(out) -> List[dict]:
    """Normalize pipeline output to a single-sample List[Dict[label, score]]."""
    # Expected shapes:
    # - Single input with return_all_scores=True: List[List[Dict]]
    # - Single input unexpectedly: List[Dict]
    # - Batch input: List[List[Dict]] (we will never pass a batch here)
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            # Already per-class for single input
            return out
        if isinstance(first, list):
            # Take first (we call pipeline per-single text in predict_proba)
            return first
    raise ValueError(f"Unexpected pipeline output shape: {type(out)} -> {repr(out)[:200]}")
