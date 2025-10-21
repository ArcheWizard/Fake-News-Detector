from typing import Optional


def explain_text_with_lime(model_dir: str, text: str, max_seq_length: int = 256):
    """Optional LIME explanation for a single text.

    Returns (exp, html) if LIME is available; otherwise returns (None, None).
    """
    try:
        from lime.lime_text import LimeTextExplainer  # type: ignore
        from transformers import pipeline

        clf = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            return_all_scores=True,
            truncation=True,
            max_length=max_seq_length,
        )

        class_names = ["real", "fake"]
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text, lambda x: [[p["score"] for p in clf(t)] for t in x], num_features=10)
        return exp, exp.as_html()
    except Exception:
        return None, None
