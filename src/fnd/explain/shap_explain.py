def explain_text_with_shap(model_dir: str, text: str, max_seq_length: int = 256):
    """Optional SHAP explanation for a single text.

    Returns a tuple (explanation, tokens) if SHAP is available; otherwise returns (None, None).
    """
    try:
        import shap  # type: ignore
        from transformers import pipeline

        clf = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            return_all_scores=True,
            truncation=True,
            max_length=max_seq_length,
        )
        # SHAP TextExplainer is experimental; fallback to generic Explainer
        explainer = shap.Explainer(clf)
        explanation = explainer([text])
        return explanation, None
    except Exception:
        return None, None
