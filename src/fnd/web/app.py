import argparse
import sys

import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_pipeline(model_dir: str):
    # Use top_k=None instead of deprecated return_all_scores=True
    return pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir,
        top_k=None,
        truncation=True,
        max_length=256
    )


def main_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_dir", required=True, help="Path to saved model directory")
    args, _ = parser.parse_known_args()
    return args


def app():
    args = main_cli()
    st.title("Fake News Detector")
    st.caption("Transformer-based classifier (binary: real vs fake)")

    clf = load_pipeline(args.model_dir)

    text = st.text_area("Enter news article text", height=200)
    if st.button("Predict") and text.strip():
        outputs = clf(text)

        # Type-safe handling of pipeline output
        if isinstance(outputs, list) and len(outputs) > 0:
            output_list = outputs[0]
            # Sort by label for consistent display
            sorted_outputs = sorted(output_list, key=lambda x: str(x.get("label", "")))

            # Display all scores
            scores_dict = {str(o.get("label", "")): round(float(o.get("score", 0)), 4) for o in sorted_outputs}
            st.write(scores_dict)

            # Get top prediction
            top = max(sorted_outputs, key=lambda x: float(x.get("score", 0)))
            label_str = str(top.get("label", ""))
            score_float = float(top.get("score", 0))
            st.success(f"Predicted: {label_str} (p={score_float:.3f})")


if __name__ == "__main__":
    app()