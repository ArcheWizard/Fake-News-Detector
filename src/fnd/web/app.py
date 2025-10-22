import argparse
import json
import os
from typing import Any, Dict, List

import streamlit as st

from fnd.models.utils import create_classification_pipeline


@st.cache_resource
def load_pipeline(model_dir: str):
    """Load classification pipeline using centralized utility."""
    return create_classification_pipeline(
        model_dir=model_dir,
        max_length=256,
        device=None,  # Auto-detect
        return_all_scores=True
    )


@st.cache_data
def load_test_samples(samples_file: str = "test_samples.json"):
    """Load test samples if available."""
    if os.path.exists(samples_file):
        with open(samples_file, "r") as f:
            return json.load(f)
    return None


def load_metrics(model_dir: str):
    """Load saved test metrics if available."""
    # Look for metrics.json in parent directory (runs/roberta-kfr/)
    parent_dir = os.path.dirname(model_dir)
    metrics_path = os.path.join(parent_dir, "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def main_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_dir", required=True, help="Path to saved model directory")
    parser.add_argument("--samples_file", default="test_samples.json", help="Test samples JSON file")
    args, _ = parser.parse_known_args()
    return args


def app():
    args = main_cli()
    st.title("Fake News Detector")
    st.caption("Transformer-based classifier (binary: real vs fake)")

    # Initialize session state
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if "true_label" not in st.session_state:
        st.session_state.true_label = None

    # Display test set metrics if available
    metrics = load_metrics(args.model_dir)
    if metrics:
        st.sidebar.title("Model Performance")
        st.sidebar.metric("Test Accuracy", f"{metrics.get('eval_accuracy', 0):.3f}")
        st.sidebar.metric("Test F1 Score", f"{metrics.get('eval_f1', 0):.3f}")
        st.sidebar.metric("Test Precision", f"{metrics.get('eval_precision', 0):.3f}")
        st.sidebar.metric("Test Recall", f"{metrics.get('eval_recall', 0):.3f}")
        if 'eval_roc_auc' in metrics and metrics['eval_roc_auc'] is not None:
            st.sidebar.metric("Test ROC AUC", f"{metrics['eval_roc_auc']:.3f}")

    clf = load_pipeline(args.model_dir)

    # Load test samples if available
    test_samples = load_test_samples(args.samples_file)

    # Add sample selector
    if test_samples:
        st.sidebar.markdown("---")
        st.sidebar.title("Test Examples")

        sample_category = st.sidebar.selectbox(
            "Select category",
            ["None", "Real News", "Fake News"]
        )

        if sample_category != "None":
            category_key = "real" if sample_category == "Real News" else "fake"
            samples = test_samples.get(category_key, [])

            if samples:
                sample_idx = st.sidebar.selectbox(
                    "Select sample",
                    range(len(samples)),
                    format_func=lambda x: f"Sample {x+1}"
                )

                selected_sample = samples[sample_idx]

                if st.sidebar.button("Load Sample"):
                    # Update session state directly
                    st.session_state.text_input = selected_sample["text"]
                    st.session_state.true_label = selected_sample["label"]
                    st.rerun()

    # Text input area - use key that matches session state variable
    text = st.text_area(
        "Enter news article text",
        height=200,
        key="text_input"  # This directly binds to st.session_state.text_input
    )

    # Show true label if loaded from samples
    if st.session_state.true_label is not None:
        st.info(f"True label: **{st.session_state.true_label}**")

    if st.button("Predict") and text.strip():
        outputs = clf(text)

        # Type-safe handling of pipeline output
        if isinstance(outputs, list) and len(outputs) > 0:
                output_list: List[Dict[str, Any]] = outputs[0]  # Explicit type hint
                # Sort by label for consistent display
                sorted_outputs = sorted(output_list, key=lambda x: str(x.get("label", "")))

            # Display all scores
                scores_dict = {
                    str(item.get("label", "")): round(float(item.get("score", 0)), 4)
                    for item in sorted_outputs
                }
            st.write("**Prediction Scores:**", scores_dict)

            # Get top prediction
            top = max(sorted_outputs, key=lambda x: float(x.get("score", 0)))
            label_str = str(top.get("label", ""))
            score_float = float(top.get("score", 0))

            # Check if prediction matches true label
            if st.session_state.true_label is not None:
                true_label = st.session_state.true_label
                if label_str.lower() == true_label.lower():
                    st.success(f"✅ Predicted: **{label_str}** (p={score_float:.3f}) - CORRECT")
                else:
                    st.error(f"❌ Predicted: **{label_str}** (p={score_float:.3f}) - Expected: {true_label}")
            else:
                st.success(f"Predicted: **{label_str}** (p={score_float:.3f})")


if __name__ == "__main__":
    app()