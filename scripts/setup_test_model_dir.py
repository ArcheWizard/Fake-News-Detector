#!/usr/bin/env python3
"""
Setup script to create a minimal fake model directory for integration tests.
Creates config.json, tokenizer.json, and other required files in a given directory.
"""

import os
import json
import sys


def setup_fake_model_dir(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    # Minimal config.json
    config = {
        "model_type": "bert",
        "num_labels": 2,
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "id2label": {"0": "real", "1": "fake"},
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    # Minimal tokenizer.json
    tokenizer = {"tokenizer_class": "BertTokenizer", "vocab_size": 30522}
    with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)
    # Add any other required files as needed
    print(f"Fake model directory created at: {model_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/setup_test_model_dir.py <model_dir>")
        sys.exit(1)
    setup_fake_model_dir(sys.argv[1])
