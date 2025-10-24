"""Tests for evaluation CLI and metrics."""

import importlib
import subprocess
import sys

# Test that evaluate.py can be imported and main() can be called


def test_evaluate_import_and_main():
    mod = importlib.import_module("fnd.eval.evaluate")
    assert hasattr(mod, "main")


def test_evaluate_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "fnd.eval.evaluate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Usage" in result.stdout or "usage" in result.stdout


def test_evaluate_cli_invalid_args():
    result = subprocess.run(
        [sys.executable, "-m", "fnd.eval.evaluate", "--invalid"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower()


# For CLI coverage, use subprocess to call the script if needed
