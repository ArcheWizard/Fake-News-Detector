"""Fake News Detector package.

Provides data loading, model training, evaluation, explainability, and serving utilities.
"""

# Re-export subpackages for convenience
from . import data as data
from . import models as models
from . import training as training
from . import eval as _eval
from . import explain as explain

# Alias to keep name stable while avoiding shadowing builtins internally
eval = _eval

__all__ = ["data", "models", "training", "eval", "explain"]

__version__ = "0.1.0"