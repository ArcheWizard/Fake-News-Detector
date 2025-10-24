"""Custom exceptions for the Fake News Detector project.

This module defines a hierarchy of custom exceptions to provide clear,
actionable error messages throughout the FND codebase.
"""


class FNDException(Exception):
    """Base exception for all Fake News Detector errors.

    All custom exceptions in the FND project inherit from this base class,
    making it easy to catch any FND-specific error.
    """

    pass


class DataLoadError(FNDException):
    """Raised when data loading or preprocessing fails.

    Examples:
        - Dataset directory not found
        - Required CSV files missing
        - Invalid data format
        - Corrupted data files
    """

    pass


class InsufficientDataError(DataLoadError):
    """Raised when dataset is too small for requested operations.

    Examples:
        - Dataset smaller than minimum threshold
        - Not enough samples for train/val/test split
        - Insufficient samples per class for stratification
    """

    pass


class ModelLoadError(FNDException):
    """Raised when model loading or initialization fails.

    Examples:
        - Model directory not found
        - Missing model configuration files
        - Incompatible model architecture
        - Corrupted model weights
    """

    pass


class ConfigurationError(FNDException):
    """Raised when configuration is invalid or inconsistent.

    Examples:
        - Invalid hyperparameter values
        - Conflicting configuration options
        - Missing required configuration fields
        - YAML parsing errors
    """

    pass


class EvaluationError(FNDException):
    """Raised when model evaluation fails.

    Examples:
        - Metric computation errors
        - Invalid evaluation dataset
        - Missing ground truth labels
        - Incompatible model output format
    """

    pass


class ExplainabilityError(FNDException):
    """Raised when explainability analysis fails.

    Examples:
        - SHAP/LIME library not available
        - Incompatible model for explanation
        - Explanation computation timeout
    """

    pass
