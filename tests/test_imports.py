"""Consolidated import/smoke tests for all modules."""


def test_import_training_modules():
    import fnd.training.metrics  # noqa: F401
    import fnd.training.train  # noqa: F401


def test_import_factory():
    from fnd.models.factory import load_model_and_tokenizer  # noqa: F401
