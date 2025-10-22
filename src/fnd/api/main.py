import os
from typing import Any, Dict, List, Optional, Union, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fnd.models.utils import create_classification_pipeline


app = FastAPI(title="Fake News Detector API")

_clf = None
_model_dir = None


class PredictRequest(BaseModel):
    text: str
    model_dir: Optional[str] = None


def _ensure_pipeline(model_dir: Optional[str] = None):
    """Load pipeline using centralized utility."""
    global _clf, _model_dir
    if _clf is not None:
        return _clf
    model_dir = model_dir or os.environ.get("MODEL_DIR")
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError("MODEL_DIR is not set or invalid. Provide via env or request body.")
    _model_dir = model_dir
    _clf = create_classification_pipeline(
        model_dir=model_dir,
        max_length=256,
        device=None,  # Auto-detect
        return_all_scores=True
    )
    return _clf


@app.get("/")
def root():
    return {
        "message": "Fake News Detector API",
        "endpoints": {
            "health": "/healthz",
            "predict": "/predict (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must be non-empty")
    clf = _ensure_pipeline(req.model_dir)
    outputs = clf(req.text, truncation=True, max_length=256)

    # Type-safe handling of pipeline output
    if isinstance(outputs, list) and len(outputs) > 0:
        # outputs can be either List[Dict[str, Any]] (top-1) or List[List[Dict[str, Any]]] (all scores)
        first_item: Union[Dict[str, Any], List[Dict[str, Any]]] = outputs[0]
        if isinstance(first_item, dict):
            output_list: List[Dict[str, Any]] = [cast(Dict[str, Any], first_item)]
        else:
            output_list = cast(List[Dict[str, Any]], first_item)
        # Sort by label for deterministic order
        sorted_outputs = sorted(output_list, key=lambda x: str(x.get("label", "")))
        top = max(sorted_outputs, key=lambda x: float(x.get("score", 0)))
        return {"scores": sorted_outputs, "prediction": top}

    raise HTTPException(status_code=500, detail="Invalid model output")
