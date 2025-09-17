import os
import sys
import math
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base dirs
BASE_DIR = Path(__file__).resolve().parents[2]          # .../SIH/BioTrace
PROJECT_ROOT = Path(__file__).resolve().parents[3]       # .../SIH
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = BASE_DIR / "edna-backend" / "temp"
PRED_RESULTS_DIR = PROJECT_ROOT / "prediction_results"
PRED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Dynamic loader for predict_csv (supports multiple locations / signatures)
# -----------------------------------------------------------------------------
def _load_predict_fn():
    """
    Tries these in order:
      1. Local package import: app.predict_csv.predict_csv (if refactored inside app)
      2. Relative sibling: app/predict_csv.py
      3. Root-level predict_csv.py
      4. .venv/predict_csv.py (as you had earlier)
    """
    candidates = [
        ("app.predict_csv", None),
        (None, BASE_DIR / "predict_csv.py"),
        (None, PROJECT_ROOT / "predict_csv.py"),
        (None, PROJECT_ROOT / ".venv" / "predict_csv.py"),
    ]

    for mod_name, path in candidates:
        try:
            if mod_name:
                module = __import__(mod_name, fromlist=["predict_csv"])
                if hasattr(module, "predict_csv"):
                    logger.info(f"Loaded predict_csv from module {mod_name}")
                    return getattr(module, "predict_csv")
            else:
                if path and path.exists():
                    if str(path.parent) not in sys.path:
                        sys.path.insert(0, str(path.parent))
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("predict_runtime", str(path))
                    module = importlib.util.module_from_spec(spec)
                    assert spec.loader
                    spec.loader.exec_module(module)  # type: ignore
                    if hasattr(module, "predict_csv"):
                        logger.info(f"Loaded predict_csv from file {path}")
                        return getattr(module, "predict_csv")
        except Exception as e:
            logger.debug(f"Attempt failed for {mod_name or path}: {e}")
    raise RuntimeError("predict_csv function not found in any expected location")

# -----------------------------------------------------------------------------
# JSON sanitization
# -----------------------------------------------------------------------------
def _sanitize(obj: Any):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

# -----------------------------------------------------------------------------
# File save helper
# -----------------------------------------------------------------------------
def save_uploaded_csv(file) -> str:
    """
    Save uploaded file-like object (FastAPI UploadFile or raw) to temp/predict.csv
    Returns absolute path.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    target = TEMP_DIR / "predict.csv"
    content = file.read() if hasattr(file, "read") else file
    if isinstance(content, str):
        content = content.encode()
    with open(target, "wb") as f:
        f.write(content)
    logger.info(f"✅ Uploaded file saved: {target}")
    return str(target)

# -----------------------------------------------------------------------------
# Prediction runner
# -----------------------------------------------------------------------------
def run_prediction(input_file: str) -> Dict[str, Any]:
    """
    Executes predict_csv and normalizes its output to a consistent API shape.
    Expected: predict_csv returns a pandas DataFrame OR path to CSV.
    """
    predict_fn = _load_predict_fn()

    # Determine supported params
    import inspect
    sig = inspect.signature(predict_fn)
    params = sig.parameters

    kwargs = {}
    if "input_csv" in params:
        kwargs["input_csv"] = input_file
    elif "input_path" in params:
        kwargs["input_path"] = input_file
    else:
        # Fallback assume first positional is input
        pass

    # Output handling
    if "output_dir" in params:
        kwargs["output_dir"] = str(PRED_RESULTS_DIR)
    elif "output_csv" in params:
        kwargs["output_csv"] = str(PRED_RESULTS_DIR / "prediction_results.csv")

    # Optional training file
    training_csv = DATA_DIR / "fasta_parsed.csv"
    if training_csv.exists():
        if "training_csv_path" in params:
            kwargs["training_csv_path"] = str(training_csv)
        elif "training_csv" in params:
            kwargs["training_csv"] = str(training_csv)

    logger.info(f"⚡ Running predict_csv with args: {kwargs}")

    result = predict_fn(**kwargs)

    # Normalize DataFrame
    if isinstance(result, pd.DataFrame):
        df = result
    elif isinstance(result, str) and os.path.isfile(result):
        df = pd.read_csv(result)
    else:
        raise RuntimeError("predict_csv returned unsupported type (need DataFrame or CSV path)")

    # Column normalization
    col_map = {
        "predicted_species": ["predicted_species", "species", "taxon"],
        "confidence": ["confidence", "confidence_score", "score"],
        "status": ["status", "novelty"],
        "cluster_id": ["cluster_id", "cluster", "clusterID"]
    }
    norm_cols = {}
    for target, options in col_map.items():
        for opt in options:
            if opt in df.columns:
                norm_cols[target] = opt
                break

    # Build list of dict rows
    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            "predicted_species": row.get(norm_cols.get("predicted_species"), None),
            "confidence": float(row.get(norm_cols.get("confidence"), 0.0)) if pd.notna(row.get(norm_cols.get("confidence"))) else None,
            "status": row.get(norm_cols.get("status"), None),
            "cluster_id": row.get(norm_cols.get("cluster_id"), None),
        })

    payload = {
        "status": "success",
        "results": predictions,
        "results_count": len(predictions),
        "alpha_diversity": {},          # Fill in when available
        "beta_diversity": {},
        "rarefaction_curve": {"x": [], "y": []},
        "visualizations": []
    }

    safe_payload = _sanitize(payload)
    logger.info(f"✅ Prediction complete. Records: {safe_payload['results_count']}")
    return safe_payload