from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from app.utils import save_uploaded_csv, run_prediction
import subprocess
import sys
from uuid import uuid4
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)
router = APIRouter()

class PredictionResult(BaseModel):
    sequence: str
    predicted_species: str
    confidence_score: float
    metadata: Dict[str, Any]

class DiversityMetrics(BaseModel):
    shannon_index: float
    simpson_index: float
    species_richness: int

class UploadResponse(BaseModel):
    status: str
    predictions: List[PredictionResult]
    summary: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Job store (in‑memory)
JOBS: dict = {}  # job_id -> {status, logs, error, started_at, finished_at, input_file}

def _append_log(job_id: str, message: str):
    job = JOBS.get(job_id)
    if job is None: return
    timestamp = datetime.now().strftime("%H:%M:%S")
    job["logs"].append(f"[{timestamp}] {message}")

def _run_prediction_job(job_id: str, predict_script: str):
    job = JOBS[job_id]
    try:
        _append_log(job_id, "Starting prediction process.")
        proc = subprocess.Popen(
            [sys.executable, predict_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _append_log(job_id, line)
        rc = proc.wait()
        if rc == 0:
            job["status"] = "completed"
            job["finished_at"] = datetime.now().isoformat()
            _append_log(job_id, "✅ Prediction script completed successfully.")
        else:
            job["status"] = "error"
            job["finished_at"] = datetime.now().isoformat()
            _append_log(job_id, f"❌ Prediction script failed (exit {rc}).")
    except Exception as e:
        job["status"] = "error"
        job["finished_at"] = datetime.now().isoformat()
        _append_log(job_id, f"Exception: {e}")

@router.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    logger.info(f"📥 Received upload request for file: {file.filename}")
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")

        input_dir = r"C:/Users/loq/OneDrive/Desktop/SIH/BioTrace/edna-backend/temp"
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, "predict.csv")
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)

        # Basic validation
        df = pd.read_csv(input_path)
        if 'sequence' not in df.columns:
            raise ValueError("CSV must contain 'sequence' column")
        if df.empty:
            raise ValueError("CSV file is empty")
        if df['sequence'].isnull().any():
            raise ValueError("Sequence column contains empty values")

        logger.info(f"✅ CSV validation passed: {len(df)} sequences found")

        predict_script = r"C:/Users/loq/OneDrive/Desktop/SIH/.venv/predict_csv.py"

        # Create job
        job_id = str(uuid4())
        JOBS[job_id] = {
            "status": "running",
            "logs": [],
            "error": None,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "input_file": input_path
        }
        _append_log(job_id, f"📥 File received: {file.filename}")
        _append_log(job_id, f"✅ CSV validation passed ({len(df)} sequences)")
        _append_log(job_id, "⚡ Running prediction script...")

        # Start background thread
        thread = threading.Thread(
            target=_run_prediction_job,
            args=(job_id, predict_script),
            daemon=True
        )
        thread.start()

        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Upload accepted. Prediction running."
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Upload failed.")
        raise HTTPException(status_code=500, detail="Server error")

@router.get("/status/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "logs": job["logs"][-200:],  # last 200 lines
        "started_at": job["started_at"],
        "finished_at": job["finished_at"]
    }

import math
from fastapi.responses import JSONResponse

def sanitize_json(obj):
    """Recursively replace NaN/Infinity with None so it's JSON safe."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


@router.get("/sample-data/")
async def get_sample_data():
    """
    Return latest prediction results from root-level prediction_results folder.
    Provides flattened structure: results, alpha_diversity, beta_diversity, rarefaction_curve, visualizations.
    """
    try:
        root_dir = Path(__file__).resolve().parents[3]
        results_file = Path("C:/Users/loq/OneDrive/Desktop/SIH/BioTrace/prediction_results/results_export.json")

        if not results_file.exists():
            logger.warning(f"Prediction results file not found: {results_file}")
            raise HTTPException(status_code=404, detail="Prediction results not found. Run a prediction first.")

        with results_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        results = raw.get("results") or []
        alpha = raw.get("alpha_diversity") or {}
        beta = raw.get("beta_diversity") or {}
        rarefaction = raw.get("rarefaction_curve") or {"x": [], "y": []}
        viz = raw.get("visualization_files") or []

        payload = {
            "status": "ok",
            "source_file": str(results_file),
            "results_count": len(results),
            "results": results,
            "alpha_diversity": alpha,
            "beta_diversity": beta,
            "rarefaction_curve": rarefaction,
            "visualizations": viz
        }

        # ✅ sanitize to avoid NaN/Infinity errors
        safe_payload = sanitize_json(payload)

        return JSONResponse(content=safe_payload)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Corrupted results file.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error reading sample data.")
        raise HTTPException(status_code=500, detail="Internal server error.")
