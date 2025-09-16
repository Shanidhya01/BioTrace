# routes/api.py
import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from predict_csv import predict_csv

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Run predictions
        output_csv = os.path.join(UPLOAD_DIR, "predicted_results.csv")
        results_df = predict_csv(file_path, output_csv)

        # Convert DataFrame to JSON
        results_json = results_df.to_dict(orient="records")

        return {"message": "✅ Prediction complete", "results": results_json}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
