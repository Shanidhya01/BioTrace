import os
import pandas as pd
import logging
from predict_csv import predict_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")

def save_uploaded_csv(file) -> str:
    """Save uploaded CSV as temporary file"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Create a temporary file path
    import uuid
    temp_filename = f"user_input_{uuid.uuid4().hex}.csv"
    temp_filepath = os.path.join(TEMP_DIR, temp_filename)
    
    # Read the file content and save it
    content = file.read()
    with open(temp_filepath, "wb") as f:
        f.write(content)
    
    logger.info(f"✅ Uploaded file saved to: {temp_filepath}")
    return temp_filepath

def run_prediction(input_file) -> dict:
    """Run ML prediction and return result as dict"""
    try:
        # Run prediction
        results_df = predict_csv(input_file, output_csv=None)
        
        # Convert to dictionary format
        predictions = results_df.to_dict(orient="records")
        
        # Clean up temporary file
        if TEMP_DIR in input_file:
            os.remove(input_file)
            logger.info(f"✅ Temporary file cleaned up: {input_file}")
        
        return {
            "status": "success",
            "predictions": predictions,
            "count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in run_prediction: {str(e)}")
        # Clean up temporary file even if prediction fails
        if TEMP_DIR in input_file and os.path.exists(input_file):
            os.remove(input_file)
            logger.info(f"✅ Temporary file cleaned up after error: {input_file}")
        raise