import logging
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow
from src.models.predict_model import predict  # Ensure this import exists

# Initialize FastAPI
app = FastAPI()

# ✅ Enable Logging
logging.basicConfig(level=logging.INFO)

# ✅ Load the best model from MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = "Champion_Model"
model_version = 1  # Update this if necessary
best_model_uri = f"models:/{model_name}/{model_version}"

try:
    model = mlflow.xgboost.load_model(best_model_uri)
    logging.info(f"Champion Model Loaded Successfully from {best_model_uri}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Define input schema
class ModelInput(BaseModel):
    UDI: int
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: int
    Torque_Nm: float
    Tool_wear_min: int
    Type_encoded: int
    Product_ID_encoded: int
    Failure_Type_encoded: int

@app.post("/predict")
def predict_endpoint(data: List[ModelInput]):
    """Make predictions using the best model."""
    
    try:
        if model is None:
            return {"error": "Model not loaded"}

        # Convert JSON input to DataFrame
        input_data = pd.DataFrame([item.dict() for item in data])

        # Make predictions
        predictions = predict(model, input_data)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        logging.error(traceback.format_exc())  # Print full error trace
        return {"error": "Internal Server Error"}
