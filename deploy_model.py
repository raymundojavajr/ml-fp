import os
import json
import pickle
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List, Dict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Load the trained model
model_path = "src/models/trained_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Load reference dataset for drift detection
reference_path = "data/processed/predictive_maintenance_processed.csv"
reference_data = pd.read_csv(reference_path)

app = FastAPI()

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
async def predict_endpoint(input_data: List[ModelInput]):
    try:
        input_features = pd.DataFrame([data.dict() for data in input_data])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e.json()}")

    with mlflow.start_run():
        for col in input_features.columns:
            mlflow.log_param(col, input_features.iloc[0][col])  

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("n_estimators", 50)

        try:
            predictions = model.predict(input_features)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

        mlflow.log_metric("prediction", predictions[0])

        true_labels = [0] * len(predictions)
        f1 = f1_score(true_labels, predictions, average="macro")
        acc = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="macro", zero_division=1)
        recall = recall_score(true_labels, predictions, average="macro", zero_division=1)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

    return {"predictions": predictions.tolist()}


def convert_numpy_types(obj):
    """Converts NumPy data types to native Python types for JSON serialization."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

@app.get("/model")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail=error_message)

    try:
        # Load model hyperparameters
        hyperparameters = {
            "model_type": "XGBoost",
            "learning_rate": 0.1,
            "max_depth": 5,
            "n_estimators": 100,
        }

        # Feature Importance (Ensure model has this attribute)
        if hasattr(model, "feature_importances_"):
            feature_importance = {
                f"Feature_{idx}": convert_numpy_types(importance)
                for idx, importance in enumerate(model.feature_importances_)
            }
        else:
            feature_importance = {}

        return {
            "model_hyperparameters": hyperparameters,
            "important_features": feature_importance,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {e}")