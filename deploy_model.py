import os
import json
import pickle
import pandas as pd
import mlflow
<<<<<<< HEAD
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
=======
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Load the trained model
model_path = "src/models/trained_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)
>>>>>>> origin/main

# Load reference dataset for drift detection
reference_path = "data/processed/predictive_maintenance_processed.csv"
reference_data = pd.read_csv(reference_path)

app = FastAPI()

<<<<<<< HEAD
=======
# Define input schema
>>>>>>> origin/main
class ModelInput(BaseModel):
    UDI: int
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: int
    Torque_Nm: float
    Tool_wear_min: int
    Type_encoded: int
    Product_ID_encoded: int
    Failure_Type_encoded: int  # Ensure this feature is included

# 🚀 Predict Endpoint (Maintains Existing Functionality)
@app.post("/predict")
async def predict_endpoint(input_data: List[ModelInput]):
<<<<<<< HEAD
    try:
        input_features = pd.DataFrame([data.dict() for data in input_data])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e.json()}")

    with mlflow.start_run():
        for col in input_features.columns:
            mlflow.log_param(col, input_features.iloc[0][col])  

=======
    input_features = pd.DataFrame([data.dict() for data in input_data])

    with mlflow.start_run():
        # ✅ Log input parameters
        for col in input_features.columns:
            mlflow.log_param(col, input_features.iloc[0][col])  

        # ✅ Log Model Hyperparameters
>>>>>>> origin/main
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("n_estimators", 50)

<<<<<<< HEAD
        try:
            predictions = model.predict(input_features)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

        mlflow.log_metric("prediction", predictions[0])

        true_labels = [0] * len(predictions)
=======
        # ✅ Make Predictions
        predictions = model.predict(input_features)

        # ✅ Log Predictions
        mlflow.log_metric("prediction", predictions[0])

        # ✅ Calculate & Log Metrics (Dummy True Labels for Now)
        true_labels = [0] * len(predictions)  # Replace with actual labels if available
>>>>>>> origin/main
        f1 = f1_score(true_labels, predictions, average="macro")
        acc = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="macro", zero_division=1)
        recall = recall_score(true_labels, predictions, average="macro", zero_division=1)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

    return {"predictions": predictions.tolist()}

<<<<<<< HEAD

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
=======
# 🔥 Drift Detection Endpoint (Separate from Prediction)
@app.post("/drift")
async def drift_endpoint(input_data: List[ModelInput]):
    input_features = pd.DataFrame([data.dict() for data in input_data])

    with mlflow.start_run():
        # ✅ Run Drift Detection
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=input_features)
        drift_results = drift_report.as_dict()

        # Extract drift information
        drift_detected = drift_results["metrics"][0]["result"]["dataset_drift"]
        drift_score = drift_results["metrics"][0]["result"]["share_drifted_features"]

        # ✅ Log Drift Metrics to MLflow
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("drift_score", drift_score)

        # ✅ Log Feature-Specific Drift
        drifted_features = []
        for feature, details in drift_results["metrics"][0]["result"]["drift_by_columns"].items():
            if details["drift_detected"]:
                drifted_features.append(feature)
                mlflow.log_metric(f"{feature}_drift", 1)  # Log 1 if drift detected

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "drifted_features": drifted_features
    }

# ✅ Combined Predict + Drift (Optional, if you want both in one call)
@app.post("/predict_and_drift")
async def predict_and_drift(input_data: List[ModelInput]):
    input_features = pd.DataFrame([data.dict() for data in input_data])

    with mlflow.start_run():
        # ✅ Log input parameters
        for col in input_features.columns:
            mlflow.log_param(col, input_features.iloc[0][col])  

        # ✅ Log Model Hyperparameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("n_estimators", 50)

        # ✅ Make Predictions
        predictions = model.predict(input_features)

        # ✅ Log Predictions
        mlflow.log_metric("prediction", predictions[0])

        # ✅ Calculate & Log Metrics (Dummy True Labels for Now)
        true_labels = [0] * len(predictions)  # Replace with actual labels if available
        f1 = f1_score(true_labels, predictions, average="macro")
        acc = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="macro", zero_division=1)
        recall = recall_score(true_labels, predictions, average="macro", zero_division=1)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # ✅ Run Drift Detection after Prediction
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=input_features)
        drift_results = drift_report.as_dict()

        drift_detected = drift_results["metrics"][0]["result"]["dataset_drift"]
        drift_score = drift_results["metrics"][0]["result"]["share_drifted_features"]

        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("drift_score", drift_score)

        drifted_features = []
        for feature, details in drift_results["metrics"][0]["result"]["drift_by_columns"].items():
            if details["drift_detected"]:
                drifted_features.append(feature)
                mlflow.log_metric(f"{feature}_drift", 1)

    return {
        "predictions": predictions.tolist(),
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "drifted_features": drifted_features
    }
>>>>>>> origin/main
