import os
import json
import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference dataset
reference_path = "data/processed/predictive_maintenance_processed.csv"
reference_data = pd.read_csv(reference_path)

# Standardize column names
reference_data.columns = reference_data.columns.str.strip().str.replace(" ", "_").str.lower()

# Drop "Target" column if it exists
if "target" in reference_data.columns:
    reference_data = reference_data.drop(columns=["target"])

app = FastAPI()

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

@app.post("/drift")
async def detect_drift(input_data: List[ModelInput]):
    input_features = pd.DataFrame([data.dict() for data in input_data])
    
    # Standardize column names
    input_features.columns = input_features.columns.str.strip().str.replace(" ", "_").str.lower()
    
    # Drop "Target" column if present
    if "target" in input_features.columns:
        input_features = input_features.drop(columns=["target"])
    
    # Ensure column names match reference data
    common_columns = list(set(reference_data.columns) & set(input_features.columns))
    input_features = input_features[common_columns]

    # Generate Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=input_features)
    drift_results = drift_report.as_dict()

    # Extract drift summary
    drift_detected = drift_results["metrics"][0]["result"].get("dataset_drift", False)
    drift_score = drift_results["metrics"][0]["result"].get("share_drifted_features", 0.0)

    # Save drift report in JSON and HTML format
    artifact_dir = "mlflow_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    drift_report_json_path = os.path.join(artifact_dir, "drift_report.json")
    drift_report_html_path = os.path.join(artifact_dir, "drift_report.html")

    # Save JSON Report
    with open(drift_report_json_path, "w") as f:
        json.dump(drift_results, f)

    # Save HTML Report
    drift_report.save_html(drift_report_html_path)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_artifact(drift_report_json_path)
        mlflow.log_artifact(drift_report_html_path)

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "drift_report_html_path": drift_report_html_path
    }
