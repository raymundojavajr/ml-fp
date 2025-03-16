import os
<<<<<<< HEAD
import json
=======
>>>>>>> origin/main
import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
<<<<<<< HEAD
from typing import List
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
=======
from typing import List, Optional
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
import json
>>>>>>> origin/main

# Load reference dataset
reference_path = "data/processed/predictive_maintenance_processed.csv"
reference_data = pd.read_csv(reference_path)

# Standardize column names
reference_data.columns = reference_data.columns.str.strip().str.replace(" ", "_").str.lower()

<<<<<<< HEAD
# Drop "Target" column if it exists
if "target" in reference_data.columns:
    reference_data = reference_data.drop(columns=["target"])

app = FastAPI()

# Define input schema
=======
# Ensure "target" is removed from the reference dataset
if "target" in reference_data.columns:
    reference_data_filtered = reference_data.drop(columns=["target"])
else:
    reference_data_filtered = reference_data.copy()

app = FastAPI()

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
<<<<<<< HEAD
    Failure_Type_encoded: int
=======
    Failure_Type_encoded: Optional[int] = None  # Optional to support unlabeled requests
>>>>>>> origin/main

@app.post("/drift")
async def detect_drift(input_data: List[ModelInput]):
    input_features = pd.DataFrame([data.dict() for data in input_data])
<<<<<<< HEAD
    
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
=======
    input_features.columns = input_features.columns.str.strip().str.replace(" ", "_").str.lower()

    # Ensure "target" is removed from input features
    if "target" in input_features.columns:
        input_features = input_features.drop(columns=["target"])

    # Ensure column consistency between reference and current input
    common_columns = list(set(reference_data_filtered.columns) & set(input_features.columns))
    input_features = input_features[common_columns]
    reference_data_filtered_aligned = reference_data_filtered[common_columns]

    # Define column mapping for classification
    column_mapping = ColumnMapping()
    column_mapping.target = "failure_type_encoded"  # Target for classification
    column_mapping.prediction = "failure_type_encoded"  # Predictions are assumed to be the same field for evaluation

    # Generate drift and classification reports
    drift_report = Report(metrics=[DataDriftPreset()])
    classification_report = Report(metrics=[ClassificationPreset()])

    drift_report.run(
        reference_data=reference_data_filtered_aligned,
        current_data=input_features,
        column_mapping=column_mapping
    )

    classification_report.run(
        reference_data=reference_data_filtered_aligned,
        current_data=input_features,
        column_mapping=column_mapping
    )

    drift_results = drift_report.as_dict()
    classification_results = classification_report.as_dict()

    drift_detected = drift_results["metrics"][0]["result"].get("dataset_drift", False)
    drift_score = drift_results["metrics"][0]["result"].get("share_drifted_features", 0.0)

    # Save drift and classification reports as artifacts
    artifact_dir = "mlflow_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    drift_report_path = os.path.join(artifact_dir, "drift_report.json")
    classification_report_path = os.path.join(artifact_dir, "classification_report.json")

    with open(drift_report_path, "w") as f:
        json.dump(drift_results, f)

    with open(classification_report_path, "w") as f:
        json.dump(classification_results, f)

    with mlflow.start_run():
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_artifact(drift_report_path)
        mlflow.log_artifact(classification_report_path)
>>>>>>> origin/main

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
<<<<<<< HEAD
        "drift_report_html_path": drift_report_html_path
=======
        "drift_report_path": drift_report_path,
        "classification_report_path": classification_report_path
>>>>>>> origin/main
    }
