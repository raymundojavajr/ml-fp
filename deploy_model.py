import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow
import shap
import joblib
import os

# Load the "champion" model from MLflow
def load_model():
    """Load the champion model from MLflow."""
    client = mlflow.tracking.MlflowClient()
    # Retrieve the best model URI from MLflow
    best_model_uri = get_best_model()
    if best_model_uri is None:
        raise HTTPException(status_code=500, detail="Champion model not found in MLflow")
    model = mlflow.xgboost.load_model(best_model_uri)
    return model

# FastAPI App Initialization
app = FastAPI()

# Define the input schema for the /predict endpoint
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
    """Make predictions and log results to MLflow."""

    try:
        # Load the trained model (champion model from MLflow)
        model = load_model()

        # Convert input data to Pandas DataFrame
        input_df = pd.DataFrame([item.dict() for item in data])

        # Make predictions
        predictions = model.predict(input_df)

        # Log input features & predictions to MLflow
        with mlflow.start_run():
            mlflow.log_params(data[0].dict())  # Log first input’s features
            mlflow.log_metric("prediction", predictions[0])  # Log first prediction

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.get("/model")
def model_info():
    """Retrieve information about the currently deployed model."""
    try:
        # Load the model
        model = load_model()

        # Retrieve hyperparameters (assuming these are logged during model training)
        hyperparameters = {
            "learning_rate": model.get_params()["learning_rate"],
            "max_depth": model.get_params()["max_depth"],
            "n_estimators": model.get_params()["n_estimators"]
        }

        # Example input data for SHAP
        sample_input = {
            "UDI": 12345,
            "Air_temperature_K": 300.0,
            "Process_temperature_K": 290.0,
            "Rotational_speed_rpm": 1500,
            "Torque_Nm": 30.0,
            "Tool_wear_min": 25,
            "Type_encoded": 1,
            "Product_ID_encoded": 10,
            "Failure_Type_encoded": 3
        }

        input_df = pd.DataFrame([sample_input])

        # SHAP explanation
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        # Get important features
        important_features = [f"{feat} ({round(val, 3)})" for feat, val in zip(input_df.columns, shap_values.mean(0))]

        # Return model information
        return {
            "model_hyperparameters": hyperparameters,
            "important_features": important_features,
            "input_schema": ModelInput.schema()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

def get_best_model():
    """Fetch the best model from MLflow based on the highest F1 score."""
    client = mlflow.tracking.MlflowClient()
    
    # Search for the best run in the "Predictive Maintenance" experiment
    experiment = client.get_experiment_by_name("Predictive Maintenance")
    
    if not experiment:
        print("Experiment not found. Train a model first!")
        return None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1_score DESC"])

    if runs:
        best_run = runs[0]
        best_model_uri = best_run.info.artifact_uri + "/model"
        print(f"Champion Model Found: {best_model_uri}")
        return best_model_uri
    else:
        print("No trained models found.")
        return None

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.get("/mlflow_test")
def test_mlflow():
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # Try logging a dummy parameter and metric to MLflow to see if it's working
        with mlflow.start_run():
            mlflow.log_param("param1", "value1")
            mlflow.log_metric("metric1", 0.1)
        
        return {"message": "MLflow is connected and working"}
    
    except Exception as e:
        return {"error": f"Error in MLflow: {str(e)}"}
