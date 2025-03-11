import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow

# Load the model
model_path = "src/models/trained_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
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

@app.post("/predict")
def predict_endpoint(data: List[ModelInput]):
    """Make predictions and log results to MLflow."""

    # Convert input data to Pandas DataFrame
    input_df = pd.DataFrame([item.dict() for item in data])

    # Make predictions
    predictions = model.predict(input_df)

    # Log input features & predictions to MLflow
    with mlflow.start_run():
        mlflow.log_params(data[0].dict())  # Log first input’s features
        mlflow.log_metric("prediction", predictions[0])  # Log first prediction

    return {"predictions": predictions.tolist()}
