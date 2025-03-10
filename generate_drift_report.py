import pandas as pd
import mlflow
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from src.data.load_data import load_processed_data  
def generate_drift_report(training_data_path="data/processed/training_data.csv"):
    # Load training and prediction data
    training_data = pd.read_csv(training_data_path)
    prediction_data = load_processed_data()  # Replace with how you load prediction data
    
    # Define the column mapping
    column_mapping = ColumnMapping(
        target="Target",  # Replace with your actual target column name
        numerical_features=["Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"],
        categorical_features=["Type_encoded", "Product_ID_encoded", "Failure_Type_encoded"]  # Replace with your actual features
    )

    # Create the Evidently Dashboard to display drift
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(training_data, prediction_data, column_mapping)
    
    # Save the drift report
    drift_report_path = "drift_report.html"
    dashboard.save(drift_report_path)
    print(f"Drift report saved as '{drift_report_path}'.")

    # Log drift report as an artifact in MLflow
    mlflow.log_artifact(drift_report_path)
    print("Drift report logged to MLflow.")

# Run the function to generate the drift report
generate_drift_report()
