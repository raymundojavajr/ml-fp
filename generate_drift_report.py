import pandas as pd
import streamlit as st
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

# Load your synthetic data for drift testing
synthetic_data = pd.read_csv("data/processed/synthetic_data.csv")

# Here you can load your baseline data (pre-processed original data)
original_data = pd.read_csv("data/processed/predictive_maintenance_processed.csv")

# Initialize the ColumnMapping for drift detection
column_mapping = ColumnMapping(
    target="Target",  # Your target column in the dataset
    numeric_features=["UDI","Air temper", "Process te", "Rotational", "Torque [Nr]", "Tool wear"],
    categorical_features=["Product ID", "Type"],
    datetime_features=[]
)

# Initialize Evidently dashboard for drift detection
dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(original_data, synthetic_data)

# Display the drift report
st.title("Data Drift Detection Report")
st.write("This is the drift report for comparing synthetic data with original data.")
dashboard.show()
