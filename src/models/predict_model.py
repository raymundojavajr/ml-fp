# src/models/predict_model.py

import joblib
import pandas as pd
from src.data.load_data import find_root

def load_model(model_path):
    """Loads the trained model from the specified file path."""
    return joblib.load(model_path)

def load_processed_data(file_relative_path="data/processed/predictive_maintenance_processed.csv"):
    """Loads the processed data CSV from a path relative to the project root."""
    root = find_root()
    file_path = root / file_relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    return pd.read_csv(file_path)

def clean_feature_names(df):
    """Cleans column names by removing brackets and replacing spaces with underscores."""
    df.columns = df.columns.str.replace(r'[\[\]]', '', regex=True).str.replace(' ', '_')
    return df

def predict(model, X):
    """Generates predictions for the provided features using the model."""
    return model.predict(X)

def main():
    root = find_root()
    model_path = root / "src" / "models" / "trained_model.pkl"

    # Load the saved model
    model = load_model(model_path)

    # Load processed data and clean feature names
    df = load_processed_data()
    df = clean_feature_names(df)

    # Prepare features (assume target column is "Target" and drop it)
    X = df.drop(columns=["Target"])

    # Generate predictions
    predictions = predict(model, X)
    print("Predictions preview:")
    print(predictions[:10])

if __name__ == "__main__":
    main()
