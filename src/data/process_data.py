# src/data/process_data.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .load_data import load_data, find_root

def define_data_columns():
    """Returns lists of categorical and numerical column names."""
    categorical_cols = ["Type", "Product ID", "Failure Type"]
    numerical_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    return categorical_cols, numerical_cols

def encode_categorical_columns(df, categorical_cols):
    """Encodes categorical columns with LabelEncoder and returns the updated DataFrame and encoder dictionary."""
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def drop_original_categorical(df, categorical_cols):
    """Drops the original categorical columns from the DataFrame."""
    return df.drop(columns=categorical_cols)

def clean_column_names(df):
    """Cleans DataFrame column names by removing brackets and replacing spaces with underscores."""
    df.columns = df.columns.str.replace(r'[\[\]]', '', regex=True).str.replace(' ', '_')
    return df

def save_processed_data(df, relative_path="data/processed/predictive_maintenance_processed.csv"):
    """Saves the processed DataFrame to a CSV file at the given relative path."""
    root = find_root()
    save_path = root / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to: {save_path}")

if __name__ == "__main__":
    # Load raw data
    df = load_data()
    
    # Define columns
    categorical_cols, numerical_cols = define_data_columns()
    
    # Encode categorical columns
    df, le_dict = encode_categorical_columns(df, categorical_cols)
    
    # Drop the original categorical columns
    df = drop_original_categorical(df, categorical_cols)
    
    # Clean column names (e.g., remove brackets, replace spaces)
    df = clean_column_names(df)
    
    print("Processed DataFrame preview:")
    print(df.head())
    
    # Save processed data
    save_processed_data(df)
