# src/process_data.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .load_data import load_data, find_root  # Import from the load_data module in the same package

def define_data_columns():
    """Returns lists of categorical and numerical column names."""
    categorical_cols = ["Type", "Product ID", "Failure Type"]  # Categorical features.
    numerical_cols = [  # Numerical features.
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    return categorical_cols, numerical_cols

def encode_categorical_columns(df, categorical_cols):
    """Encodes categorical columns with LabelEncoder and returns the updated DataFrame and encoder dictionary."""
    le_dict = {}  # Dictionary to store LabelEncoder for each categorical column.
    for col in categorical_cols:
        le = LabelEncoder()  # Initialize a new LabelEncoder.
        df[col + "_encoded"] = le.fit_transform(df[col])  # Create a new encoded column.
        le_dict[col] = le  # Store the encoder for potential inverse-transforming.
    return df, le_dict

def save_processed_data(df, relative_path="data/processed/predictive_maintenance_processed.csv"):
    """Saves the processed DataFrame to a CSV file at the given relative path."""
    root = find_root()  # Get the project root using the marker file.
    save_path = root / relative_path  # Construct the full path for the processed data.
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist.
    df.to_csv(save_path, index=False)  # Save the DataFrame without row indices.
    print(f"Processed data saved to: {save_path}")

if __name__ == "__main__":
    # Load the raw data from data/raw/predictive_maintenance.csv
    df = load_data()
    
    # Define the categorical and numerical columns
    categorical_cols, numerical_cols = define_data_columns()
    
    # Process the data by encoding the categorical columns
    df, le_dict = encode_categorical_columns(df, categorical_cols)
    
    # Print a preview of the processed DataFrame
    print("Processed DataFrame preview:")
    print(df.head())
    
    # Save the processed data to data/processed/
    save_processed_data(df)
