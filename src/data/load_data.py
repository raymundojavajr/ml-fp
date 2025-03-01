# src/data/load_data.py

from pathlib import Path
import pandas as pd

def find_root(marker="README.md"):
    """Finds the project root by searching for a marker file (default: README.md)."""
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Marker file {marker} not found in any parent directory of {current_dir}")

def load_data(find_relative_path="data/raw/predictive_maintenance.csv"):
    """Loads a CSV file from a path relative to the project root."""
    root = find_root()
    data_path = root / find_relative_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully. Here's a preview:")
    print(df.head())
