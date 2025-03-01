from pathlib import Path
import pandas as pd


def find_root(marker="README.md"):
    """Finds the project root by searching for a marker file (default: README.md)."""
    # Get the current working directory
    current_dir = Path.cwd()
    # Check the current directory and all parent directories for the marker file
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / marker).exists():
            return parent  # Return the directory if marker is found
    # Raise an error if the marker file is not found in any parent directory
    raise FileNotFoundError(
        f"Marker file {marker} not found in any parent directory of {current_dir}"
    )


def load_data(find_relative_path="data/raw/predictive_maintenance.csv"):
    """Loads a CSV file from a path relative to the project root."""
    # Find the project root using the marker file
    root = find_root()
    # Construct the full path to the data file
    data_path = root / find_relative_path
    # Check if the data file exists; if not, raise an error
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    # Load and return the CSV file as a DataFrame
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully. Here's a preview:")
    print(df.head())