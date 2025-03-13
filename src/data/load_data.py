# src/data/load_data.py

from pathlib import Path
import pandas as pd

def find_root(marker="README.md"):
    """Find project root using marker file."""
    # Skip searching for README.md if you're in the Docker container
    try:
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / marker).exists():
                return parent
        # If README.md isn't found, you can fallback to the /app directory
        print("README.md not found. Using /app as the root directory.")
        return Path('/app')  # Specify the working directory for your container here
    except Exception as e:
        raise FileNotFoundError(f"Error locating root directory: {str(e)}")


def load_data(find_relative_path="data/raw/predictive_maintenance.csv"):
    """Load CSV file from a path relative to project root."""
    try:
        root = find_root()
        data_path = root / find_relative_path
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def load_processed_data(find_relative_path="data/processed/predictive_maintenance_processed.csv"):
    """Load processed CSV file from a path relative to project root."""
    try:
        root = find_root()
        data_path = root / find_relative_path
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data file not found at {data_path}")
        return pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Error loading processed data: {e}")


if __name__ == "__main__":
    try:
        df_raw = load_data()
        print("Raw data loaded successfully. Preview:")
        print(df_raw.head())
        df_processed = load_processed_data()
        print("Processed data loaded successfully. Preview:")
        print(df_processed.head())
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
