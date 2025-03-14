import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.process_data import (
    define_data_columns,
    encode_categorical_columns,
    drop_original_categorical,
    clean_column_names,
    save_processed_data
)

def test_define_data_columns():
    """Test that define_data_columns returns the expected categorical and numerical column lists."""
    categorical_cols, numerical_cols = define_data_columns()
    assert categorical_cols == ["Type", "Product ID", "Failure Type"]
    assert numerical_cols == [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

def test_encode_categorical_columns():
    """Test that encode_categorical_columns correctly encodes categorical columns and returns a dict of encoders."""
    df = pd.DataFrame({
        "Type": ["A", "B", "A"],
        "Product ID": ["P1", "P2", "P1"],
        "Failure Type": ["F1", "F2", "F1"]
    })
    categorical_cols = ["Type", "Product ID", "Failure Type"]
    df_encoded, le_dict = encode_categorical_columns(df, categorical_cols)

    # The encoded column names will retain spaces, e.g. "Product ID_encoded"
    assert "Type_encoded" in df_encoded.columns
    assert "Product ID_encoded" in df_encoded.columns
    assert "Failure Type_encoded" in df_encoded.columns
    assert len(le_dict) == 3

def test_drop_original_categorical():
    """Test that drop_original_categorical removes the original categorical columns from the DataFrame."""
    df = pd.DataFrame({
        "Type": ["A", "B", "A"],
        "Product ID": ["P1", "P2", "P1"],
        "Failure Type": ["F1", "F2", "F1"],
        "Type_encoded": [0, 1, 0],
        "Product ID_encoded": [0, 1, 0],
        "Failure Type_encoded": [0, 1, 0]
    })
    categorical_cols = ["Type", "Product ID", "Failure Type"]
    df_dropped = drop_original_categorical(df, categorical_cols)

    assert "Type" not in df_dropped.columns
    assert "Product ID" not in df_dropped.columns
    assert "Failure Type" not in df_dropped.columns

def test_clean_column_names():
    """Test that clean_column_names correctly reformats column names by removing spaces and brackets."""
    df = pd.DataFrame({
        "Air temperature [K]": [1, 2, 3],
        "Process temperature [K]": [4, 5, 6]
    })
    df_cleaned = clean_column_names(df)

    assert "Air_temperature_K" in df_cleaned.columns
    assert "Process_temperature_K" in df_cleaned.columns

@patch("src.data.process_data.find_root")
@patch("pandas.DataFrame.to_csv")
def test_save_processed_data(mock_to_csv, mock_find_root):
    """Test that save_processed_data calls find_root and to_csv exactly once to save the DataFrame."""
    # Set up the patch for find_root to return a dummy path object.
    mock_find_root.return_value = MagicMock()
    df = pd.DataFrame({"col1": [1, 2, 3]})
    save_processed_data(df, "test_path.csv")

    mock_to_csv.assert_called_once()
    mock_find_root.assert_called_once()
