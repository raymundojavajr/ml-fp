import pandas as pd
from src.data.process_data import (
    define_data_columns,
    encode_categorical_columns,
    drop_original_categorical,
    clean_column_names
)

def test_define_data_columns():
    # Verify that the expected categorical and numerical columns are returned.
    cat_cols, num_cols = define_data_columns()
    assert cat_cols == ["Type", "Product ID", "Failure Type"]
    assert num_cols == [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

def test_encode_categorical_columns():
    # Create a sample DataFrame with categorical data.
    df = pd.DataFrame({
        "Type": ["A", "B", "A"],
        "Product ID": ["P1", "P2", "P1"],
        "Failure Type": ["F1", "F2", "F1"]
    })
    df_encoded, le_dict = encode_categorical_columns(df, ["Type", "Product ID", "Failure Type"])

    # Check that encoded columns exist.
    for col in ["Type", "Product ID", "Failure Type"]:
        assert f"{col}_encoded" in df_encoded.columns

    # Verify that the label encoders contain the expected classes.
    assert list(le_dict["Type"].classes_) == ["A", "B"]
    assert list(le_dict["Product ID"].classes_) == ["P1", "P2"]
    assert list(le_dict["Failure Type"].classes_) == ["F1", "F2"]

def test_drop_original_categorical():
    # Create a DataFrame with both original and encoded categorical columns.
    df = pd.DataFrame({
        "Type": ["A", "B", "A"],
        "Product ID": ["P1", "P2", "P1"],
        "Failure Type": ["F1", "F2", "F1"],
        "Type_encoded": [0, 1, 0],
        "Product ID_encoded": [0, 1, 0],
        "Failure Type_encoded": [0, 1, 0]
    })
    df_new = drop_original_categorical(df, ["Type", "Product ID", "Failure Type"])

    # Ensure that the original categorical columns have been dropped.
    for col in ["Type", "Product ID", "Failure Type"]:
        assert col not in df_new.columns

def test_clean_column_names():
    # Create a DataFrame with columns that have spaces and special characters.
    df = pd.DataFrame({
        "Air temperature [K]": [300, 310],
        "Process temperature [K]": [500, 510]
    })
    df_clean = clean_column_names(df)

    # Check that the cleaned column names replace spaces and brackets appropriately.
    assert "Air_temperature_K" in df_clean.columns
    assert "Process_temperature_K" in df_clean.columns
