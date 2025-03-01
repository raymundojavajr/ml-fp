# pipelines/assets.py

from dagster import asset, AssetMaterialization
from src.data.load_data import load_data
from src.data.process_data import (
    define_data_columns,
    encode_categorical_columns,
    drop_original_categorical,
    clean_column_names,
    save_processed_data,
)

@asset
def processed_data_asset():
    """
    Materializes the processed predictive maintenance data asset.

    Loads raw data, processes it using existing functions, saves the processed CSV,
    and returns the processed DataFrame.
    """
    # Load raw data using the existing function.
    df = load_data()
    
    # Define columns.
    categorical_cols, _ = define_data_columns()
    
    # Process the data:
    # - Encode categorical columns
    # - Drop original categorical columns to ensure only numeric data remains
    # - Clean column names for compatibility with downstream tools (e.g., XGBoost)
    df, _ = encode_categorical_columns(df, categorical_cols)
    df = drop_original_categorical(df, categorical_cols)
    df = clean_column_names(df)
    
    # Save the processed data using the existing save function.
    save_processed_data(df)
    
    # Materialize the asset so Dagster can track it.
    yield AssetMaterialization(
        asset_key="processed_data_asset",
        description="The processed predictive maintenance data saved to CSV."
    )
    
    return df
