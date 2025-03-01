# pipelines/assets.py

from dagster import asset, AssetMaterialization, Output
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
    # Load raw data
    df = load_data()
    
    # Define data columns
    categorical_cols, _ = define_data_columns()
    
    # Process data: encode, drop original categorical columns, and clean names
    df, _ = encode_categorical_columns(df, categorical_cols)
    df = drop_original_categorical(df, categorical_cols)
    df = clean_column_names(df)
    
    # Save the processed data for consistency
    save_processed_data(df)
    
    # Yield materialization event (for Dagster tracking)
    yield AssetMaterialization(
        asset_key="processed_data_asset",
        description="The processed predictive maintenance data saved to CSV."
    )
    
    # Yield the processed data as output named "result"
    yield Output(df, "result")
