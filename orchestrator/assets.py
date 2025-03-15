# orchestrator/assets.py

import pandas as pd
from dagster import asset, AssetMaterialization, Output, AssetIn
from src.data.load_data import load_data
from src.data.process_data import (
    define_data_columns,
    encode_categorical_columns,
    drop_original_categorical,
    clean_column_names,
    save_processed_data,
)
from src.data.split_data import split_data
from src.models.train_model import train_model, save_model
from src.models.predict_model import predict
from src.models.evaluate_model import evaluate

@asset(
    description="Loads and processes raw predictive maintenance data, then saves the processed CSV.",
)
def processed_data_asset(context) -> pd.DataFrame:
    """Asset that materializes the processed predictive maintenance data."""
    context.log.info("Loading raw data...")
    df = load_data()  # Loads from data/raw/predictive_maintenance.csv

    categorical_cols, _ = define_data_columns()
    df, _ = encode_categorical_columns(df, categorical_cols)
    df = drop_original_categorical(df, categorical_cols)
    df = clean_column_names(df)

    # Save processed data to CSV
    save_processed_data(df)
    context.log.info("Processed data saved.")

    yield AssetMaterialization(
        asset_key="processed_data_asset",
        description="Processed predictive maintenance data CSV."
    )
    yield Output(df)

@asset(
    ins={"processed_data_asset": AssetIn()},
    description="Trains a model using processed data, then saves the model to disk.",
)
def trained_model_asset(context, processed_data_asset: pd.DataFrame):
    """Asset that trains a model on processed data and saves the model file."""
    context.log.info("Splitting data for training...")
    X_train, X_test, y_train, y_test = split_data(processed_data_asset, target="Target")

    context.log.info("Training model...")
    model = train_model(X_train, y_train)
    save_model(model)  # Saves to src/models/trained_model.pkl
    context.log.info("Model trained and saved to disk.")

    yield AssetMaterialization(
        asset_key="trained_model_asset",
        description="Trained XGBoost model file."
    )
    yield Output((model, X_test, y_test))

@asset(
    ins={
        "trained_model_asset": AssetIn(),
        "processed_data_asset": AssetIn(),
    },
    description="Generates predictions using the trained model and processed data."
)
def predictions_asset(context, trained_model_asset, processed_data_asset: pd.DataFrame):
    """Asset that uses the trained model to generate predictions on processed data."""
    (model, _, _) = trained_model_asset
    X = processed_data_asset.drop(columns=["Target"])
    predictions = predict(model, X)

    yield AssetMaterialization(
        asset_key="predictions_asset",
        description="Predictions from the trained model."
    )
    yield Output(predictions)

@asset(
    ins={"trained_model_asset": AssetIn()},
    description="Evaluates the trained model on the test set, returning F1 score and accuracy."
)
def evaluation_metrics_asset(context, trained_model_asset):
    """Asset that evaluates the trained model on X_test, y_test."""
    (model, X_test, y_test) = trained_model_asset
    context.log.info("Evaluating model on test set...")

    f1, acc = evaluate(y_test, model.predict(X_test))
    yield AssetMaterialization(
        asset_key="evaluation_metrics_asset",
        description=f"F1: {f1:.4f}, Accuracy: {acc:.4f}"
    )
    yield Output({"f1_score": f1, "accuracy": acc})
