# orchestrator/jobs.py

from dagster import op, job
from src.models.evaluate_model import evaluate
from src.data.split_data import split_data
from src.models.train_model import load_processed_data, clean_feature_names, prepare_features, train_model
from src.models.predict_model import predict

@op
def load_and_preprocess_op(context):
    """Load and preprocess data."""
    try:
        context.log.info("Loading and preprocessing data...")
        df = load_processed_data()
        df = clean_feature_names(df)
        return df
    except Exception as e:
        context.log.error(f"Error during data loading and preprocessing: {e}")
        raise

@op
def split_data_op(context, df):
    """Split data into train and test sets."""
    try:
        context.log.info("Splitting data...")
        return split_data(df, target="Target")
    except Exception as e:
        context.log.error(f"Error during data split: {e}")
        raise

@op
def prepare_training_op(context, df):
    """Prepare features and target for training."""
    try:
        context.log.info("Preparing features and target...")
        _, _ = prepare_features(df, target="Target")
        return df
    except Exception as e:
        context.log.error(f"Error during feature preparation: {e}")
        raise

@op
def train_model_op(context, split_data):
    """Train model and return model, X_test, y_test."""
    try:
        X_train, X_test, y_train, y_test = split_data
        context.log.info("Training model...")
        model = train_model(X_train, y_train)
        return model, X_test, y_test
    except Exception as e:
        context.log.error(f"Error during model training: {e}")
        raise

@op
def evaluate_model_op(context, model_tuple):
    """Evaluate model and log metrics."""
    try:
        model, X_test, y_test = model_tuple
        f1, acc = evaluate(y_test, model.predict(X_test))
        context.log.info(f"Evaluation complete: F1 Score = {f1:.4f}, Accuracy = {acc:.4f}")
    except Exception as e:
        context.log.error(f"Error during model evaluation: {e}")
        raise

@op
def predict_model_op(context, model_tuple):
    """Generate predictions using the trained model."""
    try:
        model, _, _ = model_tuple
        context.log.info("Generating predictions...")
        processed_df = load_processed_data()
        processed_df = clean_feature_names(processed_df)
        X = processed_df.drop(columns=["Target"])
        predictions = predict(model, X)
        context.log.info(f"Predictions generated: {predictions[:10]}")
        return predictions
    except Exception as e:
        context.log.error(f"Error during prediction generation: {e}")
        raise

@job
def ml_pipeline_job():
    """Run detailed ML pipeline for better graph visualization."""
    df = load_and_preprocess_op()
    prepared_df = prepare_training_op(df)
    split = split_data_op(prepared_df)
    model_tuple = train_model_op(split)
    evaluate_model_op(model_tuple)
    predict_model_op(model_tuple)
