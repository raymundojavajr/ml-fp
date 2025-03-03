from dagster import op, job
from src.models.evaluate_model import evaluate
from src.data.split_data import split_data
from src.models.train_model import load_processed_data, clean_feature_names, prepare_features, train_model
from src.models.predict_model import predict

@op
def load_and_preprocess_op(context):
    """Load and preprocess data."""
    context.log.info("Loading and preprocessing data...")
    df = load_processed_data()
    df = clean_feature_names(df)
    return df  # Return the full DataFrame

@op
def split_data_op(context, df):
    """Split data into train and test sets."""
    context.log.info("Splitting data...")
    return split_data(df, target="Target")

@op
def prepare_training_op(context, df):
    """Prepare features and target for training."""
    context.log.info("Preparing features and target...")
    _, _ = prepare_features(df, target="Target")  # This op can perform additional logic if needed
    return df

@op
def train_model_op(context, split_data):
    """Train model and return model, X_test, y_test."""
    X_train, X_test, y_train, y_test = split_data
    context.log.info("Training model...")
    model = train_model(X_train, y_train)
    return model, X_test, y_test

@op
def evaluate_model_op(context, model_tuple):
    """Evaluate model and log metrics."""
    model, X_test, y_test = model_tuple
    f1, acc = evaluate(y_test, model.predict(X_test))
    context.log.info(f"Evaluation complete: F1 Score = {f1:.4f}, Accuracy = {acc:.4f}")

@op
def predict_model_op(context, model_tuple):
    """Generate predictions using the trained model."""
    model, _, _ = model_tuple
    context.log.info("Generating predictions...")
    processed_df = load_processed_data()
    processed_df = clean_feature_names(processed_df)
    X = processed_df.drop(columns=["Target"])
    predictions = predict(model, X)
    context.log.info(f"Predictions generated: {predictions[:10]}")
    return predictions

@job
def ml_pipeline_job():
    """Run detailed ML pipeline for better graph visualization."""
    df = load_and_preprocess_op()
    # Optionally, prepare features further if needed:
    prepared_df = prepare_training_op(df)
    split = split_data_op(prepared_df)
    model_tuple = train_model_op(split)
    evaluate_model_op(model_tuple)
    predict_model_op(model_tuple)
