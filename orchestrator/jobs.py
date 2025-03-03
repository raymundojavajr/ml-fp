# orchestrator/jobs.py

from dagster import op, job
from src.models.evaluate_model import evaluate
from src.data.split_data import split_data


@op
def train_model_op(context):
    """Train model and return model, X_test, y_test."""
    context.log.info("Running train_model_op...")
    from src.models.train_model import load_processed_data, clean_feature_names, prepare_features, train_model
    df = load_processed_data()
    df = clean_feature_names(df)
    X, y = prepare_features(df, target="Target")
    X_train, X_test, y_train, y_test = split_data(df, target="Target")
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
    from src.models.predict_model import predict, load_processed_data, clean_feature_names
    processed_df = load_processed_data()
    processed_df = clean_feature_names(processed_df)
    X = processed_df.drop(columns=["Target"])
    predictions = predict(model, X)
    context.log.info(f"Predictions generated: {predictions[:10]}")
    return predictions


@job
def ml_pipeline_job():
    """Run train, evaluate, and predict ops sequentially."""
    model_tuple = train_model_op()
    evaluate_model_op(model_tuple)
    predict_model_op(model_tuple)
