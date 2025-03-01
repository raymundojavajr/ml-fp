# pipelines/dagster_pipeline.py

from dagster import op, job, repository, Out
from pipelines.assets import processed_data_asset

@op
def get_processed_data_op(context):
    """Materializes and returns the processed data asset."""
    context.log.info("Materializing processed data asset...")
    # Convert the generator returned by processed_data_asset() to a list and take the last output
    asset_outputs = list(processed_data_asset())
    processed_data = asset_outputs[-1].value
    context.log.info("Processed data asset materialized.")
    return processed_data

@op(out={"model": Out(), "X_test": Out(), "y_test": Out()})
def train_model_op(context, processed_data):
    """
    Splits the processed data using the existing split_data function,
    trains the model using existing functions, saves the model,
    and returns a tuple with the model, X_test, and y_test.
    """
    context.log.info("Splitting data using split_data function...")
    from src.data.split_data import split_data  # Reuse your split_data function
    X_train, X_test, y_train, y_test = split_data(processed_data, target="Target")
    
    context.log.info("Training model...")
    from src.models.train_model import train_model, save_model
    model = train_model(X_train, y_train)
    save_model(model)
    context.log.info("Model training complete and saved.")
    
    return model, X_test, y_test

@op
def predict_model_op(context, model):
    """
    Loads the processed data, cleans feature names,
    drops the target column, and generates predictions using the trained model.
    """
    context.log.info("Loading processed data for prediction...")
    from src.models.predict_model import predict, load_processed_data, clean_feature_names
    processed_df = load_processed_data()
    processed_df = clean_feature_names(processed_df)
    X = processed_df.drop(columns=["Target"])
    context.log.info("Generating predictions...")
    predictions = predict(model, X)
    context.log.info(f"Predictions generated: {predictions[:10]}")
    return predictions

@op
def evaluate_model_op(context, model, X_test, y_test):
    """
    Evaluates the model on the test set using the existing evaluate function.
    Returns a tuple with F1 score and accuracy.
    """
    context.log.info("Evaluating model...")
    from src.models.evaluate_model import evaluate
    y_pred = model.predict(X_test)
    f1, acc = evaluate(y_test, y_pred)
    context.log.info(f"Evaluation complete: F1 Score = {f1:.4f}, Accuracy = {acc:.4f}")
    return f1, acc

@job
def ml_pipeline_job():
    processed_data = get_processed_data_op()
    model, X_test, y_test = train_model_op(processed_data)
    predict_model_op(model)
    evaluate_model_op(model, X_test, y_test)

@repository
def ml_ops_repository():
    return [ml_pipeline_job]
