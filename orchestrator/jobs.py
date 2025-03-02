# orchestrator/jobs.py

from dagster import op, job, repository, Out
from src.data.load_data import load_processed_data
from src.models.train_model import main as train_main
from src.models.predict_model import main as predict_main
from src.models.evaluate_model import evaluate
from src.data.split_data import split_data

@op
def train_model_op(context):
    """
    A simple op that calls your existing train_main() or trains inline.
    Returns a trained model object, X_test, y_test for further ops.
    """
    context.log.info("Running train_model_op...")
    # If you want to do it inline, you can replicate train_main logic:
    from src.models.train_model import load_processed_data, clean_feature_names, prepare_features, train_model
    df = load_processed_data()
    df = clean_feature_names(df)
    X, y = prepare_features(df, target="Target")

    X_train, X_test, y_train, y_test = split_data(df, target="Target")
    model = train_model(X_train, y_train)

    return model, X_test, y_test

@op
def evaluate_model_op(context, model_tuple):
    """
    Evaluates the model on X_test, y_test and logs metrics.
    """
    (model, X_test, y_test) = model_tuple
    from src.models.evaluate_model import evaluate
    f1, acc = evaluate(y_test, model.predict(X_test))
    context.log.info(f"Evaluation complete: F1 Score = {f1:.4f}, Accuracy = {acc:.4f}")

@op
def predict_model_op(context, model_tuple):
    """
    Generates predictions using the trained model.
    """
    (model, X_test, y_test) = model_tuple
    from src.models.predict_model import predict, load_processed_data, clean_feature_names
    processed_df = load_processed_data()
    processed_df = clean_feature_names(processed_df)
    X = processed_df.drop(columns=["Target"])
    predictions = predict(model, X)
    context.log.info(f"Predictions generated: {predictions[:10]}")
    return predictions

@job
def ml_pipeline_job():
    """
    A traditional pipeline job that runs train, evaluate, predict in sequence.
    """
    model_tuple = train_model_op()
    evaluate_model_op(model_tuple)
    predict_model_op(model_tuple)
