import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.data.load_data import find_root, load_processed_data
import datetime


def clean_feature_names(df):
    """Cleans column names by removing brackets and replacing spaces with underscores."""
    df.columns = df.columns.str.replace(r'[\[\]]', '', regex=True).str.replace(' ', '_')
    return df


def prepare_features(df, target="Target"):
    """Prepares the feature matrix and target variable."""
    X = df.drop(columns=[target])  
    y = df[target]
    X = clean_feature_names(X)
    return X, y


def train_model(X_train, y_train, random_state=42):
    """Trains an XGBoost model."""
    model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs metrics."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, acc, precision, recall


def save_model(model, model_name="trained_model.pkl"):
    """Saves the trained model locally."""
    root = find_root()
    models_dir = root / "src" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / model_name
    joblib.dump(model, save_path)
    return save_path


def main():
    # Load and prepare data
    df = load_processed_data()
    df = clean_feature_names(df)
    X, y = prepare_features(df, target="Target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set MLflow Tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = "Predictive Maintenance"
    mlflow.set_experiment(experiment_name)

    run_name = f"training-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        model = train_model(X_train, y_train)

        # Evaluate the model
        f1, acc, precision, recall = evaluate_model(model, X_test, y_test)

        # Log Hyperparameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("n_estimators", 100)

        # Log Metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log Feature Importance
        importance = model.feature_importances_
        importance_dict = {col: importance[idx] for idx, col in enumerate(X_train.columns)}
        mlflow.log_dict(importance_dict, "feature_importance.json")

        # Register the Model in MLflow Model Registry
        mlflow.xgboost.log_model(model, "model", registered_model_name="predictive_maintenance_model")

        # Save Model Locally
        save_model(model)

    print(f"Model training complete! Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
