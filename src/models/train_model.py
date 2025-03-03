import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from src.data.load_data import find_root, load_processed_data  # Updated import
from mlflow.tracking import MlflowClient

def clean_feature_names(df):
    """Cleans DataFrame column names by removing brackets and replacing spaces with underscores."""
    df.columns = df.columns.str.replace(r'[\[\]]', '', regex=True).str.replace(' ', '_')
    return df

def prepare_features(df, target="Target"):
    """Prepares the feature matrix and target vector."""
    X = df.drop(columns=[target])
    y = df[target]
    X = clean_feature_names(X)
    return X, y

def train_model(X_train, y_train, random_state=42):
    """Trains an XGBoost classifier and logs it in MLflow."""
    model = XGBClassifier(random_state=random_state, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs metrics in MLflow."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return f1, acc, y_pred

def save_model(model, model_name="trained_model.pkl"):
    """Saves the trained model to the models directory."""
    root = find_root()
    models_dir = root / "src" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / model_name
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")

def main():
    # Load and prepare data
    df = load_processed_data()
    df = clean_feature_names(df)
    X, y = prepare_features(df, target="Target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow experiment tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure MLflow tracking URI is set
    mlflow.set_experiment("Predictive Maintenance")
    
    with mlflow.start_run():
        model = train_model(X_train, y_train)

        # Evaluate the model
        f1, acc, _ = evaluate_model(model, X_test, y_test)
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test Accuracy: {acc:.4f}")

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # ✅ Register the model in MLflow as "Champion_Model"
        mlflow.xgboost.log_model(model, "model", registered_model_name="Champion_Model")

        # Save locally
        save_model(model)

        
def get_best_model():
    """Fetch the best model from MLflow based on the highest F1 score."""
    client = MlflowClient()
    
    # Search for the best run in the "Predictive Maintenance" experiment
    experiment = client.get_experiment_by_name("Predictive Maintenance")
    
    if not experiment:
        print("Experiment not found. Train a model first!")
        return None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1_score DESC"])

    if runs:
        best_run = runs[0]
        best_model_uri = best_run.info.artifact_uri + "/model"
        print(f"Champion Model Found: {best_model_uri}")
        return best_model_uri
    else:
        print("No trained models found.")
        return None

if __name__ == "__main__":
    main()
