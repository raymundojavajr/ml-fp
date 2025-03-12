import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from src.data.load_data import find_root, load_processed_data
from mlflow.tracking import MlflowClient
import shap
import matplotlib.pyplot as plt

# Ensure this function is imported correctly
from src.models.evaluate_model import evaluate  # Import your custom evaluate function

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

def train_model(X_train, y_train, random_state=42, learning_rate=0.1, max_depth=6, n_estimators=100):
    """Trains an XGBoost classifier and logs it in MLflow."""
    model = XGBClassifier(
        random_state=random_state, 
        eval_metric="logloss",
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators
    )
    
    # Log these hyperparameters to MLflow
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs metrics in MLflow."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return f1, acc

def generate_shap_plot(X_train, model):
    """Generate and save SHAP plot as an artifact."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    
    # Save SHAP plot
    shap.summary_plot(shap_values, X_train)
    plt.savefig("shap_plot.png")
    mlflow.log_artifact("shap_plot.png")  # Log SHAP plot as an artifact

def save_model(model, model_name="trained_model.pkl"):
    """Saves the trained model to the models directory."""
    root = find_root()  # Ensure this function is imported from your utils
    models_dir = root / "src" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / model_name
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")

def train_model_and_log(X_train, y_train, X_test, y_test):
    """Train, evaluate, log, and save model."""
    run_name = "training-run-with-featureengineering-X"  # Customize this name as per your run
    
    with mlflow.start_run(run_name=run_name):  # This will assign the name to the run
        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        f1, acc = evaluate(y_test, model.predict(X_test))  # Use the evaluate function to get f1 and accuracy
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test Accuracy: {acc:.4f}")

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # Register the model in MLflow as "Champion_Model"
        mlflow.xgboost.log_model(model, "model", registered_model_name="Champion_Model")

        # Generate SHAP plot after training the model
        generate_shap_plot(X_train, model)

        # Save the model locally
        save_model(model)

# Add a function to generate and log drift reports:
def generate_drift_reports(reference_data, current_data):
    """Generate and log drift reports."""
    
    # Create the Evidently dashboard
    from evidently.report import Report
    from evidently.metrics import DataDriftMetric, TargetDriftMetric
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
    dashboard = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), ClassificationPreset()])
    
    # Calculate the drift between the reference and current datasets
    dashboard.calculate(reference_data, current_data)
    
    # Save the drift report as HTML files
    data_drift_report = "data_drift_report.html"
    target_drift_report = "target_drift_report.html"
    performance_report = "performance_report.html"
    
    dashboard.save(data_drift_report)
    dashboard.save(target_drift_report)
    dashboard.save(performance_report)
    
    # Log the drift reports as artifacts in MLflow
    mlflow.log_artifact(data_drift_report)
    mlflow.log_artifact(target_drift_report)
    mlflow.log_artifact(performance_report)
    
    print(f"Drift reports saved and logged as artifacts in MLflow.")

def main():
    # Load and prepare data
    df = load_processed_data()
    df = clean_feature_names(df)
    X, y = prepare_features(df, target="Target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow experiment tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # Ensure MLflow tracking URI is set
    mlflow.set_experiment("Predictive Maintenance")

    run_name = "training-run-with-featureengineering-X"  # Customize this name as per your run
    with mlflow.start_run(run_name=run_name):  # This will assign the name to the run
        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        f1, acc = evaluate(y_test, model.predict(X_test))  # Use the evaluate function to get f1 and accuracy
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test Accuracy: {acc:.4f}")

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # Register the model in MLflow as "Champion_Model"
        mlflow.xgboost.log_model(model, "model", registered_model_name="Champion_Model")

        # Generate SHAP plot after training the model
        generate_shap_plot(X_train, model)

        # Save the model locally
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
