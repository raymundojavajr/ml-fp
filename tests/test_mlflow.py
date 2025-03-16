import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Check connection
try:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()  # ✅ Use this instead of list_experiments()
    print("✅ MLflow is reachable!")
    for exp in experiments:
        print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
except Exception as e:
    print(f"❌ Error connecting to MLflow: {e}")
