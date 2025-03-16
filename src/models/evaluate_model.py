import logging
import mlflow
from sklearn.metrics import f1_score, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate(y_true, y_pred):
    """Evaluates predictions by computing F1 score and accuracy, with MLflow logging."""
    try:
        f1 = f1_score(y_true, y_pred, zero_division=1)  # Handles cases where one class is missing
        acc = accuracy_score(y_true, y_pred)

        # Log evaluation metrics
        with mlflow.start_run():
            mlflow.log_metric("F1 Score", f1)
            mlflow.log_metric("Accuracy", acc)

        logging.info(f"Evaluation Complete: F1 Score = {f1:.4f}, Accuracy = {acc:.4f}")
        
        return {"F1 Score": f1, "Accuracy": acc}

    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Example for quick testing
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]
    print(evaluate(y_true, y_pred))
