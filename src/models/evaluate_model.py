import logging
from sklearn.metrics import f1_score, accuracy_score

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate(y_true, y_pred):
    """Evaluates predictions by computing F1 score and accuracy."""
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Accuracy: {acc:.4f}")
    return f1, acc

if __name__ == "__main__":
    # Example for quick testing
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]
    evaluate(y_true, y_pred)
