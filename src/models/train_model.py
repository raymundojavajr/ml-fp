# src/models/train_model.py

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from src.data.load_data import find_root

def load_processed_data(file_relative_path="data/processed/predictive_maintenance_processed.csv"):
    """Loads the processed data CSV from a path relative to the project root."""
    root = find_root()
    file_path = root / file_relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    return pd.read_csv(file_path)

def clean_feature_names(df):
    """Cleans DataFrame column names by removing brackets and replacing spaces with underscores."""
    df.columns = df.columns.str.replace(r'[\[\]]', '', regex=True).str.replace(' ', '_')
    return df

def prepare_features(df, target="Target"):
    """
    Prepares the feature matrix and target vector.
    Drops the target column and cleans the feature names.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X = clean_feature_names(X)
    return X, y

def train_model(X_train, y_train, random_state=42):
    """Trains an XGBoost classifier on the training data and returns the model."""
    model = XGBClassifier(random_state=random_state, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set and returns F1 score, accuracy, and predictions."""
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
    # Load the stable, processed data
    df = load_processed_data()
    
    # Clean the DataFrame's column names
    df = clean_feature_names(df)
    
    # Prepare features (drop target and clean feature names for features)
    X, y = prepare_features(df, target="Target")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model on the training set
    model = train_model(X_train, y_train)
    
    # Evaluate the model on the test set
    f1, acc, _ = evaluate_model(model, X_test, y_test)
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
