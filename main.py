# main.py

def main():
    print("Starting the ML pipeline...\n")
    
    # Step 1: Train the model
    print("Training the model...")
    from src.models.train_model import main as train_main
    train_main()
    print("Model training complete.\n")
    
    # Step 2: Generate predictions using the saved model
    print("Generating predictions...")
    from src.models.predict_model import main as predict_main
    predict_main()
    print("Prediction complete.\n")
    
    # Optionally, add more steps (e.g., evaluation, drift detection, etc.)
    print("ML pipeline finished successfully.")

if __name__ == "__main__":
    main()


