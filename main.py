import logging

def main():
    # Configure logging at the INFO level
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    logging.info("Starting the ML pipeline...\n")
    
    # Step 1: Train the model
    logging.info("Training the model...")
    from src.models.train_model import main as train_main
    train_main()
    logging.info("Model training complete.\n")
    
    # Step 2: Generate predictions using the saved model
    logging.info("Generating predictions...")
    from src.models.predict_model import main as predict_main
    predict_main()
    logging.info("Prediction complete.\n")
    
    # Optionally, add more steps (e.g., evaluation, drift detection, etc.)
    logging.info("ML pipeline finished successfully.")

if __name__ == "__main__":
    main()
