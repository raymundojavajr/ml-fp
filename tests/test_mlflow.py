# import joblib

# # Define the model path
# model_path = "src/models/trained_model.pkl"

# # Load the model
# model = joblib.load(model_path)
# print("\nModel loaded successfully")
# print(f"Model Type: {type(model)}")

# # Retrieve Model Hyperparameters
# params = model.get_xgb_params()
# print("\n===== Model Hyperparameters =====")
# for param, value in params.items():
#     print(f"  {param}: {value}")

# # Retrieve Feature Importance
# if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
#     feature_importance = {feature: importance for feature, importance in zip(model.feature_names_in_, model.feature_importances_)}
#     print("\n===== Feature Importances =====")
#     for feature, importance in feature_importance.items():
#         print(f"  {feature}: {importance:.4f}")
# else:
#     print("\nFeature importance is not available.")

# # Verify Model Training
# if hasattr(model, "n_features_in_"):
#     print(f"\nModel trained on {model.n_features_in_} features")
# else:
#     print("\nModel might not be properly trained")
