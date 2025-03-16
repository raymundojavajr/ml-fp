# import requests
# import pandas as pd

# # ✅ Load test data
# csv_path = "data/processed/synthetic_data.csv"
# test_df = pd.read_csv(csv_path)
# json_data = test_df.to_dict(orient="records")

# # ✅ Define FastAPI Endpoints
# predict_url = "http://localhost:8000/predict"
# drift_url = "http://localhost:8085/drift"

# # ✅ Send Prediction Request
# predict_response = requests.post(predict_url, json=json_data)
# print("\n===== Prediction Response =====")
# print("Status Code:", predict_response.status_code)
# print("Response:", predict_response.json())

# # ✅ Send Drift Detection Request
# drift_response = requests.post(drift_url, json=json_data)
# print("\n===== Drift Detection Response =====")
# print("Status Code:", drift_response.status_code)

# # ✅ Print Response Safely
# try:
#     print("Response:", drift_response.json())
# except requests.exceptions.JSONDecodeError:
#     print("Drift Monitoring Service returned invalid response.")
