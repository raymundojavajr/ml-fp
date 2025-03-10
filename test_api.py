import requests

url = "http://127.0.0.1:8000/predict"
data = [
    {
        "UDI": 1,  # Add the missing UDI field
        "Air_temperature_K": 298.1,
        "Process_temperature_K": 308.6,
        "Rotational_speed_rpm": 1551,
        "Torque_Nm": 42.8,
        "Tool_wear_min": 0,
        "Type_encoded": 2,
        "Product_ID_encoded": 7003,
        "Failure_Type_encoded": 1
    }
]

response = requests.post(url, json=data)

#  Print raw response to debug
print("Raw Response:", response.text)  
print("Status Code:", response.status_code)

try:
    print("Prediction Response:", response.json())  
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not valid JSON. Check FastAPI logs for errors.")
