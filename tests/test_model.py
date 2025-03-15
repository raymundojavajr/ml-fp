import requests


url = "http://127.0.0.1:8000/train_model"

response = requests.post(url)

print("Status Code:", response.status_code)
print("Response:", response.json())
