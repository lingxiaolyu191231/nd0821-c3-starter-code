import json
import requests

request_data = {
    "age": 30,
    "workclass": "Never-married",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Divorced",
    "occupation": "Adm-clerical",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 1000,
    "capital_loss": 0,
    "hours_per_week": 20,
    "native_country": "United-States",
    "salary": 10000
}

r = requests.post("http://127.0.0.1:8000/prediction/", data=json.dumps(request_data))

print(r.json())