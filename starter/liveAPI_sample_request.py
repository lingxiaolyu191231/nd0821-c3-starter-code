import requests
import json

data = {
    "age": 30,
    "workclass": "Never-married",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Divorced",
    "occupation": "Adm-clerical",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 1000,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
    "salary": 10000
}

response = requests.post("https://udacity-fastapi-ling09032021.herokuapp.com/", auth=('usr', 'pass'), data=json.dumps(data))
print(response.status_code)
print(response.json())