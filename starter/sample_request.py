import json
import requests

request_data = {
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
    "salary": "<=50K"
}

r = requests.post(
    "http://127.0.0.1:8000/prediction/",
    data=json.dumps(request_data))
print(r)
if r.json()['predict'] == 0:
    print('predicted salary: <=50K')
else:
    print('predicted salary: >50K')
