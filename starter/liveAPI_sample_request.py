import os
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
    "salary": ">50K"
}

response = requests.post("http://udacity-fastapi-project.herokuapp.com/prediction/", data=json.dumps(data))
print(response)
print(os.getcwd())
if response.status_code == 200:
    print(response)
    if response.json()['predict'] == 0:
        print('predicted salary: <=50K')
    else:
        print('predicted salary: >50K')
else:
    print('prediction process failed!')
