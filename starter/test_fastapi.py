from fastapi.testclient import TestClient
import requests
from main import app
import json

client = TestClient(app)

def test_get_api_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome! You are at the Homepage of FastAPI"

def test_get_prediont_one():

    request_data = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
    "salary": "<=50K"
}

    r = requests.post("http://127.0.0.1:8000/prediction/", data=json.dumps(request_data))

    assert r.status_code == 200
    assert r.json()['predict'] == 0
    # class 0 represent salary <=50K, according to LabelBinarizer output
    

def test_get_prediont_two():
    
    # salary (not provided)
    request_data = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14084,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
    "salary": "<=50K"
}
    r = requests.post("http://127.0.0.1:8000/prediction/", data=json.dumps(request_data))
    
    assert r.status_code == 200
    assert r.json()['predict'] == 1
    # class 1 represent salary >50K, according to LabelBinarizer output
    
    
def test_miss_race_feature():
    
    # no race feature
    request_data = {
    "age": 30,
    "workclass": "Never-married",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Divorced",
    "occupation": "Adm-clerical",
    "relationship": "Husband",
    "sex": "Male",
    "capital-gain": 1000,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
    "salary": ">50K"
}
    r = requests.post("http://127.0.0.1:8000/prediction/", data=json.dumps(request_data))
    
    assert r.status_code != 200

def test_wrong_feature_type():

    # age: string (wrong type)
    request_data = {
    "age": "45",
    "workclass": "Never-married",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Divorced",
    "occupation": "Adm-clerical",
    "relationship": "Husband",
    "sex": "Male",
    "capital-gain": 1000,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
    "salary": "<=50K"
}
    r = requests.post("http://127.0.0.1:8000/prediction/", data=json.dumps(request_data))
    
    assert r.status_code != 200
    
