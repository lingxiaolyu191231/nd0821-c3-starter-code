"""
Author: Lingxiao Lyu
Date: August 27, 2021

This module is used to implement ML pipeline in FastAPI
"""
import sys
sys.path.insert(1, './starter/ml')
from data import process_data
import pickle
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Dict, Optional


class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education-num: int
    marital-status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital-gain: int
    capital-loss: int
    hours-per-week: int
    native-country: str
    salary: Optional[int] = None
    
app = FastAPI()

@app.get("/root")
async def welcome():
    return "Welcome to FASTAPI!"

@app.post("/model")
async def predict(
            input: Input = Body(None,
                examples = {
                    "standard": {
                        "summary": "A standard input example",
                        "description": "an input example",
                        "value": {
                            "age": 24,
                            "workclass": "Never-married",
                            "fnlgt": 77516,
                            "education": "Bachelors",
                            "education-num": 13,
                            "marital-status": "Divorced",
                            "occupation": "Adm-clerical",
                            "relationship": "Husband",
                            "race": "White",
                            "sex": "Male",
                            "capital-gain": 0,
                            "capital-loss": 0,
                            "hours-per-week": 40,
                            "native-country": "United-States",
                            "salary": 1000
                                  }
                                }
                            
                                }
                                )
                            ):
    load_gbc = pickle.load(open("./model/gbclassifier.pkl", "rb"))
    
    # load encoder
    data = pd.read_csv("../data/clean_data.csv")
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
    _, _, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
    
    # load predict_data
    request_data = pd.DataFrame.from_dict(Input, orient="columns")
    X_request, y_request, _, _ = process_data(
                request_data,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb)
    
    y_request_pred = load_gbc.predict(X_request)
    
    return y_request_pred
    
    

