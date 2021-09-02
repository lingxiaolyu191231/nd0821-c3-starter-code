"""
Author: Lingxiao Lyu
Date: August 27, 2021

This module is used to implement ML pipeline in FastAPI
"""
import sys
import pandas as pd
sys.path.insert(1, './starter/ml')
from data import process_data
import pickle
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Dict, Optional
from sklearn.model_selection import train_test_split


class Input(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example = "Never-married")
    fnlgt: int = Field(..., example = 77516)
    education: str = Field(..., example = "Bachelors")
    education_num: int = Field(..., alias = "education-num", example = 13)
    marital_status: str = Field(..., alias = "marital-status", example = "Divorced")
    occupation: str = Field(..., example = "Adm-clerical")
    relationship: str = Field(..., example = "Husband")
    race: str = Field(..., example = "White")
    sex: str = Field(..., example = "Male")
    capital_gain: int = Field(..., alias = "capital-gain", example = 0)
    capital_loss: int = Field(..., alias = "capital-loss", example = 0)
    hours_per_week: int = Field(..., alias = "hours-per-week", example = 40)
    native_country: str = Field(..., alias = "native-country", example = "United-States")
    salary: Optional[int]


class Output(Input):
    predict: float

app = FastAPI()

# Load gradiant boosting classifier
load_gbc = pickle.load(open("./model/gbclassifier.pkl", "rb"))

# load encoder
encoder = pickle.load(open("./model/encoder.pkl", "rb"))

# load lb
lb = pickle.load(open("./model/lb.pkl", "rb"))

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


@app.get("/")
async def welcome():
    return "Welcome! You are at the Homepage of FastAPI"

@app.post("/prediction/", response_model=Output, status_code=200)
async def predict(input: Input):
    
    # load predict_data
    request_dict = input.dict()
    for key in request_dict.keys():
        key = key.replace('_','-')
        new_request_dict[key] = request_dict[key]
    new_request_data = pd.DataFrame(request_dict,index=[0])
    print(new_request_data)


    X_request, y_request, _, _ = process_data(
                request_data,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb)
    
    y_request_pred = load_gbc.predict(X_request)
    
    return {"prediction": y_request_pred}
    
    

