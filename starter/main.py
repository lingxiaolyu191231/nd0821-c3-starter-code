"""
Author: Lingxiao Lyu
Date: August 27, 2021

This module is used to implement ML pipeline in FastAPI
"""
import os
import sys
import pandas as pd
sys.path.append('../../starter/starter/ml')
import data
from starter.ml.data import process_data
import pickle
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Dict, Optional
from sklearn.model_selection import train_test_split

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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


class Output(BaseModel):
    predict: float

app = FastAPI()

@app.get("/")
async def welcome():
    return "Welcome! You are at the Homepage of FastAPI"

@app.post("/prediction/", response_model=Output, status_code=200)
async def predict(input: Input):

    # Load gradiant boosting classifier
    load_gbc = pickle.load(open("./model/gbclassifier.pkl", "rb"))

    # load encoder
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))

    # load lb
    lb = pickle.load(open("./model/lb.pkl", "rb"))

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

    # load predict_data
    request_dict = input.dict(by_alias=True)
    request_data = pd.DataFrame(request_dict, index=[0])
    
    X_request, _, _, _ = process_data(
                request_data,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb)
    
    y_request_pred = load_gbc.predict(X_request)
    print(y_request_pred)
    return {"predict": y_request_pred[0]}
    
    

