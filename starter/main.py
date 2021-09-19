"""
Author: Lingxiao Lyu
Date: August 27, 2021

This module is used to implement ML pipeline in FastAPI
"""
import os
import sys
import git
import subprocess
import pandas as pd
sys.path.insert(1, './starter/ml')
sys.path.append('./starter/starter/ml')
from data import process_data
import pickle
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Dict, Optional
from sklearn.model_selection import train_test_split

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    dvc_output = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    print(dvc_output.stdout)
    print(dvc_output.stderr)
    if dvc_output.returncode != 0:
        print("dvc pull failed")
    else:
        os.system("rm -r .dvc .apt/usr/lib/dvc")

def root(path = os.getcwd()):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    print(git_root)
    
    return git_root

root = root(path = os.getcwd())

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
    predict: int

app = FastAPI()

@app.get("/")
async def welcome():
    return "Welcome! You are at the Homepage of FastAPI"

@app.post("/prediction/", response_model=Output, status_code=200)
async def predict(input: Input):

    #print(os.listdir("../starter/model/"))
    # Load gradiant boosting classifier
    print(os.getcwd())
    try:
        load_gbc = pickle.load(open(os.path.join(root,"starter/model/gbclassifier.pkl"), "rb"))
    except FileNotFoundError:
        load_gbc = pickle.load(open("./model/gbclassifier.pkl", "rb"))

    # load encoder
    try:
        encoder = pickle.load(open(os.path.join(root,"starter/model/encoder.pkl"), "rb"))
    except FileNotFoundError:
        encoder = pickle.load(open("./model/encoder.pkl", "rb"))

    # load lb
    try:
        lb = pickle.load(open(os.path.join(root,"starter/model/lb.pkl"), "rb"))
    except FileNotFoundError:
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
    print(request_dict)
    request_data = pd.DataFrame(request_dict, index=[0])
    print(request_data)
    
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
    
    

