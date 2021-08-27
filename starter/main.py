"""
Author: Lingxiao Lyu
Date: August 27, 2021

This module is used to implement ML pipeline in FastAPI
"""
import pickle
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Dict, Optional

class inference_model(BaseModel):
    model_parameters: Dict[str, str]
    
app = FastAPI()

@app.get("/root")
async def welcome():
    return "Welcome to FASTAPI!"

@app.post("/model")
async def display_model(
            model: inference_model = Body(None,
                examples = {
                    "normal": {
                        "summary": "A standard example",
                        "description": "expected output",
                        "value": {
                            "model_name":
                                "classification model on publicly available Census Bureau data",
                            "model_type": "Gradient boosted tree",
                            "model_parameters":
                                {'estimator__ccp_alpha': '0.0',
                                 'estimator__criterion': 'friedman_mse',
                                 'estimator__learning_rate': '0.1',
                                 'estimator__loss': 'deviance',
                                 'estimator__max_depth': '3',
                                 'estimator__min_impurity_decrease': '0.0',
                                 'estimator__min_samples_leaf': '1',
                                 'estimator__min_samples_split': '2',
                                 'estimator__min_weight_fraction_leaf': '0.0',
                                 'estimator__n_estimators': '100',
                                 'estimator__subsample': '1.0',
                                 'estimator__tol': '0.0001',
                                 'estimator__validation_fraction': '0.1',
                                 'estimator__verbose': '0',
                                 'estimator__warm_start': 'False',
                                 'estimator': 'GradientBoostingClassifier()',
                                 'my_extra_param': 'random'}
                                          }
                                    }
                                })

):
    return {"model_name":
                "classification model on publicly available Census Bureau data",
            "model_type": "Gradient boosted tree",
            "model_parameters": model
}
    

