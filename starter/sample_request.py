import json
import pickle
import requests
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier

class MyEstimator(BaseEstimator):
     def __init__(self, estimator=None, my_extra_param="random"):
         self.estimator = estimator
         self.my_extra_param = my_extra_param

load_model = pickle.load(open("./model/gbclassifier.pkl", "rb"))
my_estimator = MyEstimator(estimator=GradientBoostingClassifier())
param_dict = dict()
for param, value in my_estimator.get_params(deep=True).items():
    param_dict[param] = str(value)

update_param_dict = dict()
for key in param_dict.keys():
    if param_dict[key] != None:
        update_param_dict[key] = str(param_dict[key])

r = requests.post("http://127.0.0.1:8000/model", data=json.dumps(update_param_dict))

print(r.json())