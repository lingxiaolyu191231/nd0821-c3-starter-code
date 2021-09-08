import os
import sys
sys.path.insert(1, './ml')
import pytest
import pandas as pd 
from data import process_data
from sklearn.model_selection import train_test_split
import pickle
from model import inference,train_model, compute_model_metrics

@pytest.fixture
def data():
    """Load data"""
    data = pd.read_csv("../../../starter/data/clean_data.csv")
    return data

def test_data_shape(data):
    """Test whether enough records are passed and number of columns is as expected"""
    assert data.shape[0]>30000
    assert data.shape[1]==15

def test_train_model(data):
    """Test whether model is built and file exists"""
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
    model = train_model(X_train, y_train)
    filepath = "../../../starter/model/gbclassifier.pkl"
    assert os.path.exists(filepath)

@pytest.fixture
def train_test_data(data):
    """Fix train-test data"""
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return (train, test)


def test_inference(train_test_data):
    train, test = train_test_data
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
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
    X_test, y_test, _ , _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
    model = "../../../starter/model/gbclassifier.pkl"
    gbc = pickle.load(open(model, 'rb'))

    y_train_pred = inference(gbc, X_train)
    assert len(y_train_pred) == X_train.shape[0]
    assert len(y_train_pred) > 0
    
    y_test_pred = inference(gbc, X_test)
    assert len(y_test_pred) == X_test.shape[0]
    assert len(y_test_pred) > 0


def test_compute_model_metrics(train_test_data):
    train, test = train_test_data

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

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
    X_test, y_test, _ , _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
    model = "../../../starter/model/gbclassifier.pkl"
    with open(model, 'rb') as file:  
        gbc = pickle.load(file)

    y_train_pred = inference(gbc, X_train)   
    y_test_pred = inference(gbc, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, y_train_pred)
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, y_test_pred)

    assert isinstance(precision_train, float)
    assert isinstance(precision_test, float)
    assert isinstance(recall_train, float)
    assert isinstance(recall_test, float)
    assert isinstance(fbeta_train, float)
    assert isinstance(fbeta_test, float)

    assert (precision_train<=1) & (precision_train>=0)
    assert (precision_test<=1) & (precision_test>=0)
    assert (recall_train<=1) & (recall_train>=0)
    assert (recall_test<=1) & (recall_test>=0)
    assert (fbeta_train<=1) & (fbeta_train>=0)
    assert (fbeta_test<=1) & (fbeta_test>=0)
