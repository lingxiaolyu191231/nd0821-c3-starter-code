import pandas as pd
from .ml.model import inference, train_model, compute_model_metrics
from sklearn.model_selection import train_test_split
from .ml.data import process_data
import pytest
import pickle
import os


@pytest.fixture
def root():
    return os.getcwd()


@pytest.fixture
def alldata(root):
    """Load data"""
    try:
        data = pd.read_csv(os.path.join(root, 'starter/data/clean_data.csv'))
    except FileNotFoundError:
        data = pd.read_csv('../data/clean_data.csv')

    if os.path.exists(os.path.join(root, "starter/model/gbclassifier.pkl")):
        model = os.path.join(root, "starter/model/gbclassifier.pkl")
    else:
        model = "../model/gbclassifier.pkl"
    with open(model, "rb") as f:
        model = pickle.load(f)

    if os.path.exists(os.path.join(root, "starter/model/encoder.pkl")):
        encoder = os.path.join(root, "starter/model/encoder.pkl")
    else:
        encoder = "../model/encoder.pkl"
    with open(encoder, "rb") as f:
        encoder = pickle.load(f)

    if os.path.exists(os.path.join(root, "starter/model/lb.pkl")):
        lb = os.path.join(root, "starter/model/lb.pkl")
    else:
        lb = "../model/lb.pkl"
    with open(lb, "rb") as f:
        lb = pickle.load(f)

    return data, model, encoder, lb


def test_data_shape(alldata):
    """Test whether enough records are passed and number
    of columns is as expected"""
    data, model, encoder, lb = alldata

    assert data.shape[0] > 30000
    assert data.shape[1] == 15


def test_train_model(alldata, root):
    """Test whether model is built and file exists"""
    data, model, encoder, lb = alldata

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
    if os.path.exists(
        os.path.join(
            root,
            "starter/model/gbclassifier_test.pkl")):
        filepath = os.path.join(root, "starter/model/gbclassifier_test.pkl")
    else:
        filepath = "./gbclassifier_test.pkl"

    model = train_model(X_train, y_train, filepath=filepath)

    assert os.path.exists(filepath)
    return X_train, y_train, model, encoder, lb


@pytest.fixture
def train_test_data(alldata):
    """Fix train-test data"""
    data, model, encoder, lb = alldata

    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return (train, test)


def test_inference(train_test_data, alldata):

    _, model, encoder, lb = alldata
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
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary",
        training=True, encoder=encoder, lb=lb)
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb)

    y_train_pred = inference(model, X_train)
    assert len(y_train_pred) == X_train.shape[0]
    assert len(y_train_pred) > 0

    y_test_pred = inference(model, X_test)
    assert len(y_test_pred) == X_test.shape[0]
    assert len(y_test_pred) > 0


def test_compute_model_metrics(alldata, train_test_data):

    _, model, encoder, lb = alldata
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

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary",
        training=True, encoder=encoder, lb=lb)

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb)

    y_train_pred = inference(model, X_train)
    y_test_pred = inference(model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(
        y_train, y_train_pred)
    precision_test, recall_test, fbeta_test = compute_model_metrics(
        y_test, y_test_pred)

    assert isinstance(precision_train, float)
    assert isinstance(precision_test, float)
    assert isinstance(recall_train, float)
    assert isinstance(recall_test, float)
    assert isinstance(fbeta_train, float)
    assert isinstance(fbeta_test, float)

    assert (precision_train <= 1) & (precision_train >= 0)
    assert (precision_test <= 1) & (precision_test >= 0)
    assert (recall_train <= 1) & (recall_train >= 0)
    assert (recall_test <= 1) & (recall_test >= 0)
    assert (fbeta_train <= 1) & (fbeta_train >= 0)
    assert (fbeta_test <= 1) & (fbeta_test >= 0)
