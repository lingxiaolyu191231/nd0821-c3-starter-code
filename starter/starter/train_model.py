# Script to train machine learning model.
import sys
sys.path.insert(1, './ml')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd
from data import process_data
from model import train_model, compute_model_metrics, inference
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("../data/clean_data.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

pickle.dump(encoder, open('../model/encoder.pkl', 'wb'))
pickle.dump(lb, open('../model/lb.pkl', "wb"))

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False, 
    encoder=encoder, 
    lb=lb
)

# Train and save a model.
classifier = train_model(X_train, y_train)
print(f"Train data: \nprediction metrics: ")
y_train_pred = inference(classifier, X_train)
train_precision, train_recall, train_fbeta = compute_model_metrics(y_train, y_train_pred)
print(f"\nTest data: \nprediction metrics: ")
y_test_pred = inference(classifier, X_test)
test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_test_pred)





