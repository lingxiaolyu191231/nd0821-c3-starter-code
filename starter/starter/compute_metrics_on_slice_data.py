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
load_gbc = pickle.load(open("../model/gbclassifier.pkl", "rb"))
encoder = pickle.load(open("../model/encoder.pkl", "rb"))
lb = pickle.load(open("../model/lb.pkl", "rb"))
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

def compute_metrics_on_data_slice(categorical_feature, 
                                  categorical_feature_value,
                                  data=data,
                                  model = load_gbc,
                                  encoder = encoder,
                                  lb = lb
                                  ):
    X_slice, y_slice, _, _ = process_data(data, 
                                    categorical_features=cat_features,
                                    label = 'salary',
                                    training=False,
                                    encoder=encoder,
                                    lb=lb
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    with open('../screenshots/slice_output.txt', 'w') as f:
        f.write(f'metrics on slice data\ncategorical feature: {categorical_feature}\n')
        f.write(f'categorical_feature_value: {categorical_feature_value}\n')
        f.write('\n')
        f.write('precision: ')
        f.write(str(precision))
        f.write('\n')
        f.write('recall: ')
        f.write(str(recall))
        f.write('\n')
        f.write('fbeta: ')
        f.write(str(fbeta))
    
    f.close()

def main():
    if len(sys.argv) == 3:
        categorical_feature = sys.argv[1]
        categorical_feature_value = sys.argv[2]

        compute_metrics_on_data_slice(categorical_feature, 
                                  categorical_feature_value,
                                  data=data,
                                  model = load_gbc,
                                  encoder = encoder,
                                  lb = lb
                                  )
    else:
        print("format is incorrect. \nformat example: python .py education Master")

if __name__ =='__main__':
    main()

