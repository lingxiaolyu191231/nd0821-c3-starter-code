# Script to train machine learning model.
import sys
from model import compute_model_metrics, inference
from data import process_data
import pandas as pd
import pickle

sys.path.insert(1, './ml')

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
    "native-country"]


def compute_metrics_on_data_slice(
        categorical_feature,
        data=data,
        model=load_gbc,
        encoder=encoder,
        lb=lb):
    feature_value_list = []
    precision_value = []
    recall_value = []
    fbeta_value = []
    n = len(data[categorical_feature].unique())
    for f_value in data[categorical_feature].unique():
        data_slice = data.loc[data[categorical_feature] == f_value, ]
        X_slice, y_slice, _, _ = process_data(data_slice,
                                              cat_features,
                                              label='salary', training=False,
                                              encoder=encoder, lb=lb)

        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        feature_value_list.append(f_value)
        precision_value.append(precision)
        recall_value.append(recall)
        fbeta_value.append(fbeta)

    with open('../screenshots/slice_output.txt', 'w') as f:
        f.write('metrics on slice data\ncategorical feature')
        f.write(str(categorical_feature))
        f.write('\n\n')
        for i in range(n):
            f.write(f'categorical_feature_value: {feature_value_list[i]}')
            f.write('\n')
            f.write('precision: ')
            f.write(str(precision_value[i]))
            f.write('\n')
            f.write('recall: ')
            f.write(str(recall_value[i]))
            f.write('\n')
            f.write('fbeta: ')
            f.write(str(fbeta_value[i]))
            f.write('\n')
            f.write('\n')

    f.close()


def main():
    if len(sys.argv) == 2:
        categorical_feature = sys.argv[1]
        compute_metrics_on_data_slice(
            categorical_feature,
            data=data,
            model=load_gbc,
            encoder=encoder,
            lb=lb)
    else:
        print("format is incorrect. \nformat example: python .py education")


if __name__ == '__main__':
    main()
