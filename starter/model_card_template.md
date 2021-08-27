# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Author: Lingxiao Lyu & Udacity Machine Learning DevOps Nano Degree
Date: August 27, 2021
Model Implemented: Gradient Boosted Classifier
Contact info:
    - email: lingxiaolyu19@gmail.com

## Intended Use
Primary Intended Uses: \
    - This model was developed as one of the Udacity Machine Learning DevOps Nano Degree projects, which allowed learners to apply skills including developing a machine learning pipeline and deploying it in real time using FastAPI, AWS S3, DVC, and Git

Primiary Intended Users: 
    - For academic training purpose

Out-of-scope Uses:
    - Not for large-scale datasets as FastAPI limit number of requests sent to the pipeline
    - Not for enterprise usage

## Training Data
- The model was trained on 80% of the data, approximately 25,000+ samples, and was split by the train_test_split method in the sklearn package with a random seed of 42.
- The model was trained using gridsearch to find the best set of hyperparameters.

## Evaluation Data
- After trained, the best model was selected and then was evaluated on the rest 20% of the data, approximately 6000+ samples. 

## Metrics
### Metrics used for model evaluation:
- precision: calculated as (number of true positives) / (number of all labels with positive predictions)
- recall: calculated as (number of true positives) / (number of all labels with real positive )
- fbeta: the weighted harmonic mean of precision and recall; in this model, weight on recall = 0.7 and weight on precision = 0.3

## Ethical Considerations
Data is publicly available Census Bureau data. The model can be used to disclose whether a person's salary is associated with his or her 
## Caveats and Recommendations
