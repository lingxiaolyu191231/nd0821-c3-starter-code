# Model Card

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
Data is publicly available Census Bureau data. The model can be used to disclose to which degree a person's salary is associated with his or her age, workclass, education, etc. but not directly connected due to the limit number of features presented in the dataset. The model results should be assessed with other data and models to address a more considerate conclusion.

## Caveats and Recommendations
The model results suggest that there is a weak correlation between feature presented in the data, but due to the limit number of samples, I would recommend a much larger dataset ideally with proportions similar to the U.S. population proportions. In addition, I would recommend to include more features not currently presented in the dataset but with promising impact on salary. Last, more types of models and a longer time in fine-tuning would likely bring a higher classification accuracy. 
