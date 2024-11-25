from data_preprocessing import get_processed_data
from utils.data_classification_utils import *

"""
Author: Arvin Bayat Manesh  
Created: 2024-11-23  
Last Modified: 2024-11-23  

Description:  
This script performs training and evaluation of machine learning models on a preprocessed dataset.  
It includes the following functionalities:

- Importing the preprocessed dataset, including features (`X`) and target labels (`y`), using the `get_processed_data` function.
- Splitting the dataset into training and testing sets using a specified ratio to ensure reliable model evaluation.
- Training a Support Vector Machine (SVM) classifier with an RBF kernel on the training dataset.
- Calculating the validation score (accuracy) and generating a detailed classification report for the SVM model.
- Training a Random Forest classifier for multi-output classification tasks.
- Evaluating the Random Forest model by calculating its validation score and generating a classification report.
- Printing validation scores and classification reports for both models to assess their performance.

The script leverages Scikit-Learn for machine learning models, training, evaluation, and metrics calculation.  
It supports multi-output classification tasks and can handle datasets with numerical and categorical features.
"""

# Obtain the processed data for training
processed_data, y = get_processed_data()

# Split the data to train and test datasets
X_train, X_test, y_train, y_test = split_to_train_test_dataset(processed_data, y)

# Train the SVM model
svm_model = train_SVM_model(X_train, y_train)

# Get the validation score
validation_score_svm = get_validation_score(svm_model, X_test, y_test)
print("========================================")
print(f"Validation Score for the SVM model: {validation_score_svm:.2f}")
print("========================================")
print("Classification Report for the SVM model:")
print_classification_report(svm_model, X_test, y_test)

# Train the RandomForest model
random_forest_model = train_random_forest(X_train, y_train)

# Get the validation score
validation_score_random_forest = get_validation_score(random_forest_model, X_test, y_test)
print("========================================")
print(f"Validation Score for the RandomForest model: {validation_score_random_forest:.2f}")
print("========================================")
print("Classification Report for the Random Forest model:")
print_classification_report(random_forest_model, X_test, y_test)



