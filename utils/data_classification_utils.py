import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC

"""
Author: Arvin Bayat Manesh
Created: 2024-11-23
Last Modified: 2024-11-23

Description: This script provides functionality for training and evaluating machine learning models on a dataset.
It includes the following functionalities:

- Splitting the dataset into training and testing sets based on specified target columns (e.g., 'death' and 'hospdead').
- Training a Support Vector Machine (SVM) model with an RBF kernel for classification tasks.
- Training a Random Forest classifier to handle multi-output or multi-class classification tasks.
- Evaluating trained models using a validation score (accuracy) and generating classification reports.

The script leverages scikit-learn's machine learning library and supports parallel processing for faster training using Random Forest.
"""

def split_to_train_test_dataset(X, y):
    """
    Splits the input dataset into training and testing sets based on the specified labels.

    Parameters:
        X (pd.DataFrame): The feature DataFrame containing the input variables.
        y (pd.DataFrame): The target DataFrame containing the label columns.

    Returns:
        tuple: A tuple containing four elements:
            - X_train (pd.DataFrame): Training set features.
            - X_test (pd.DataFrame): Testing set features.
            - y_train (pd.DataFrame): Training set labels (selected columns).
            - y_test (pd.DataFrame): Testing set labels (selected columns).
    """

    selected_columns_for_labels = y[['death', 'hospdead']]
    
    # Split X and the selected y columns into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, selected_columns_for_labels,
                                                        test_size=0.20, random_state=13, stratify=selected_columns_for_labels)
    
    return X_train, X_test, y_train, y_test

def train_SVM_model(X_train, y_train):
    """
    Trains a Support Vector Machine (SVM) model using a kernel-based approach (RBF kernel by default).

    Parameters:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training labels.

    Returns:
        sklearn.svm.SVC: The trained SVM model.
    """
    # Initialize SVM and wrap it for multi-output classification
    base_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    multi_output_svm = MultiOutputClassifier(base_svm)
    
    # Train the SVM model
    multi_output_svm.fit(X_train, y_train)
    
    return multi_output_svm

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier using scikit-learn.

    Parameters:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training labels.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained Random Forest model.
    """
    
    # Initialize the Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        max_depth=None, 
        random_state=13,
        n_jobs=-1       
    )
    
    # Train the Random Forest model
    rf_model.fit(X_train, y_train)
    
    return rf_model


def get_validation_score(model, X_test, y_test):
    return model.score(X_test, y_test)

def print_classification_report(model, X_test, y_test):
    # Make predictions using the model
    y_pred = model.predict(X_test)

    # Print the report
    print(classification_report(y_pred, y_test, zero_division=0))