import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc , accuracy_score

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from .func_utils import map_labels


from sklearn.multiclass import OneVsRestClassifier
"""
Author: Arvin Bayat Manesh (split_to_train_test_dataset , train_SVM_model , train_random_forest )
Author: Amr Sharafeldin (train_decision_tree , train_xgboost_model  , xgboos grid search )

Created: 2024-11-23
Last Modified: 2024-11-24 

Description: This script provides functionality for training and evaluating machine learning models on a dataset.
It includes the following functionalities:

- Splitting the dataset into training and testing sets based on specified target columns (e.g., 'death' and 'hospdead').
- Training a Support Vector Machine (SVM) model with an RBF kernel for classification tasks.
- Training a Random Forest classifier to handle multi-output or multi-class classification tasks.
- Evaluating trained models using a validation score (accuracy) and generating classification reports.

The script leverages scikit-learn's machine learning library and supports parallel processing for faster training using Random Forest.
"""

def split_to_train_test_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20, random_state=13, stratify=y)

    return X_train, X_test, y_train, y_test


# Baselines : 

def train_decision_tree(X_train, y_train, params):

    dt_model = DecisionTreeClassifier(
        criterion=params.get('criterion', 'gini'),
        splitter=params.get('splitter', 'best'),
        max_depth=params.get('max_depth', None),
        min_samples_split=params.get('min_samples_split', 2),
        random_state=params.get('random_state', 13)
    )
    cross_val_scores = cross_val_score(dt_model, X_train, y_train, cv=params.get('cv', 5), scoring='accuracy')
    dt_model.fit(X_train, y_train)

    y_train_pred = dt_model.predict(X_train)

    train_score = accuracy_score(y_train, y_train_pred)

    return dt_model, train_score, cross_val_scores

#https://scikit-learn.org/1.5/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
def train_SVM_model(X_train, y_train, params):

    svm_model = SVC(
        kernel=params.get('kernel', 'rbf'),
        C=params.get('C', 1.0),
        gamma=params.get('gamma', 'scale'),
        probability=True,
        class_weight=params.get('class_weight', None)  # Handle class imbalance
    )
    ovr_model = OneVsRestClassifier(svm_model)
    cross_val_scores = cross_val_score(ovr_model, X_train, y_train, cv=params.get('cv', 5), scoring='accuracy')
    ovr_model.fit(X_train, y_train)

    y_train_pred = ovr_model.predict(X_train)

    train_score = accuracy_score(y_train, y_train_pred)

    return ovr_model, train_score, cross_val_scores



def train_knn_model(X_train, y_train, params):

    knn_clf = KNeighborsClassifier(
    n_neighbors=params.get('n_neighbors', 5),
    metric=params.get('metric', 'minkowski')  # Default is 'minkowski'
)
    cross_val_scores = cross_val_score(knn_clf, X_train, y_train, cv=params.get('cv', 5), scoring='accuracy')
    knn_clf.fit(X_train, y_train)

    y_train_pred = knn_clf.predict(X_train)
    train_score = accuracy_score(y_train, y_train_pred)

    return knn_clf, train_score, cross_val_scores



def train_naive_bayes(X_train, y_train, params):
    # Initialize Gaussian Naive Bayes model
    nb_model = GaussianNB(
        var_smoothing=params.get('var_smoothing', 1e-9)  # Add var_smoothing as the main parameter
    )
    
    # Perform cross-validation
    cross_val_scores = cross_val_score(nb_model, X_train, y_train, cv=params.get('cv', 5), scoring='accuracy')
    
    # Fit the model
    nb_model.fit(X_train, y_train)
    
    # Predict on training data
    y_train_pred = nb_model.predict(X_train)
    
    # Calculate training accuracy
    train_score = accuracy_score(y_train, y_train_pred)
    
    return nb_model, train_score, cross_val_scores


def train_random_forest(X_train, y_train, params):

    rf_model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        min_samples_split=params.get('min_samples_split', 2),
        max_depth=params.get('max_depth', None),
        random_state=params.get('random_state', 13),
        n_jobs=params.get('n_jobs', -1),
        verbose=params.get('verbose', 1)  # Add verbose with a default value of 0
    )
    cross_val_scores = cross_val_score(rf_model, X_train, y_train, cv=params.get('cv', 5), scoring='accuracy')
    rf_model.fit(X_train, y_train)

    y_train_pred = rf_model.predict(X_train)

    train_score = accuracy_score(y_train, y_train_pred)

    return rf_model, train_score, cross_val_scores



def train_xgboost_model(X_train, y_train, params):

    xgb_model = XGBClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        random_state=params.get('random_state', 13),
        use_label_encoder=params.get('use_label_encoder', False),
        eval_metric=params.get('eval_metric', 'logloss'),
        verbosity=params.get('verbosity', 1)  
    )
    cross_val_scores = cross_val_score(xgb_model, X_train, y_train, cv=params.get('cv', 5), scoring='accuracy')
    xgb_model.fit(X_train, y_train)

    y_train_pred = xgb_model.predict(X_train)

    train_score = accuracy_score(y_train, y_train_pred)

    return xgb_model, train_score, cross_val_scores




def grid_search_xgboost(X_train, y_train, param_grid, cv=5, scoring='accuracy', use_gpu=True):

    if use_gpu:
        param_grid['tree_method'] = ['hist']
        param_grid['device'] = ['cuda']
    else:
        param_grid['tree_method'] = ['hist']

    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,  
        verbose=1  
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': grid_search.cv_results_
    }



def get_validation_score(model, X_test, y_test):
    return model.score(X_test, y_test)

def print_classification_report(model, X_test, y_test):
    # Make predictions using the model
    y_pred = model.predict(X_test)

    # Print the report
    print(classification_report(y_pred, y_test, zero_division=0))




def evaluate_and_plot_roc(model, X_test, y_test, model_name, output_folder=".", output_file_suffix="_roc.png"):
    classes = np.unique(y_test)

    y_score = model.predict_proba(X_test)

    y_test_binarized = label_binarize(y_test, classes=classes)
    n_classes = y_test_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette('hsv', n_classes)

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Multi-Class Classification ({model_name})')
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f"{model_name}{output_file_suffix}")

    plt.savefig(output_file)
    plt.close()




def evaluate_and_plot_confusion_matrix(model, X_test, y_test, model_name, output_folder=".", output_file_suffix="_confusion_matrix.png"):

    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)

    plt.title(f"Confusion Matrix - {model_name}")

    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f"{model_name}{output_file_suffix}")

    # Save the plot
    plt.savefig(output_file)
    plt.close()
    
