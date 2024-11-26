import os
import time
import yaml
import logging

from data_preprocessing import get_processed_data
from utils.data_classification_utils import (
    split_to_train_test_dataset,
    train_decision_tree,
    train_SVM_model,
    train_random_forest,
    train_xgboost_model,
    train_knn_model,
    grid_search_xgboost,
    get_validation_score,
    evaluate_and_plot_confusion_matrix,
    evaluate_and_plot_roc
)
from utils.logger_utils import setup_logger

"""
Author: Arvin Bayat Manesh  
Created: 2024-11-23  
Last Modified: 2024-11-23  

Description:  
This script performs training and evaluation of machine learning models on a preprocessed dataset.  
It supports multiple models, grid search for hyperparameter tuning, and GPU acceleration for XGBoost.
"""

def main():

    # Load config file
    config_path = "clf_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup logger
    log_file = config["log_file"]

    # Load and preprocess data
    processed_data, y = get_processed_data(log_file)
    X_train, X_test, y_train, y_test = split_to_train_test_dataset(processed_data, y)

    logger = setup_logger(log_file)

    # Create directories for plots
    plot_curves_file = config["roc_curves_plots_file"]
    confusion_matrix_file = config["confusion_matrix_plots_file"]
    os.makedirs(os.path.dirname(plot_curves_file), exist_ok=True)
    os.makedirs(os.path.dirname(confusion_matrix_file), exist_ok=True)

    # Load model parameters from config
    decision_tree_params = config["models"]["decision_tree"]
    svm_params = config["models"]["svm"]
    random_forest_params = config["models"]["random_forest"]
    xgboost_params = config["models"]["xgboost"]
    knn_params = config["models"]["knn"]
    xgboost_grid_search_params = config["models"]["xgboost_grid_search_params"]
    use_gpu = xgboost_grid_search_params.pop('use_gpu', False)

    # Log the start of the process

    logger.info("Starting model training, cross-validation, and evaluation ...")


    # Train and evaluate models
    models = {
        "Decision Tree": (train_decision_tree, decision_tree_params),
        "SVM": (train_SVM_model, svm_params),
        "Random Forest": (train_random_forest, random_forest_params),
        "XGBoost": (train_xgboost_model, xgboost_params),
        "k-NN": (train_knn_model, knn_params),
    }

    for model_name, (train_function, params) in models.items():
        logger.info(f"Starting {model_name} training...")
        start_time = time.time()

        model, training_score, cv_scores = train_function(X_train, y_train, params)

        logger.info(f"{model_name} - Training Accuracy: {training_score:.4f}")
        logger.info(f"{model_name} - Cross-Validation Scores: {cv_scores}")
        logger.info(f"{model_name} - Mean CV Accuracy: {cv_scores.mean():.4f}")

        validation_score = get_validation_score(model, X_test, y_test)
        logger.info(f"Validation Score for {model_name}: {validation_score:.4f}")

        evaluate_and_plot_confusion_matrix(
            model, X_test, y_test, model_name.replace(" ", "_"), confusion_matrix_file
        )

        evaluate_and_plot_roc(model, X_test, y_test, model_name, plot_curves_file, output_file_suffix="_roc.png")
        logger.info(f"{model_name} completed in {time.time() - start_time:.2f} seconds.")
        logger.info("==========================================")

    # XGBoost with Grid Search
    if use_gpu:
        logger.info("Starting XGBoost Grid Search using device: CUDA (GPU)...")
    else:
        logger.info("Starting XGBoost Grid Search using device: CPU...")

    grid_search_start_time = time.time()
    grid_search_results = grid_search_xgboost(X_train, y_train, xgboost_grid_search_params, use_gpu=use_gpu)

    best_xgboost_model = grid_search_results['best_model']
    best_xgboost_params = grid_search_results['best_params']
    best_xgboost_score = grid_search_results['best_score']
    cv_results = grid_search_results['cv_results']

    logger.info(f"Best XGBoost Parameters: {best_xgboost_params}")
    logger.info(f"Best XGBoost Cross-Validation Score: {best_xgboost_score:.4f}")
    logger.info("Detailed Cross-Validation Results:")
    for mean_score, std_score, params in zip(cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['params']):
        logger.info(f"Score: {mean_score:.4f} (+/- {std_score:.4f}), Params: {params}")

    validation_score_gs_xgb = get_validation_score(best_xgboost_model, X_test, y_test)
    evaluate_and_plot_roc(best_xgboost_model, X_test, y_test, "Best_XGBoost_GridSearch", plot_curves_file, output_file_suffix="_roc.png")
    logger.info(f"Validation Score for Grid Search XGBoost: {validation_score_gs_xgb:.4f}")

    evaluate_and_plot_confusion_matrix(
        best_xgboost_model, X_test, y_test, "XGBoost_GridSearch", confusion_matrix_file
    )
    logger.info(f"XGBoost Grid Search completed in {time.time() - grid_search_start_time:.2f} seconds.")
    logger.info("==========================================")

    logger.info("All models training, cross-validation, and evaluation completed.")
    print("All models training, cross-validation, and evaluation completed. Logs saved in the specified log folder.")

if __name__ == "__main__":
    main()
