from data_preprocessing import get_processed_data
from utils.data_training_utils import *

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



