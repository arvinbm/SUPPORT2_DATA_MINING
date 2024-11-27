# SUPPORT2_DATA_MINING
This repository provides a comprehensive data preprocessing pipeline for the SUPPORT2 dataset, sourced from the UCI Machine Learning Repository. 




## Instructions to Run the SUPPORT2 Data Preprocessing Pipeline

### 1. Setting Up a Virtual Environment
```
python -m venv support2
source support2/bin/activate  # On Linux/Mac
support2\Scripts\activate     # On Windows
```


### 2. Installing Required Dependencies

```
pip install -r requirements.txt
```

### 3. Running the Preprocessing Script
```
python data_preprocessing.py
```

Outputs
- **Logs**:
  - Logs all preprocessing steps, including handling missing values, feature splitting, normalization, and encoding.
  - File Location: "./logs/data_preprocessing_log/execution_log.txt"
-**Feature Logs**:
  - Logs details of numeric and categorical features in the dataset, including any dropped or encoded features.
  - File Locations:
    - Numeric Features: "./logs/data_preprocessing_log/numeric_columns.csv"
    - Categorical Features: "./logs/data_preprocessing_log/categorical_columns.csv"
    - Missing Values: "./logs/data_preprocessing_log/missing_values.csv"

Key Features of the Preprocessing Pipeline:
-**Dataset Import**:
  - Fetches the SUPPORT2 dataset from the UCI Machine Learning Repository.
-**Feature Analysis**:
  - Logs characteristics of dataset features, including:
    - Number of Missing Values
    - Data Types
    - Unique Values Per Feature
-**Handling Missing Data (Data Imputation)**:
  - Removes columns with more than 30% missing values.
  - Imputes missing values using mean or mode based on feature type and class labels (death and hospdead).
-**Feature Engineering**:
  - Splits features into numerical and categorical subsets.
  - Applies one-hot encoding for categorical features.
  - Normalizes numerical features using z-score normalization.
  - Special handling for features like diabetes and dementia that require no normalization.

### 4. Running the Generate Visualization Script (Exploratory Data Analysis)
```
python data_visualization.py
```

Outputs
- **Pie Charts**:
  - Generates pie charts for categorical features to visualize their distributions.
  - For target labels (`death` and `hospdead`).
  - File Location: "./plots/pie_charts/"
- **Word Clouds**:
  - Creates word clouds for numerical features based on their frequency distributions.
  - File Location: "./plots/word_clouds/"
- **Heatmaps**:
  - Generates a heatmap to visualize feature correlations in the dataset.
  - File Locations: ./plots/heatmap/
- **Histograms**:
  - Saves histograms for numerical features to display their data distributions.
  - File Locations: ./plots/histograms/

**Output Directories**
Ensures necessary directories exist for saving visualizations:
- `plots/pie_charts`
- `plots/word_cloud`
- `plots/heatmap`
- `plots/histograms`

### 5. Running the data_classification script
```
python data_classification.py
```

Outputs
- **ROC Curves**:
  - Plots the Receiver Operating Characteristic (ROC) curves for each trained model to visualize classification performance.
  - File Location: "./logs/clf/training_results.log"
- **Confusion Matrices**:
  - Saves confusion matrices for each trained model to evaluate predictions against true labels.
  - File Location: "./plots/clf/confusion_matrix/"
- **Logs**:
  - Logs the training progress, cross-validation scores, validation scores, and best hyperparameters (for grid search).
  - File Location: "./logs/clf/training_results.log"

### 6. Running the data_clustering script
```
python data_clustering.py
```

Outputs
- **WSS vs. Clusters Plot**:
  - Visualizes the Elbow Method to determine the optimal number of clusters for K-Means.
  - Location: "./plots/clustering/wss_vs_clusters/WSS_vs_Clusters.png"
- **K-Means Clustering Visualization**:
  - A 3D scatter plot of data points clustered by K-Means, with centroids marked.
  - File Location: "./plots/clustering/kmeans/KMeans_Clustering_3D.png"
- **DBSCAN Clustering Visualization**:
  - A 3D scatter plot of data points clustered by DBSCAN, highlighting core samples, non-core samples, and noise points.
  - File Location: "./plots/clustering/dbscan/dbscan_clustering_3d.png"
- **Silhouette Scores**:
  - Logs the Silhouette scores for K-Means and DBSCAN to evaluate clustering performance.
  - File Location: "./logs/clustering_results.log"
- **Logs**:
  - Logs the progress and results of clustering.
  - File Location: "./logs/clustering_results.log"

### 7. Running the outlier_detection script
```
python outlier_detection.py
```
Outputs
- **Isolation Forest Outlier Detection Visualization**:
  - A 3D scatter plot showing inliers and outliers detected by the Isolation Forest algorithm.
  - File Location: "./plots/outlier_detection/isolation_forest/isolation_forest_3d.png"
- **Elliptic Envelope Outlier Detection Visualization**:
  - A 3D scatter plot showing inliers and outliers detected by the Elliptic Envelope algorithm.
  - File Location: "./plots/outlier_detection/elliptic_envelope/elliptic_envelope_3d.png"
- **Logs**:
  - Logs the progress and results of the outlier detection methods.
  - File Location: "./logs/outlier_detection.log"

---

## Data Preprocessing

---

## Data Classification
### SVM (Radial Basis Function kernel)
### Random Forest
### K-Nearest Neighbors
### XGBoost
### Grid Search

---

## Data Clustering
### K-Means
### DBSCAN

---

## Outlier Detection
### Isolation Forest
### Elliptic Envelope


