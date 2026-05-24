# SUPPORT2 Data Mining

A full data mining pipeline applied to the **SUPPORT2** dataset (UCI ML Repository, ID 880) — a clinical study of seriously ill hospitalized adults. The pipeline covers every major stage of a data mining workflow: preprocessing, exploratory visualization, feature selection, classification, clustering, and outlier detection, with an optional deep learning comparison at the end.

This was developed as a course project for **CMPT 459 (Data Mining)** at Simon Fraser University.

![Project Poster](CMPT_459(Poster).jpg)

---

## About the Dataset

The **SUPPORT2** dataset contains data from ~9,000 critically ill patients across multiple U.S. hospitals, collected as part of the Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments (SUPPORT). Features include physiological measurements, demographic information, comorbidities, and lab results. The primary prediction targets are in-hospital death (`hospdead`) and six-month mortality (`death`).

Dataset source: [UCI ML Repository — SUPPORT2](https://archive.ics.uci.edu/dataset/880/support2)

---

## Project Structure

```
SUPPORT2_DATA_MINING/
  data_preprocessing.py       # Core preprocessing pipeline (entry point for all scripts)
  data_visualization.py       # Exploratory data analysis — charts and plots
  feature_selection.py        # Mutual information feature ranking
  data_classification.py      # Classification models (6 algorithms + XGBoost grid search)
  data_clustering.py          # K-Means and DBSCAN clustering
  outlier_detection.py        # Isolation Forest and Elliptic Envelope outlier detection
  deep_L.ipynb                # Bonus: FFNN and NODE deep learning comparison (Colab)

  clf_config.yaml             # Hyperparameters for classification models
  clu_config.yaml             # Config for clustering algorithms
  out_config.yaml             # Config for outlier detection
  featSel_config.yaml         # Config for feature selection

  utils/
    data_preprocessing_utils.py
    data_visualization_utils.py
    data_classification_utils.py
    data_clustering_utils.py
    outlier_detection_utils.py
    func_utils.py             # Mutual information, label mapping helpers
    logger_utils.py           # Shared logging setup

  requirements.txt
  report.pdf                  # Full project report
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
git clone https://github.com/arvinbm/SUPPORT2_DATA_MINING.git
cd SUPPORT2_DATA_MINING
python -m venv support2
source support2/bin/activate   # macOS / Linux
support2\Scripts\activate      # Windows
pip install -r requirements.txt
```

The dataset is fetched automatically from the UCI ML Repository at runtime via the `ucimlrepo` package — no manual download needed.

---

## Pipeline Overview

All scripts share the same preprocessing entry point (`data_preprocessing.py → get_processed_data()`), which handles dataset fetching, imputation, encoding, and normalization before handing data off to the downstream script.

```
UCI ML Repository
       ↓
data_preprocessing.py  ──► data_visualization.py
                       ──► feature_selection.py
                       ──► data_classification.py
                       ──► data_clustering.py
                       ──► outlier_detection.py
```

---

## Scripts

### 1. Preprocessing

```bash
python data_preprocessing.py
```

The preprocessing pipeline does the following:

- **Fetches** the SUPPORT2 dataset from the UCI ML Repository
- **Analyzes** each feature: missing value counts, data types, unique value cardinality
- **Drops** columns with more than 30% missing values
- **Imputes** remaining missing values by mean (numeric) or mode (categorical), stratified by class label
- **Encodes** categorical features with one-hot encoding
- **Normalizes** numeric features with z-score standardization (except binary flags like `diabetes` and `dementia`)
- **Logs** all steps and saves feature summaries to CSV

**Outputs:**

| File | Description |
|---|---|
| `logs/data_preprocessing_log/execution_log.txt` | Step-by-step preprocessing log |
| `logs/data_preprocessing_log/numeric_columns.csv` | Numeric feature summary |
| `logs/data_preprocessing_log/categorical_columns.csv` | Categorical feature summary |
| `logs/data_preprocessing_log/missing_values.csv` | Missing value counts per feature |

---

### 2. Visualization (EDA)

```bash
python data_visualization.py
```

Generates exploratory plots to understand feature distributions and correlations before modeling.

**Outputs:**

| Plot | Location |
|---|---|
| Pie charts (categorical feature distributions + target labels) | `plots/pie_charts/` |
| Word clouds (numeric feature frequency distributions) | `plots/word_clouds/` |
| Correlation heatmap | `plots/heatmap/` |
| Histograms (numeric features) | `plots/histograms/` |

---

### 3. Feature Selection

```bash
python feature_selection.py
```

Computes **mutual information** scores between each feature and the target label, then plots the ranked feature importances as a bar chart. The top-N features identified here can be passed directly to `data_classification.py` via the `--mode FElim -n N` flags.

**Outputs:**

| File | Description |
|---|---|
| `plots/feature_selection/mutual_information_scores.png` | Ranked feature importance bar chart |
| `logs/feature_selection/feature_selection.log` | Execution log |

---

### 4. Classification

```bash
python data_classification.py [--mode {no_FElim,FElim}] [-n N] [-g]
```

Trains and evaluates six classification models against the preprocessed SUPPORT2 data. Model hyperparameters are loaded from `clf_config.yaml`.

**Models trained:**
- Decision Tree
- Support Vector Machine (RBF kernel)
- Random Forest
- XGBoost
- k-Nearest Neighbors
- Naive Bayes (Gaussian)

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--mode` | `no_FElim` | `FElim` selects the top N features by mutual information before training; `no_FElim` uses all features |
| `-n`, `--number_of_features` | `6` | Number of top features to keep when `--mode FElim` is set |
| `-g`, `--grid_searchXG` | `False` | Run an exhaustive grid search for XGBoost hyperparameters instead of the default config values |

**Examples:**

```bash
# Train all models with all features (uses clf_config.yaml hyperparameters)
python data_classification.py

# Reduce to top 8 features by mutual information, then train
python data_classification.py --mode FElim -n 8

# XGBoost grid search only (all features)
python data_classification.py -g

# Feature elimination + XGBoost grid search
python data_classification.py --mode FElim -n 5 -g
```

**Outputs:**

| File | Description |
|---|---|
| `logs/clf/training_results.log` | Training accuracy, CV scores, validation scores, classification reports |
| `plots/clf/roc_curves/` | ROC curve per model |
| `plots/clf/confusion_matrix/` | Confusion matrix per model |

---

### 5. Clustering

```bash
python data_clustering.py
```

Clusters the preprocessed data using two unsupervised algorithms. Parameters for DBSCAN are loaded from `clu_config.yaml`. PCA is used to reduce dimensionality to 3D for visualization.

**Algorithms:**
- **K-Means** — optimal cluster count determined by the Elbow Method (WSS vs. clusters plot); currently set to 2
- **DBSCAN** — density-based clustering; parameters `eps=5`, `min_samples=5` (grid-searched offline)

**Outputs:**

| File | Description |
|---|---|
| `plots/clustering/wss_vs_clusters/WSS_vs_Clusters.png` | Elbow method plot |
| `plots/clustering/kmeans/KMeans_Clustering_3D.png` | 3D K-Means scatter with centroids |
| `plots/clustering/dbscan/dbscan_clustering_3d.png` | 3D DBSCAN scatter (core, non-core, noise) |
| `logs/clustering/clustering.log` | Silhouette scores + execution log |

---

### 6. Outlier Detection

```bash
python outlier_detection.py
```

Detects anomalous patient records using two methods. Parameters are loaded from `out_config.yaml`. PCA reduces data to 2D for visualization.

**Methods:**
- **Isolation Forest** — ensemble of random trees that isolate anomalies; `n_estimators=1000`
- **Elliptic Envelope** — fits a Gaussian distribution to the data and flags points outside the envelope; `contamination=0.1`

**Outputs:**

| File | Description |
|---|---|
| `plots/outlier_detection/isolation_forest/` | 3D scatter: inliers vs. outliers (Isolation Forest) |
| `plots/outlier_detection/elliptic_envelope/` | 3D scatter: inliers vs. outliers (Elliptic Envelope) |
| `logs/outlier_detection/outlier_detection.log` | Execution log |

---

## Configuration Files

All scripts load their parameters from YAML files, making it easy to tune algorithms without touching code.

| File | Used by | Key parameters |
|---|---|---|
| `clf_config.yaml` | `data_classification.py` | Per-model hyperparameters, XGBoost grid search grid, output paths |
| `clu_config.yaml` | `data_clustering.py` | DBSCAN `eps` and `min_samples`, output paths |
| `out_config.yaml` | `outlier_detection.py` | Isolation Forest `n_estimators`, Elliptic Envelope `contamination`, output paths |
| `featSel_config.yaml` | `feature_selection.py` | Output paths |

---

## Bonus: XGBoost Is All You Need?

`deep_L.ipynb` compares XGBoost against two neural network approaches on the same SUPPORT2 data:

- **FeedForward Neural Network (FFNN)** — a standard dense network trained on the preprocessed feature matrix
- **Neural Oblivious Decision Ensembles (NODE)** — a differentiable tree ensemble architecture designed specifically for tabular data

The notebook is intended to run in **Google Colab** (GPU recommended for NODE). Open it directly:
[Open in Colab](https://colab.research.google.com/github/arvinbm/SUPPORT2_DATA_MINING/blob/master/deep_L.ipynb)

Full methodology, results, and analysis are in [`report.pdf`](report.pdf).

---

## Acknowledgments

Thanks to **Professor Martin Ester** and **Arash Khoeini** for their supervision and guidance throughout this project (CMPT 459, Simon Fraser University).
