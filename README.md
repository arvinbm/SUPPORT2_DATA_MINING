# SUPPORT2_DATA_MINING
This repository provides a comprehensive data preprocessing pipeline for the SUPPORT2 dataset, sourced from the UCI Machine Learning Repository. 




## Instructions to Run the SUPPORT2 Data Preprocessing Pipeline

### 1. Set Up a Virtual Environment
```
python -m venv support2
source support2/bin/activate  # On Linux/Mac
support2\Scripts\activate     # On Windows
```


### 2. Install Required Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Preprocessing Script
```
python data_preprocessing.py
```

1. **Import the Dataset**  
   The script imports the SUPPORT2 dataset, preparing it for preprocessing.

2. **Perform Preprocessing**  
   - Handles missing values.
   - Applies necessary data transformations to prepare the dataset for analysis or modeling.

3. **Save Logs**  
   - Logs all preprocessing steps and outcomes in the `logs/data_preprocessing_log` directory.

4. **Display Processed Data**  
   - Outputs the first few rows of the processed dataset to the console for quick verification.

### 4. Run the Generate Visualization Script
```
python data_visualization.py.py
```

Generates and saves multiple types of visualizations:
- **Pie Charts**:
  - For categorical features.
  - For target labels (`death` and `hospdead`).
- **Word Clouds**:
  - For numerical features.
- **Heatmaps**:
  - Visualizes the entire dataset to understand feature correlations.
- **Histograms**:
  - For numerical features to observe data distributions.

**Output Directories**
Ensures necessary directories exist for saving visualizations:
- `plots/word_cloud`
- `plots/heatmap`
- `plots/histograms`

---
