import os
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from utils.data_visualization_utils import *
from utils.data_preprocessing_utils import import_dataset, split_features_by_type

"""
Author: Amr Sharafeldin
Created: 2024-11-20
Last Modified: 2024-11-20
Description: This script contains functions to generate and save various types of visualizations for exploratory data analysis (EDA). 
It includes functionalities for creating pie charts for categorical features, word clouds for numerical columns, 
heatmaps for correlation matrices, and histograms for numerical feature distributions. The visualizations are styled for clarity, 
saved to specified output directories, and provide insightful representations of data distributions and relationships.
"""

# Define the plots path
plots_pie_charts_folder = "plots_pie_charts"
plots_wordclouds_folder = "plots_wordcounts"
plots_heatmap_folder = "plots_heatmap"
plots_histogram_folder = "plots_histograms"

# Ensure plots directory exits
os.makedirs(plots_pie_charts_folder, exist_ok=True)
os.makedirs(plots_wordclouds_folder, exist_ok=True)
os.makedirs(plots_heatmap_folder, exist_ok=True)
os.makedirs(plots_histogram_folder, exist_ok=True)


# Import the dataset
X, y = import_dataset()

# Split Categorical and numerical data
numeric_cols, categorical_cols = split_features_by_type(X)

# Perform the data visualization
plot_and_save_pie_charts(X, categorical_cols, plots_pie_charts_folder)
plot_and_save_wordclouds(X, numeric_cols, plots_wordclouds_folder)
plot_and_save_heatmap(numeric_cols, plots_heatmap_folder)
plot_and_save_histograms(X, plots_histogram_folder)




