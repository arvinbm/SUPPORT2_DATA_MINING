import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

"""
Author: Amr Sharafeldin
Created: 2024-11-20
Last Modified: 2024-11-20
Description: This script contains functions to generate and save various types of visualizations for exploratory data analysis (EDA). 
It includes functionalities for creating pie charts for categorical features, word clouds for numerical columns, 
heatmaps for correlation matrices, and histograms for numerical feature distributions. The visualizations are styled for clarity, 
saved to specified output directories, and provide insightful representations of data distributions and relationships.
"""

def plot_and_save_pie_charts(df, categorical_columns, folder_path="plots/pie_charts"):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for col in categorical_columns:
        if col in df.columns:
            data = df[col].value_counts()
            
            custom_colors = plt.cm.Paired(range(len(data))) 
            
            plt.figure(figsize=(8, 8))
            wedges, texts, autotexts = plt.pie(
                data, 
                labels=None, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=custom_colors,  
                wedgeprops={'edgecolor': 'white'},  
                pctdistance=0.85  
            )
            plt.title(f"Distribution of {col}", fontsize=14)
            
            centre_circle = plt.Circle((0, 0), 0.7, fc='white', edgecolor='black', linewidth=1.5)
            plt.gca().add_artist(centre_circle)
            
            plt.legend(
                handles=wedges,
                labels=[f"{label} ({count})" for label, count in zip(data.index, data)],  
                title=col.capitalize(),
                loc='center left',
                bbox_to_anchor=(1, 0.5),  
                frameon=True,  
                shadow=False
            )
            
            save_path = os.path.join(folder_path, f"{col}_pie_chart_with_custom_style.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            print(f"Column '{col}' not found in the DataFrame.")


def plot_and_save_wordclouds(data, numerical_columns, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for column in numerical_columns:
        if column not in data.columns:
            print(f"Column '{column}' does not exist in the dataset.")
            continue
        
        data[column] = data[column].round(decimals=2)
        counts = data[column].value_counts()
        counts.index = counts.index.map(str)

        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color='white',
            collocations=False,
            max_font_size=100,
            min_font_size=50,
            contour_width=0.5,
            contour_color='black'
        ).generate_from_frequencies(counts)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{column} Word Cloud")
        plot_path = os.path.join(output_folder, f"{column}_wordcloud.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Word cloud for '{column}' saved at: {plot_path}")
    

def plot_and_save_heatmap(data, output_folder, title="Heatmap", dpi=300):

    os.makedirs(output_folder, exist_ok=True)
    
    correlation_matrix = data.corr()
    
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=False,  
        cmap="coolwarm", 
        square=True,  
        cbar_kws={"shrink": 0.75}  
    )
    plt.title(title)
    plt.tight_layout()
    
    plot_path = os.path.join(output_folder, "heatmap.png")
    plt.savefig(plot_path, dpi=dpi)
    plt.close()
    print(f"Heatmap saved at: {plot_path}")


def plot_and_save_histograms(data, output_folder, bins=30, dpi=300):

    os.makedirs(output_folder, exist_ok=True)
    
    for column in data.select_dtypes(include=["float", "int"]).columns:
        plt.figure(figsize=(8, 6))
        
        sns.histplot(data[column], bins=bins, kde=True, color="blue", alpha=0.7)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        
        plot_path = os.path.join(output_folder, f"{column}_histogram.png")
        plt.savefig(plot_path, dpi=dpi)
        plt.close()
        print(f"Histogram saved for {column} at: {plot_path}")
