import os
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
Author: Arvin Bayat Manesh 
Created: 2024-11-25
Last Modified: 2024-11-25  

Description:  
This module provides utility functions for performing Isolation Forest-based outlier detection and visualizing the results.  
It includes implementations for applying the Isolation Forest algorithm, reducing data dimensionality using PCA,  
and plotting the results to help interpret outliers in the dataset.

The functionalities include:
- Applying Isolation Forest for outlier detection.
- Dimensionality reduction using PCA to prepare data for visualization.
- Plotting and saving the results of outlier detection for easy interpretation.
"""

def perform_isolation_forest(X, num_estimators):
    isolation_forest = IsolationForest(n_estimators=num_estimators, random_state=13, contamination='auto')
    isolation_forest.fit(X)
    predictions = isolation_forest.predict(X)
    return predictions

def perform_pca(X):
    pca = PCA(n_components=3)
    reduced_2d_dataset = pca.fit_transform(X)
    return reduced_2d_dataset

def plot_outlier_detection_results(predictions, reduced_3d_dataset, output_folder, method="Isolation Forest"):
    """
    Creates a 3D scatter plot of inliers and outliers detected by a specified method.

    Parameters:
        predictions (numpy array): Array of predictions, where 1 represents inliers and -1 represents outliers.
        reduced_3d_dataset (numpy array): 3D reduced dataset (e.g., from PCA or t-SNE) for visualization.
        output_folder (str): Path to the folder where the plot will be saved.
        method (str): Outlier detection method used, e.g., "Isolation Forest" or "Elliptic Envelope".
    """
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separate inliers and outliers for plotting
    inliers = reduced_3d_dataset[predictions == 1]
    outliers = reduced_3d_dataset[predictions == -1]

    # Plot inliers
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], 
               c='blue', label='Inliers', alpha=0.5, s=20)

    # Plot outliers
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], 
               c='red', label='Outliers', alpha=0.5, s=20)

    # Set plot labels and title
    ax.set_title(f"{method}: Outlier Detection Visualization (3D Reduced Data)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend()

    # Save the plot
    plot_filename = f"{method.lower().replace(' ', '_')}_3d.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"3D Plot saved to {plot_path}")



def apply_elliptic_envelope(data, contamination=0.1):
    """
    Perform outlier detection using Elliptic Envelope.

    Parameters:
    - data: pd.DataFrame, the input data for outlier detection.
    - contamination: float, the proportion of outliers in the data.

    Returns:
    - cleaned_data: pd.DataFrame, data without outliers.
    - outliers: pd.DataFrame, data identified as outliers.
    """
    envelope = EllipticEnvelope(contamination=contamination, random_state=42)
    envelope.fit(data)

    labels = envelope.predict(data)
    return labels