# helper gui functions for visualizations

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from google.cloud import storage

from io import BytesIO

### OBJECTS ###
@st.cache_resource
class CloudDataHandler:
    """
    Class to handle data retrieval from Google Cloud Storage.
    """
    def __init__(self, bucket_name: str = "hsg-annotation-data"):
        print("Initializing CloudDataHandler...")
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        self.navigator = Navigator()

    def retrieve_array(self, expansion: int, layer: int, track: str, fragment: int):
        expansion = f"ef{expansion}"
        layer = f"layer_{layer}"
        fragment = f"{fragment}.npz"
        # check that fragment is in the bucket
        path = f"{expansion}/{layer}/{track}/{fragment}"
        if not self.bucket.blob(path).exists():
            raise ValueError(f"Fragment {path} not found in bucket {self.bucket_name}.")
        # read the file into memory from the bucket
        blob = self.bucket.get_blob(f"{expansion}/{layer}/{track}/{fragment}")
        with blob.open("rb") as f:
            data = np.load(f)
            pearson_scores = data["pearson_scores"]
            xcorr_arrays = data["xcorr_arrays"]
            print(f"Pearson Data: {type(pearson_scores)} with shape: {pearson_scores.shape}")
            print(f"XCorr Data: {type(xcorr_arrays)} with shape: {xcorr_arrays.shape}")

        return pearson_scores, xcorr_arrays
    
    def list_fragments(self, expansion: int, layer: int, track: str):
        blobs = self.bucket.list_blobs(prefix=f"ef{expansion}/layer_{layer}/{track}/")
        return [blob.name.split("/")[-1][:-4] for blob in blobs if blob.name.endswith(".npz")]
    
class Navigator:
    def __init__(self):
        self.expansions = ["ef8"]
        self.layers = [f"layer_{i}" for i in range(24)]
        with open("data/Annotation Data/tracks.txt", "r") as f:
            self.tracks = [line.strip() for line in f.readlines()]

### FUNCTIONS ###
def plot_embeddings(embeddings, title="HSG Embeddings") -> px.scatter:
    """
    Plot the HSG embeddings using PCA and t-SNE.

    Parameters:
    - embeddings: numpy array of shape (n_samples, n_features)
    - title: title of the plot
    """
    # Perform PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Perform t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(embeddings)

    # Create a scatter plot using Plotly
    fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], title=f"{title} - PCA")
    return fig

def plot_cross_correlation_lags(corr, lags, title="Cross-Correlation Lags") -> plt.figure:
    """
    Plot the cross-correlation lags.

    Parameters:
    - corr: numpy array of cross-correlation values
    - lags: numpy array of lag values
    - title: title of the plot
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(lags, corr)
    plt.title(title)
    plt.xlabel("Lags")
    plt.ylabel("Cross-Correlation")
    plt.grid()
    return fig

def plot_all_feature_correlations(pearson_scores, title="Fragment Feature Correlations") -> plt.figure:
    """
    Order the correlations from highest to lowest and plot a bar chart representing their values (0,1).
    
    Parameters:
    - pearson_scores: 1-D numpy array of shape (n_features,)
    - title: title of the plot
    """
    pearson_scores = np.sort(pearson_scores)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(range(len(pearson_scores)), pearson_scores)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Pearson Correlation")
    plt.grid()
    return fig

def plot_best_correlations(pearson_scores, title="Best Feature Correlations") -> plt.figure:
    """
    Plot the top 10 features by pearson correlation.
    """
    fig = plt.figure(figsize=(10, 4))
    ps = pd.DataFrame(pearson_scores)
    sorted_indices = ps[0].nlargest(10).index
    sorted_scores = ps[0].nlargest(10).values
    sorted_indices = [f"f/{idx}" for idx in sorted_indices]
    plt.barh(y=sorted_indices, width=sorted_scores)
    plt.title(title)
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Features")
    return fig

# prevents script output when importing functions
if __name__ == "__main__":
    pass