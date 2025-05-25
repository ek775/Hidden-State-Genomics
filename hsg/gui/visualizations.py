# helper gui functions for visualizations

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
@st.cache_data
def plot_cross_correlation_lags(corr, lags, title="Cross-Correlation Lags") -> plt.figure:
    """
    Plot the cross-correlation lags.

    Parameters:
    - corr: numpy array of cross-correlation values
    - lags: numpy array of lag values
    - title: title of the plot
    """
    fig = plt.figure()
    plt.plot(lags, corr)
    plt.title(title)
    plt.xlabel("Lags")
    plt.ylabel("Cross-Correlation")
    plt.grid()
    return fig


@st.cache_data
def plot_all_feature_correlations(pearson_scores, title="Fragment Feature Correlations") -> plt.figure:
    """
    Order the correlations from highest to lowest and plot a bar chart representing their values (0,1).
    
    Parameters:
    - pearson_scores: 1-D numpy array of shape (n_features,)
    - title: title of the plot
    """
    pearson_scores = np.sort(pearson_scores)
    fig, ax = plt.subplots()
    plt.bar(range(len(pearson_scores)), pearson_scores)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Pearson Correlation")
    plt.ylim(-1,1)
    plt.grid()
    return fig, ax


@st.cache_data
def plot_best_correlations(pearson_scores, title="Best Feature Correlations") -> plt.figure:
    """
    Plot the top 10 features by pearson correlation.
    """
    fig, ax = plt.subplots()
    ps = pd.DataFrame(pearson_scores)
    sorted_indices = ps[0].nlargest(10).index
    sorted_scores = ps[0].nlargest(10).values
    sorted_indices = [f"f/{idx}" for idx in sorted_indices]
    plt.barh(y=sorted_indices, width=sorted_scores)
    plt.title(title)
    plt.xlabel("Pearson Correlation")
    plt.xlim(0, 1)
    plt.ylabel("Features")
    return fig, ax


def feature_views(suptitle: str, pearson_scores: np.ndarray, xcorr: np.ndarray) -> plt.figure:
    """
    Configure the feature views for the GUI.
    """
    # data
    sorted_pearson_scores = np.sort(pearson_scores)
    ps = pd.DataFrame(sorted_pearson_scores)
    sorted_indices = ps[0].nlargest(5).index
    sorted_scores = ps[0].nlargest(5).values
    sorted_indices = [f"f/{idx}" for idx in sorted_indices]

    # create a grid layout for the plots
    main_fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(nrows=3, ncols=3, wspace=1.5, hspace=1.5)
    main_fig.suptitle(suptitle, fontsize=16)

    # best feature correlations
    ax2 = main_fig.add_subplot(gs[:, -1])
    ax2.barh(y=sorted_indices, width=sorted_scores, color="purple")
    ax2.set_title("Top 5 Features by Pearson Correlation")
    ax2.set_xlabel("Pearson Correlation")
    ax2.set_xlim(0, 1)
    ax2.set_ylabel("Features")
    ax2.grid()

    # all feature correlations
    ax1 = main_fig.add_subplot(gs[0,:-1])
    ax1.bar(range(len(pearson_scores)), pearson_scores)
    ax1.set_title("Fragment Feature Correlations")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Pearson Correlation")
    ax1.set_ylim(-1, 1)
    ax1.grid()

    # active feature correlations
    ax3 = main_fig.add_subplot(gs[1, :-1])
    ax3.bar(range(len(sorted_pearson_scores)), sorted_pearson_scores, color="orange")
    ax3.set_title("Active Feature Correlations")
    ax3.set_xlabel("Features")
    ax3.set_ylabel("Pearson Correlation")
    ax3.set_ylim(-1, 1)
    ax3.grid()

    # cross-correlation lags
    ax4 = main_fig.add_subplot(gs[2, :-1])
    sigs = {}
    for xc in sorted_indices:
        idx = int(xc.split("/")[1])
        sigs[xc] = xcorr[idx]
    for s in sigs:
        ax4.plot(sigs[s], label=s)
    ax4.set_title("Best Feature Cross-Correlation Lags")
    ax4.set_xlabel("Lags")
    ax4.set_ylabel("Cross-Correlation")
    ax4.legend()
    ax4.grid()

    return main_fig


# prevents script output when importing functions
if __name__ == "__main__":
    pass