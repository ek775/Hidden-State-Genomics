# helper gui functions for visualizations

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from google.cloud import storage

### OBJECTS ###
class CloudDataHandler:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)

    def download_blob(self, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        blob = self.bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

    def upload_blob(self, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

### FUNCTIONS ###
def plot_embeddings(embeddings, labels, title="HSG Embeddings"):
    """
    Plot the HSG embeddings using PCA and t-SNE.

    Parameters:
    - embeddings: numpy array of shape (n_samples, n_features)
    - labels: list of labels for each sample
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
    fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], color=labels, title=f"{title} - PCA")
    fig.show()

def plot_cross_correlation_lags(corr, lags, title="Cross-Correlation Lags"):
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

# prevents script output when importing functions
if __name__ == "__main__":
    pass