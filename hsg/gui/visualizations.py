# helper gui functions for visualizations

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from google.cloud import storage

### OBJECTS ###
class CloudDataHandler:
    def __init__(self, bucket_name: str = "hsg-annotation-data"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        self.navigator = Navigator()

    def retrieve_array(self, expansion: int, layer: int, track: str, fragment: int):
        expansion = f"ef{expansion}"
        layer = f"layer_{layer}"
        fragment = f"{fragment}.npz"
        # check that fragment is in the bucket
        if not self.bucket.blob(f"{expansion}/{layer}/{track}/{fragment}").exists():
            raise ValueError(f"Fragment {fragment} not found in bucket {self.bucket_name}.")
        # read the file into memory from the bucket
        blob = self.bucket.blob(f"{expansion}/{layer}/{track}/{fragment}")
        with blob.open("rb") as f:
            pearson_scores, xcorr_arrays = np.load(f, allow_pickle=True)

        return pearson_scores, xcorr_arrays
    
    def list_fragments(self, expansion: int, layer: int, track: str):
        blobs = self.bucket.list_blobs(prefix=f"ef{expansion}/layer_{layer}/{track}/")
        return [blob.name.split("/")[-1][:-4] for blob in blobs if blob.name.endswith(".npz")]
    
class Navigator:
    def __init__(self):
        self.expansions = ["ef8"]
        self.layers = [f"layer_{i}" for i in range(24)]
        with open("data/Annotation\ Data/tracks.txt", "r") as f:
            self.tracks = [line.strip() for line in f.readlines()]

### FUNCTIONS ###
def plot_embeddings(embeddings, labels, title="HSG Embeddings") -> px.scatter:
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

# prevents script output when importing functions
if __name__ == "__main__":
    pass