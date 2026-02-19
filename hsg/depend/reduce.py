import numpy as np


# Helper functions for reducing dependency maps (nxn matrices) to n-dimensional vectors or scalars, for use in HSG algorithms.
def minmax_normalize(array: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to the range [0, 1].
    Args:
        array: NumPy array of any shape (1D vector, 2D matrix, etc.)
        
    Returns:
        Normalized array with same shape, values in [0, 1]
    """
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return np.zeros_like(array)  # Avoid division by zero
    return (array - min_val) / (max_val - min_val)

##############################################################################
# Matrix reduction functions
##############################################################################
def diagonal(matrix: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Return the diagonal elements of a matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    diag = np.diag(matrix)
    if normalize:
        diag = minmax_normalize(diag)
    return diag

def rowprojection(matrix: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Project a vector onto the row space of a matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    vector = np.ones(matrix.shape[0])
    result = np.dot(matrix, vector)
    if normalize:
        result = minmax_normalize(result)
    return result

def columnprojection(matrix: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Project a vector onto the column space of a matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    vector = np.ones(matrix.shape[0])
    result = np.dot(vector, matrix)
    if normalize:
        result = minmax_normalize(result)
    return result


###############################################################################
# Graph/Network reduction functions (Dep. Maps Considered as Adjacency matrices)
###############################################################################
def eigenvec(matrix: np.ndarray) -> np.ndarray:
    """Return the eigenvector centrality of a graph represented by an adjacency matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Find the index of the largest real eigenvalue
    max_idx = np.argmax(eigenvalues.real)
    
    # Extract the corresponding eigenvector
    principal_eigenvector = eigenvectors[:, max_idx].real
    
    # Take absolute values and normalize to unit L2 norm
    centrality = np.abs(principal_eigenvector)
    centrality = centrality / np.linalg.norm(centrality)
    
    return centrality

##################################################################################
# Information diffusion
##################################################################################

def probatransition(matrix: np.ndarray) -> np.ndarray:
    """Return the random walk transition probabilities of a graph (directed) represented by an adjacency matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    degree = np.sum(matrix, axis=1)
    transition = matrix / degree[:, np.newaxis]
    return transition
    
def laplacian(matrix: np.ndarray) -> np.ndarray:
    """Return the Laplacian of a graph represented by an adjacency matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    degree = np.sum(matrix, axis=1)
    laplacian = np.diag(degree) - matrix
    return laplacian



###################################################################################
# Test
###################################################################################

if __name__ == "__main__":
    from hsg.depend.featurewise import calculate_position_dependency_map
    from hsg.stattools.features import get_latent_model
    import os
    
    sequence = "CAGGAAGGAGGATGGAGGCTGGTGGGAGGG"
    feature_id = 3378
    
    # Load model
    print("Loading nucleotide transformer and SAE...")
    nt_model_path = os.getenv("NT_MODEL")
    sae_checkpoint = "checkpoints/hidden-state-genomics/ef8/sae/layer_23.pt"
    model = get_latent_model(nt_model_path, layer_idx=23, sae_path=sae_checkpoint)
    
    # Calculate dependency map for a featured RNA G-quadruplex sequence
    print(f"\nCalculating dependency map (f/{feature_id}) for sequence (length {len(sequence)} bp)")
    print(f"Sequence: {sequence}")
    print(f"Feature ID: {feature_id}\n")
    
    dep_map = calculate_position_dependency_map(model, sequence, feature_id)
    print(f"Dependency map shape: {dep_map.shape}")
    print(f"Dependency map stats: min={dep_map.min():.4f}, max={dep_map.max():.4f}, mean={dep_map.mean():.4f}\n")
    
    # Compute Information diffusion representations
    random_walk = probatransition(dep_map)
    snp_laplace = laplacian(dep_map)

    # Compute Metrics on each representation
    representations = {
        "Raw": dep_map,
        "Random Walk Transition": random_walk,
        "Laplacian": snp_laplace
    }

    for name, matrix in representations.items():
        print(f"=== {name} Representation ===\n")
        diag = diagonal(matrix, normalize=True)
        row_proj = rowprojection(matrix, normalize=True)
        col_proj = columnprojection(matrix, normalize=True)
        eigvec = eigenvec(matrix)
        print(f"Diagonal: {diag}\n")
        print(f"Diagonal Stats: min={diag.min():.4f}, max={diag.max():.4f}, mean={diag.mean():.4f}\n")
        print(f"Row Projection: {row_proj}\n")
        print(f"Row Projection Stats: min={row_proj.min():.4f}, max={row_proj.max():.4f}, mean={row_proj.mean():.4f}\n")
        print(f"Column Projection: {col_proj}\n")
        print(f"Column Projection Stats: min={col_proj.min():.4f}, max={col_proj.max():.4f}, mean={col_proj.mean():.4f}\n")
        print(f"Eigenvector Centrality: {eigvec}\n")    
        print(f"Eigenvector Centrality Stats: min={eigvec.min():.4f}, max={eigvec.max():.4f}, mean={eigvec.mean():.4f}\n")