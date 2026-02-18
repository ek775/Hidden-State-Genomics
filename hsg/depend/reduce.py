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
    if normalize:
        vector = minmax_normalize(vector)
    return np.dot(vector, matrix)


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
    
    # Compute all vector reductions
    print("="*70)
    print("COMPUTING ALL REDUCTION FUNCTIONS")
    print("="*70)
    
    diag = diagonal(dep_map, normalize=True)
    row_proj = rowprojection(dep_map, normalize=True)
    col_proj = columnprojection(dep_map, normalize=True)
    eigen_cent = eigenvec(dep_map) # already normalized to unit norm
    
    print("\nVector reductions computed successfully.")
    print(f"Expected shape: {diag.shape}\n")
    
    # Position-by-position comparison
    print("="*100)
    print("POSITION-BY-POSITION COMPARISON")
    print("="*100)
    print(f"{'Pos':<4} {'Base':<5} {'Diagonal':<12} {'Row Proj':<12} {'Col Proj':<12} {'Eigenvec':<12}")
    print(f"{'':>4} {'':>5} {'(self-eff)':<12} {'(out-deg)':<12} {'(in-deg)':<12} {'(central)':<12}")
    print("-"*100)
    
    for i in range(len(sequence)):
        print(f"{i:<4} {sequence[i]:<5} {diag[i]:<12.6f} {row_proj[i]:<12.6f} {col_proj[i]:<12.6f} {eigen_cent[i]:<12.6f}")
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    metrics = {
        'Diagonal (self-effect)': diag,
        'Row Projection (out-degree)': row_proj,
        'Column Projection (in-degree)': col_proj,
        'Eigenvector Centrality': eigen_cent
    }
    
    print(f"\n{'Metric':<35} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print("-"*100)
    for name, vec in metrics.items():
        print(f"{name:<35} {vec.min():<12.6f} {vec.max():<12.6f} {vec.mean():<12.6f} {vec.std():<12.6f}")
    
    # Top positions for each metric
    print("\n" + "="*100)
    print("TOP 5 POSITIONS BY EACH METRIC")
    print("="*100)
    
    for name, vec in metrics.items():
        top_indices = np.argsort(vec)[-5:][::-1]
        print(f"\n{name}:")
        print(f"  Positions: {top_indices}")
        print(f"  Bases:     {[sequence[i] for i in top_indices]}")
        print(f"  Values:    {[f'{vec[i]:.6f}' for i in top_indices]}")
    
    # Matrix reductions (just show info, not full matrices)
    print("\n" + "="*100)
    print("MATRIX REDUCTIONS (returns nxn matrices)")
    print("="*100)
    
    trans_prob = probatransition(dep_map)
    lap = laplacian(dep_map)
    
    print(f"\nTransition Probability Matrix (P = D^-1 * A):")
    print(f"  Shape: {trans_prob.shape}")
    print(trans_prob)
    print(f"\nGraph Laplacian (L = D - A):")
    print(f"  Shape: {lap.shape}")
    print(lap)
    
    print("\n" + "="*100)
