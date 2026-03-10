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

def laplacian(matrix: np.ndarray) -> np.ndarray:
    """Return the Laplacian of a graph represented by an adjacency matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    degree = np.sum(matrix, axis=1)
    laplacian = np.diag(degree) - matrix
    return laplacian

def normalized_laplacian(matrix: np.ndarray) -> np.ndarray:
    """Return the normalized Laplacian of a graph represented by an adjacency matrix."""
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    degree = np.sum(matrix, axis=1)
    degree_sqrt_inv = np.zeros_like(degree)
    nonzero_mask = degree > 0
    degree_sqrt_inv[nonzero_mask] = 1.0 / np.sqrt(degree[nonzero_mask])
    D_inv_sqrt = np.diag(degree_sqrt_inv)
    norm_lap = np.eye(matrix.shape[0]) - D_inv_sqrt @ matrix @ D_inv_sqrt
    return norm_lap

##############################################################################
# Matrix reduction functions
##############################################################################

def principal_eigenvector(matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute the principal eigenvector for positionwise importance scores.
    
    Finds the eigenvector corresponding to the largest eigenvalue (principal 
    eigenvector), which represents the dominant pattern of connectivity/dependency.
    
    Args:
        matrix: Square nxn adjacency or dependency matrix
        normalize: Whether to normalize scores to [0, 1]
        
    Returns:
        1D array of length n with positionwise scores from principal eigenvector
    """
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Find index of largest eigenvalue (by absolute value)
    max_idx = np.argmax(np.abs(eigenvalues))
    
    # Get corresponding eigenvector and take absolute values
    principal_vec = np.abs(eigenvectors[:, max_idx].real)
    
    if normalize:
        return minmax_normalize(principal_vec)
    return principal_vec


def fiedler_vector(matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute the Fiedler vector for regional sequence assignment.
    
    The Fiedler vector is the eigenvector corresponding to the second smallest
    eigenvalue of the Laplacian matrix. It's useful for graph partitioning and
    identifying communities/regions in the dependency structure.
    
    Args:
        matrix: Square nxn adjacency or dependency matrix
        normalize: Whether to normalize scores to [0, 1]
        
    Returns:
        1D array of length n with regional assignment scores (Fiedler vector)
    """
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    
    # Compute Laplacian
    lap = laplacian(matrix)
    
    # Compute eigenvalues and eigenvectors (eigh for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(lap)
    
    # Sort by eigenvalue magnitude
    sorted_indices = np.argsort(eigenvalues)
    
    # Second smallest eigenvalue's eigenvector is the Fiedler vector
    fiedler_vec = eigenvectors[:, sorted_indices[1]].real
    
    if normalize:
        return minmax_normalize(fiedler_vec)
    return fiedler_vec


def spectral_embedding(matrix: np.ndarray, n_components: int = 1, normalize: bool = True) -> np.ndarray:
    """Compute spectral embedding for hybrid positionwise scores.
    
    Uses eigenvectors of the normalized Laplacian to create a low-dimensional
    embedding of the graph nodes, capturing both local and global structure.
    
    Args:
        matrix: Square nxn adjacency or dependency matrix
        n_components: Number of eigenvector components to use (default: 1 for 1D output)
        normalize: Whether to normalize scores to [0, 1]
        
    Returns:
        1D array of length n with hybrid spectral embedding scores
    """
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    
    # Compute normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
    normalized_lap = normalized_laplacian(matrix)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_lap)
    
    # Sort by eigenvalue (ascending)
    sorted_indices = np.argsort(eigenvalues)
    
    # Use the first n_components eigenvectors (excluding the trivial one at 0)
    # Start from index 1 to skip the first eigenvector
    embedding_vecs = eigenvectors[:, sorted_indices[1:1+n_components]]
    
    # If n_components == 1, return as 1D array
    if n_components == 1:
        embedding_scores = embedding_vecs.flatten()
    else:
        # Combine multiple components (take L2 norm)
        embedding_scores = np.linalg.norm(embedding_vecs, axis=1)
    
    if normalize:
        return minmax_normalize(embedding_scores)
    return embedding_scores




###################################################################################
# Test
###################################################################################

if __name__ == "__main__":
    from hsg.depend.featurewise import calculate_position_dependency_map
    from hsg.stattools.features import get_latent_model
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os, argparse
    from hsg.sequence import revcomp

    parser = argparse.ArgumentParser(description="Test spectral reduction methods on a dependency map")
    parser.add_argument("-s", "--sequence", type=str, default="CAGGAAGGAGGATGGAGGCTGGTGGGAGGG", help="RNA sequence to analyze")
    parser.add_argument("-f", "--feature", type=int, default=3378, help="Feature ID to calculate dependency map for")
    parser.add_argument("--normalize", action="store_true", default=False, help="Whether to normalize output scores to [0, 1]")
    args = parser.parse_args()

    sequence = args.sequence.upper().strip()
    
    # Load model
    print("Loading nucleotide transformer and SAE...")
    nt_model_path = os.getenv("NT_MODEL")
    sae_checkpoint = "checkpoints/hidden-state-genomics/ef8/sae/layer_23.pt"
    model = get_latent_model(nt_model_path, layer_idx=23, sae_path=sae_checkpoint)
    
    # Calculate dependency map for a featured RNA G-quadruplex sequence
    print(f"\nCalculating dependency map (f/{args.feature}) for sequence (length {len(sequence)} bp)")
    print(f"Sequence: {sequence}")
    print(f"Feature ID: {args.feature}\n")
    
    dep_map = calculate_position_dependency_map(model, sequence, args.feature)
    print(f"Dependency map shape: {dep_map.shape}")
    print(f"Dependency map stats: min={dep_map.min():.4f}, max={dep_map.max():.4f}, mean={dep_map.mean():.4f}\n")
    
    # Test spectral reduction methods
    print("=" * 80)
    print("SPECTRAL REDUCTION METHODS (nxn → 1xn)")
    print("=" * 80 + "\n")
    
    # Compute all reduction methods
    reduction_methods = {
        "Diagonal (self-dependency) [CONTROL]": (
            minmax_normalize(np.diag(dep_map)),
            "Extract diagonal values representing self-dependency at each position"
        ),
        "Principal Eigenvector": (
            principal_eigenvector(dep_map, normalize=args.normalize),
            "Eigenvector of largest eigenvalue (dominant connectivity pattern)"
        ),
        "Fiedler Vector": (
            fiedler_vector(dep_map, normalize=args.normalize),
            "2nd eigenvector of Laplacian (for graph partitioning/regions)"
        ),
        "Spectral Embedding (1-comp)": (
            spectral_embedding(dep_map, n_components=1, normalize=args.normalize),
            "Normalized Laplacian embedding (hybrid local+global structure)"
        ),
        "Spectral Embedding (3-comp)": (
            spectral_embedding(dep_map, n_components=3, normalize=args.normalize),
            "Multi-dimensional embedding combined via L2 norm"
        )
    }
    
    # Display results for each method
    for method_name, (result_vec, description) in reduction_methods.items():
        print(f"=== {method_name} ===")
        print(f"Description: {description}")
        print(f"Stats: min={result_vec.min():.4f}, max={result_vec.max():.4f}, mean={result_vec.mean():.4f}\n")
    
    # Create visualization
    print("=" * 80)
    print("GENERATING LINE CHART")
    print("=" * 80 + "\n")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Nucleotide positions (1-indexed for biological convention)
    positions = np.arange(1, len(sequence) + 1)
    
    # Plot each reduction method
    for method_name, (result_vec, _) in reduction_methods.items():
        ax.plot(positions, result_vec, marker='o', markersize=4, linewidth=2, 
                label=method_name, alpha=0.8)
    
    # Add nucleotide labels on x-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(list(sequence), fontfamily='monospace', fontsize=9)
    
    # Labels and styling
    ax.set_xlabel('Nucleotide Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Spectral Reduction Methods Comparison\nFeature {args.feature} | Sequence Length: {len(sequence)} bp', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add sequence as subtitle
    ax.text(0.5, -0.15, f'Sequence: {sequence}', 
            transform=ax.transAxes, ha='center', fontfamily='monospace', 
            fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)