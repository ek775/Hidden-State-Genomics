from dotenv import load_dotenv
from tqdm import tqdm
import os, re

import matplotlib.pyplot as plt
import torch
import numpy as np


load_dotenv()  # take environment variables from .env file


def extract_feature_activations_with_tokens(model, sequence, feature_id) -> tuple[np.ndarray, list[str]]:
    """
    Extract SAE feature activations for a specific feature from a single sequence with token information.
    
    Args:
        model: The LatentModel (ntsae23) with loaded SAE
        sequence: DNA sequence string to analyze
        feature_id: Feature index to extract
        
    Returns:
        tuple: (feature_activations, tokens)
            - feature_activations: numpy array of activations for the specified feature [num_tokens]
            - tokens: list of token strings for the sequence
    """
    model.eval()
    with torch.no_grad():
        sae_acts, tokens = model(sequence, return_tokens=True)
        # Strip special tokens if present (e.g., "<CLS>")
        tokens = [t for t in tokens if not re.match(r'^<.*>$', t)]
        
        # Extract activations for this specific feature across all tokens
        # sae_acts shape: [num_tokens, num_features]
        feat_acts = sae_acts[:, feature_id].cpu().numpy()
    
    return feat_acts, tokens


def site_directed_mutagenesis(sequence, position) -> list[str]:

    bases = ["A", "T", "C", "G"]

    if type(sequence) == str:
        sequence = sequence.upper()
    else:
        raise ValueError("Input sequence must be a string")
    
    mutated_sequences = []
    for mut_base in bases:
        if mut_base != sequence[position]:
            mutated_seq = sequence[:position] + mut_base + sequence[position+1:]
            mutated_sequences.append(mutated_seq)

    return mutated_sequences


def convert_basis(feature_activations, tokens) -> np.ndarray:
    """
    Convert token-level feature activations to base-level activations by averaging
    activations from overlapping k-mer tokens that cover each base.
    
    Args:
        feature_activations: numpy.ndarray of shape [num_tokens] - single feature activations
        tokens: List of token strings corresponding to the feature activations
        
    Returns:
        numpy.ndarray: Base-level feature activations corresponding to each base in the sequence
    """
    base_activations = []
    for t in tokens:
        base_activations.extend([feature_activations[tokens.index(t)]] * len(t))
    
    array = np.array(base_activations)
    return array


def snp_mutation_effect(model, sequence, feature_id, position) -> np.ndarray:
    """
    Calculate the effect of mutating a single base position to all 3 alternative bases
    on feature activation at all base positions.
    
    Args:
        model: The LatentModel (ntsae23) with loaded SAE
        sequence: reference DNA sequence string
        feature_id: The feature index to analyze
        position: Base position to mutate
        
    Returns:
        numpy.ndarray: 1D array [affected_base_position] with averaged feature activation changes
                       across 3 mutations at the specified position
    """
    # Get baseline activations
    baseline_feat, baseline_tokens = extract_feature_activations_with_tokens(model, sequence, feature_id)
    baseline_base_acts = convert_basis(baseline_feat, baseline_tokens)
    
    # Get mutated sequences
    mut_seqs = site_directed_mutagenesis(sequence, position)
    
    # Calculate effects for each mutation
    effects = []
    for mut_seq in mut_seqs:
        mut_feat, mut_tokens = extract_feature_activations_with_tokens(model, mut_seq, feature_id)
        mut_base_acts = convert_basis(mut_feat, mut_tokens)
        
        # Calculate absolute difference
        effect = np.abs(mut_base_acts - baseline_base_acts)
        effects.append(effect)
    
    # Average across the 3 mutations
    mean_effect = np.mean(effects, axis=0)
    return mean_effect


def calculate_position_dependency_map(model, sequence, feature_id) -> np.ndarray:
    """
    Calculate position-wise feature activity dependency for a sequence at base resolution.
    
    For each base position, mutate it to all 3 alternative bases and measure 
    the effect on feature activation at all base positions. Handles tokenization
    by attributing token activations to each base within the token.
    
    Args:
        model: The LatentModel (ntsae23) with loaded SAE
        sequence: DNA sequence string
        feature_id: The feature index to analyze
        
    Returns:
        numpy.ndarray: 2D array [mutated_base_position, affected_base_position] with averaged
                       feature activation changes across 3 mutations per position
    """
    sequence = sequence.upper()
    seq_len = len(sequence)
    dependency_matrix = np.zeros((seq_len, seq_len))

    print(f"Computing dependency map for {seq_len} positions...")
    for pos in tqdm(range(seq_len), desc="Mutating positions"):
        dependency_matrix[pos, :] = snp_mutation_effect(model, sequence, feature_id, pos)
    
    return dependency_matrix


def plot_dependency_heatmap(dependency_matrix: np.ndarray, sequence: str, feature_id: int, 
                           title=None, figsize=(14, 10), cmap='viridis') -> tuple:
    """
    Plot position-wise feature activity dependency heat map at base resolution.
    
    Args:
        dependency_matrix: 2D numpy array [mutated_base_position, affected_base_position]
        sequence: Original DNA sequence string
        feature_id: Feature ID for labeling
        title: Custom title for the plot
        figsize: Figure size tuple
        cmap: Colormap name
        
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sequence = sequence.upper()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(dependency_matrix, aspect='auto', cmap=cmap, 
                   interpolation='nearest', origin='lower')
    
    # Set labels - Y axis has numeric positions, X axis has reference bases
    ax.set_xlabel('Affected Position (index:reference base)', fontsize=12)
    ax.set_ylabel('Mutated Position (index:reference base)', fontsize=12)
    
    if title is None:
        title = f'Position-wise Feature {feature_id} Activity Dependency\n(Average of Possible mutations per base position)'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Δ Feature Activation|', rotation=270, labelpad=20, fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set tick labels - show all positions
    seq_len = len(sequence)
    tick_positions = list(range(seq_len))
    
    # X-axis: show reference bases at affected positions
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{i}: ({sequence[i]})" for i in tick_positions], fontsize=6, 
                       rotation=90, ha='center')
    
    # Y-axis: show numeric positions that are mutated
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{i}: ({sequence[i]})" for i in tick_positions], fontsize=6)
    
    plt.tight_layout()
    return fig, ax


def analyze_sequence_dependency(model, sequence, feature_id, plot=True, save_path=None, title=None) -> dict:
    """
    Complete workflow: calculate and optionally plot position-wise dependency at base resolution.
    
    Args:
        model: The LatentModel (ntsae23) with loaded SAE
        sequence: DNA sequence string
        feature_id: The feature index to analyze
        plot: Whether to create visualization
        save_path: Optional path to save the plot
        title: Optional title for the plot
    Returns:
        dict: Contains dependency_matrix, and optionally fig/ax if plot=True
    """
    print(f"Calculating base-level position dependency map for feature {feature_id}...")
    print(f"Sequence length: {len(sequence)} bp")
    
    # Calculate dependency matrix at base resolution
    dep_matrix = calculate_position_dependency_map(model, sequence, feature_id)
    
    results = {'dependency_matrix': dep_matrix, 'sequence': sequence}
    
    if plot:
        print("Generating heat map...")
        fig, ax = plot_dependency_heatmap(dep_matrix, sequence, feature_id, title=title)
        results['fig'] = fig
        results['ax'] = ax
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
    
    print(f"Max dependency delta: {dep_matrix.max():.4f}")
    print(f"Mean dependency delta: {dep_matrix.mean():.4f}")
    
    return results


def generate_feature_dependency_heatmaps(model, sequence, feature_ids, output_dir='data/dependency_maps', show_plots=True,
                                         seq_label=None):
    """
    Generate position-wise dependency heatmaps for multiple features.
    
    Args:
        model: The LatentModel (ntsae23) with loaded SAE
        sequence: DNA sequence string to analyze
        feature_ids: Array/list of feature IDs to generate heatmaps for
        output_dir: Directory to save the heatmap images
        show_plots: Whether to display plots (default: True)
        
    Returns:
        dict: Dictionary mapping feature_id to results dict
    """
    all_results = {}
    
    for feature_id in feature_ids:
        print(f"\n{'='*60}")
        print(f"Processing Feature {feature_id}")
        print(f"{'='*60}")
        
        # Generate appropriate title
        if seq_label:
            title = f'Feature {feature_id} Position-wise Dependency Map for {seq_label}'
        else:
            title = None
        
        # Generate save path
        os.makedirs(output_dir, exist_ok=True)
        save_path = f'{output_dir}/feature_{feature_id}_dependency_map.png'
        
        # Analyze and plot
        results = analyze_sequence_dependency(
            model=model,
            sequence=sequence,
            feature_id=feature_id,
            plot=True,
            save_path=save_path,
            title=title
        )
        
        all_results[feature_id] = results
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    print(f"\n{'='*60}")
    print(f"Completed {len(feature_ids)} feature dependency maps")
    print(f"{'='*60}")
    
    return all_results
