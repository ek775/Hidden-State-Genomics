from hsg.depend.reduce import spectral_embedding, minmax_normalize
from hsg.sequence import transcribe
from hsg.depend.featurewise import calculate_position_dependency_map
from hsg.stattools.features import get_latent_model
import os, pathlib
import numpy as np
import Bio
from Bio.PDB import MMCIFParser, MMCIFIO, Structure
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


targetfeatures = [7030, 8161, 1422, 3378] # 3 steered/validated, 1 positive for curiousity


def parse_mmcif(cif_path: str, structure_id: Optional[str] = None) -> Structure.Structure:
    """
    Parse an mmCIF file into a BioPython Structure object.
    
    Args:
        cif_path: Path to the mmCIF file
        structure_id: Optional identifier for the structure. If None, uses the filename stem.
        
    Returns:
        Bio.PDB.Structure.Structure object containing the parsed structure
        
    Example:
        >>> structure = parse_mmcif("data/rg4/boltz2structs/chrX:19836171-19836201(-)_cisplatin_1.cif")
        >>> print(structure.id)
        chrX:19836171-19836201(-)_cisplatin_1
        >>> # Access chains
        >>> for chain in structure.get_chains():
        ...     print(f"Chain {chain.id}: {len(list(chain.get_residues()))} residues")
        >>> # Access atoms
        >>> for atom in structure.get_atoms():
        ...     print(f"{atom.name}: bfactor={atom.bfactor}")
    """
    cif_path = Path(cif_path)
    
    if not cif_path.exists():
        raise FileNotFoundError(f"mmCIF file not found: {cif_path}")
    
    # Use filename stem as structure ID if not provided
    if structure_id is None:
        structure_id = cif_path.stem
    
    # Initialize parser
    parser = MMCIFParser(QUIET=True)  # QUIET=True suppresses warnings
    
    # Parse the structure
    try:
        structure = parser.get_structure(structure_id, str(cif_path))
    except Exception as e:
        raise ValueError(f"Failed to parse mmCIF file {cif_path}: {e}")
    
    return structure


def get_structure_info(structure: Structure.Structure) -> dict:
    """
    Extract basic information from a parsed Structure object.
    
    Args:
        structure: Bio.PDB.Structure.Structure object
        
    Returns:
        Dictionary with structure information including number of models, chains, residues, atoms
        
    Example:
        >>> structure = parse_mmcif("example.cif")
        >>> info = get_structure_info(structure)
        >>> print(f"Structure has {info['n_residues']} residues")
    """
    info = {
        'structure_id': structure.id,
        'n_models': len(list(structure.get_models())),
        'n_chains': len(list(structure.get_chains())),
        'n_residues': len(list(structure.get_residues())),
        'n_atoms': len(list(structure.get_atoms())),
        'chains': []
    }
    
    # Get info for each chain
    for chain in structure.get_chains():
        chain_info = {
            'chain_id': chain.id,
            'n_residues': len(list(chain.get_residues())),
            'residue_ids': [res.id[1] for res in chain.get_residues()]
        }
        info['chains'].append(chain_info)
    
    return info


def update_bfactors(
    structure: Structure.Structure, 
    bfactor_values: np.ndarray,
    chain_id: Optional[str] = None,
    polymer_only: bool = True
) -> Structure.Structure:
    """
    Update b-factor values for all atoms in a structure based on per-residue values.
    
    Args:
        structure: Bio.PDB.Structure.Structure object to modify
        bfactor_values: 1D numpy array of b-factor values, one per residue
        chain_id: Specific chain ID to update. If None, uses first polymer chain.
        polymer_only: If True, only update polymer chains (skip ligands/hetero atoms)
        
    Returns:
        Modified structure with updated b-factors
        
    Raises:
        ValueError: If vector length doesn't match number of residues in chain
        
    Example:
        >>> structure = parse_mmcif("example.cif")
        >>> # Create dependency projection (30 values for 30 residues)
        >>> projection = np.random.rand(30) * 100
        >>> structure = update_bfactors(structure, projection)
        >>> # Save modified structure
        >>> io = MMCIFIO()
        >>> io.set_structure(structure)
        >>> io.save("output.cif")
    """
    # Find the target chain
    target_chain = None
    
    for model in structure:
        for chain in model:
            # If specific chain requested, match it
            if chain_id is not None:
                if chain.id == chain_id:
                    target_chain = chain
                    break
            # Otherwise, find first polymer chain (skip ligands)
            elif polymer_only:
                # Check if this is a polymer chain by looking at residue types
                # Polymer residues have standard names like A, C, G, U (RNA) or standard amino acids
                residues = list(chain.get_residues())
                if residues and residues[0].id[0] == ' ':  # ' ' means standard residue (not hetero)
                    target_chain = chain
                    break
            else:
                target_chain = chain
                break
        if target_chain:
            break
    
    if target_chain is None:
        raise ValueError(f"Could not find target chain (chain_id={chain_id}, polymer_only={polymer_only})")
    
    # Get residues from the target chain
    residues = list(target_chain.get_residues())
    
    # Filter to only standard residues if polymer_only
    if polymer_only:
        residues = [r for r in residues if r.id[0] == ' ']
    
    # Verify vector length matches
    if len(bfactor_values) != len(residues):
        raise ValueError(
            f"Vector length ({len(bfactor_values)}) does not match number of residues "
            f"({len(residues)}) in chain {target_chain.id}"
        )
    
    # Update b-factors for all atoms in each residue
    for residue_idx, residue in enumerate(residues):
        new_bfactor = float(bfactor_values[residue_idx])
        for atom in residue.get_atoms():
            atom.bfactor = new_bfactor
    
    return structure


def save_mmcif(structure: Structure.Structure, output_path: str):
    """
    Save a Structure object to an mmCIF file.
    
    Args:
        structure: Bio.PDB.Structure.Structure object to save
        output_path: Path for the output mmCIF file
        
    Example:
        >>> structure = parse_mmcif("input.cif")
        >>> # ... modify structure ...
        >>> save_mmcif(structure, "output.cif")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(output_path))


def get_rna_sequence_from_structure(structure: Structure.Structure, chain_id: Optional[str] = None) -> str:
    """
    Extract RNA sequence from a structure.
    
    Args:
        structure: Bio.PDB.Structure.Structure object
        chain_id: Specific chain ID to extract. If None, uses first polymer chain.
        
    Returns:
        RNA sequence as a string (e.g., "AUCGAUCG")
    """
    # Find target chain
    target_chain = None
    for model in structure:
        for chain in model:
            if chain_id is not None:
                if chain.id == chain_id:
                    target_chain = chain
                    break
            else:
                # Find first polymer chain
                residues = list(chain.get_residues())
                if residues and residues[0].id[0] == ' ':
                    target_chain = chain
                    break
        if target_chain:
            break
    
    if target_chain is None:
        raise ValueError(f"Could not find target chain (chain_id={chain_id})")
    
    # Extract sequence from residues
    residues = [r for r in target_chain.get_residues() if r.id[0] == ' ']
    sequence = ''.join([r.resname for r in residues])
    
    return sequence


def reverse_transcribe(rna_seq: str) -> str:
    """
    Convert RNA sequence back to DNA using reverse transcription.
    
    Args:
        rna_seq: RNA sequence string
        
    Returns:
        DNA sequence string (reverse complement with U->T)
        
    Example:
        >>> reverse_transcribe("AUCGAUCG")
        'CGATCGAT'
    """
    # Use the transcribe function with rt=True for proper reverse transcription
    return transcribe(rna_seq, rt=True)


def compute_dependency_projections(
    model,
    dna_sequence: str,
    feature_id: int
) -> dict[str, np.ndarray]:
    """
    Compute dependency map and extract multiple projection types.
    
    Args:
        model: LatentModel with loaded SAE
        dna_sequence: DNA sequence string
        feature_id: SAE feature ID to analyze
        
    Returns:
        Dictionary with keys:
            - 'diagonal': Diagonal values (self-dependency)
            - 'spectral_1comp': Spectral embedding with 1 component
            - 'spectral_3comp': Spectral embedding with 3 components
    """
    # Calculate dependency map
    dep_map = calculate_position_dependency_map(model, dna_sequence, feature_id)
    
    # Extract diagonal (self-dependency)
    diagonal = np.diag(dep_map)
    
    # Spectral embedding with 1 component
    spectral_1 = spectral_embedding(dep_map, n_components=1, normalize=True)
    
    # Spectral embedding with 3 components
    spectral_3 = spectral_embedding(dep_map, n_components=3, normalize=True)
    
    # Normalize all to 0-100 for b-factor range
    projections = {
        'diagonal': minmax_normalize(diagonal) * 100,
        'spectral_1comp': spectral_1 * 100,
        'spectral_3comp': spectral_3 * 100
    }
    
    return projections


def process_mmcif_file(
    cif_path: Path,
    model,
    feature_id: int,
    output_base_dir: Path,
    verbose: bool = True
) -> dict[str, Path]:
    """
    Process a single mmCIF file: extract sequence, compute projections, and save variants.
    
    Args:
        cif_path: Path to input mmCIF file
        model: LatentModel for dependency calculations
        feature_id: SAE feature ID
        output_base_dir: Base output directory (will create subdirs)
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping projection types to output file paths
        
    Directory structure created:
        output_base_dir/
            feature_{feature_id}/
                diagonal/
                    {filename}.cif
                spectral_1comp/
                    {filename}.cif
                spectral_3comp/
                    {filename}.cif
    """
    if verbose:
        print(f"Processing {cif_path.name}...")
    
    # Parse structure
    structure = parse_mmcif(str(cif_path))
    
    # Extract RNA sequence
    rna_seq = get_rna_sequence_from_structure(structure)
    
    # Reverse transcribe to DNA
    dna_seq = reverse_transcribe(rna_seq)
    
    if verbose:
        print(f"  RNA sequence: {rna_seq[:30]}{'...' if len(rna_seq) > 30 else ''}")
        print(f"  DNA sequence: {dna_seq[:30]}{'...' if len(dna_seq) > 30 else ''}")
        print(f"  Computing dependency projections...")
    
    # Compute all projections
    projections = compute_dependency_projections(model, dna_seq, feature_id)
    
    # Create output directories and save each projection type
    output_paths = {}
    feature_dir = output_base_dir / f"feature_{feature_id}"
    
    for proj_type, proj_values in projections.items():
        # Create subdirectory for this projection type
        proj_dir = feature_dir / proj_type
        proj_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse a fresh copy of the structure for each projection
        proj_structure = parse_mmcif(str(cif_path))
        
        # Reverse the projection values to match original RNA structure orientation
        # (since reverse_transcribe reverses the sequence for DNA computation)
        proj_values_reversed = proj_values[::-1]
        
        # Update b-factors
        proj_structure = update_bfactors(proj_structure, proj_values_reversed)
        
        # Save to output directory
        output_path = proj_dir / cif_path.name
        save_mmcif(proj_structure, str(output_path))
        output_paths[proj_type] = output_path
        
        if verbose:
            print(f"  Saved {proj_type}: {output_path}")
    
    return output_paths


def batch_process_directory(
    input_dir: str,
    output_dir: str,
    feature_ids: list[int],
    model_path: Optional[str] = None,
    sae_path: str = "checkpoints/hidden-state-genomics/ef8/sae/layer_23.pt",
    layer_idx: int = 23,
    pattern: str = "*.cif"
):
    """
    Process all mmCIF files in a directory for multiple features.
    
    Args:
        input_dir: Directory containing input mmCIF files
        output_dir: Base output directory
        feature_ids: List of SAE feature IDs to process
        model_path: Path to nucleotide transformer (default: from NT_MODEL env)
        sae_path: Path to SAE checkpoint
        layer_idx: Transformer layer index
        pattern: Glob pattern for finding mmCIF files
        
    Example:
        >>> batch_process_directory(
        ...     "data/rg4/boltz2structs",
        ...     "data/rg4/dependency_bfactors",
        ...     feature_ids=[3378, 7030, 8161, 1422]
        ... )
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("=" * 80)
    print("mmCIF Dependency Map B-Factor Injection Pipeline")
    print("=" * 80)
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Feature IDs: {feature_ids}")
    print(f"Pattern: {pattern}")
    print("=" * 80)
    
    # Load model
    print("\nLoading nucleotide transformer and SAE...")
    if model_path is None:
        model_path = os.environ.get("NT_MODEL")
        if model_path is None:
            raise ValueError("NT_MODEL not set in environment and no model_path provided")
    
    model = get_latent_model(
        parent_model_path=model_path,
        layer_idx=layer_idx,
        sae_path=sae_path
    )
    print("✓ Model loaded")
    
    # Find all mmCIF files
    cif_files = list(input_path.glob(pattern))
    print(f"\n✓ Found {len(cif_files)} mmCIF files")
    
    if len(cif_files) == 0:
        print(f"Warning: No files matching pattern '{pattern}' found in {input_path}")
        return
    
    # Process each feature
    for feature_id in feature_ids:
        print(f"\n{'=' * 80}")
        print(f"Processing Feature {feature_id}")
        print(f"{'=' * 80}")
        
        successful = 0
        failed = 0
        
        for cif_file in cif_files:
            try:
                process_mmcif_file(
                    cif_file,
                    model,
                    feature_id,
                    output_path,
                    verbose=True
                )
                successful += 1
            except Exception as e:
                print(f"✗ Error processing {cif_file.name}: {e}")
                failed += 1
        
        print(f"\nFeature {feature_id} complete: {successful} successful, {failed} failed")
    
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}")
    print(f"Output saved to: {output_path}")
    print("Directory structure:")
    print("  feature_<ID>/")
    print("    diagonal/")
    print("    spectral_1comp/")
    print("    spectral_3comp/")
    print("=" * 80)


if __name__ == "__main__":
    from tap import tapify
    
    def main(
        input_dir: str = "data/rg4/boltz2structs",
        output_dir: str = "data/rg4/dependency_bfactors",
        feature_ids: list[int] = None,
        layer_idx: int = 23,
        sae_path: str = "checkpoints/hidden-state-genomics/ef8/sae/layer_23.pt"
    ):
        """
        Process mmCIF files to inject dependency map projections as b-factors.
        
        Args:
            input_dir: Directory containing input mmCIF files
            output_dir: Output directory for processed files
            feature_ids: List of feature IDs to process (default: [7030, 8161, 1422, 3378])
            layer_idx: Transformer layer index (default: 23)
            sae_path: Path to SAE checkpoint
        """
        if feature_ids is None:
            feature_ids = targetfeatures
        
        batch_process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            feature_ids=feature_ids,
            layer_idx=layer_idx,
            sae_path=sae_path
        )
    
    tapify(main)


