"""
Functions for extracting and writing mmCIF structure files from JSON data.
"""

import json
from pathlib import Path
from typing import Union, Optional


def extract_mmcif_from_json(
    json_file: Union[str, Path],
    structure_index: int = None
) -> list[str]:
    """
    Extract mmCIF string from a JSON file and optionally save it to a properly formatted mmCIF file.
    
    This function reads a JSON file containing structural prediction data (e.g., from Boltz-2),
    extracts the mmCIF formatted structure string, and optionally writes it to a file.
    
    Args:
        json_file: Path to the JSON file containing the structure data.
        structure_index: Index of the structure in the "structures" array (default: 0).
    
    Returns:
        A list of mmCIF formatted structure strings.
    
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        KeyError: If the expected JSON structure is not found.
        IndexError: If the structure_index is out of range.
        ValueError: If the JSON file is invalid or empty.
    
    Example:
        >>> # Extract and save to file
        >>> mmcif_str = extract_mmcif_from_json(
        ...     "data/rg4/boltz2predstruc/chrX:19836171-19836201(-).json",
        ...     "output.cif"
        ... )
        >>> 
        >>> # Extract without saving
        >>> mmcif_str = extract_mmcif_from_json("input.json")
    """
    json_path = Path(json_file)
    
    # Check if file exists
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    # Read and parse JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    
    # Validate JSON structure
    if not isinstance(data, dict):
        raise ValueError("JSON root must be a dictionary")
    
    if "structures" not in data:
        raise KeyError("JSON must contain 'structures' key")
    
    structures = data["structures"]
    if not isinstance(structures, list) or len(structures) == 0:
        raise ValueError("'structures' must be a non-empty list")
    
    # Extract structure at specified index
    strings = []
    if structure_index:
        try:
            struct = structures[structure_index]
            if "structure" not in struct:
                raise KeyError(f"Structure at index {structure_index} missing 'structure' key")
            strings.append(struct["structure"])
        except IndexError:
            raise IndexError(
                f"structure_index {structure_index} out of range "
                f"(available: 0-{len(structures)-1})"
            )
    # extract all structures if no index specified
    else:
        for idx, values in enumerate(structures):
            if "structure" not in values:
                raise KeyError(f"Structure at index {idx} missing 'structure' key")
            else:
                strings.append(values["structure"])
    
    return strings


def generate_mmcif_filename(json_file: Union[str, Path], suffix: str = ".cif", struct_idx: int = None) -> Path:
    """
    Generate an output mmCIF filename based on the input JSON filename.
    
    Args:
        json_file: Path to the JSON file.
        suffix: File extension for the output file (default: ".cif").
    
    Returns:
        Path object for the output mmCIF file.
    
    Example:
        >>> generate_mmcif_filename("chrX:19836171-19836201(-).json")
        PosixPath('chrX:19836171-19836201(-).cif')
    """
    json_path = Path(json_file)
    if struct_idx is not None:
        suffix = f".{struct_idx}{suffix}"
    return json_path.with_suffix(suffix)


def batch_extract_mmcif(
    json_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.json"
) -> list[Path]:
    """
    Extract mmCIF files from all JSON files in a directory.
    
    Args:
        json_dir: Directory containing JSON files.
        output_dir: Directory to save mmCIF files. If None, uses json_dir.
        pattern: Glob pattern for JSON files (default: "*.json").
    
    Returns:
        List of paths to the created mmCIF files.
    
    Example:
        >>> batch_extract_mmcif("data/rg4/boltz2predstruc", "data/rg4/cif_files")
    """
    json_path = Path(json_dir)
    output_path = Path(output_dir) if output_dir else json_path
    
    if not json_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {json_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    json_files = list(json_path.glob(pattern))
    
    if not json_files:
        print(f"No JSON files found matching pattern '{pattern}' in {json_dir}")
        return created_files
    
    for json_file in json_files:
        try:
            cifstrings = extract_mmcif_from_json(json_file)
            for idx, cif in enumerate(cifstrings):
                output_file = output_path / generate_mmcif_filename(json_file.name, struct_idx=idx)
                with open(output_file, 'w') as f:
                    f.write(cif)
                created_files.append(output_file)
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    print(f"\nProcessed {len(created_files)}/{len(json_files)} files successfully")
    return created_files






if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mmcif.py <json_file> [output_file]")
        print("       python mmcif.py --batch <json_dir> [output_dir]")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: --batch requires a directory path")
            sys.exit(1)
        json_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        batch_extract_mmcif(json_dir, output_dir)
    else:
        json_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        cifstrings = extract_mmcif_from_json(json_file)
        if output_file:
            for idx, cif in enumerate(cifstrings):
                out_file = generate_mmcif_filename(json_file, struct_idx=idx) if len(cifstrings) > 1 else output_file
                with open(out_file, 'w') as f:
                    f.write(cif)
                print(f"mmCIF structure saved to {out_file}")
        else:
            for cif in cifstrings:
                print(cif)
