def transcribe(seq: str, rt: bool = False) -> str:
    """
    Compute the reverse complement of a DNA sequence, returning an RNA sequence.
    
    If rt=True, reverses the process to compute the reverse complement of an RNA sequence, returning a DNA sequence.
    """
    if rt:
        complement = {
            "A":"T",
            "C":"G",
            "G":"C",
            "U":"A",
        }
    else:
        complement = {
            "A":"U",
            "C":"G",
            "G":"C",
            "T":"A",
        }
    return "".join([complement.get(base, base) for base in reversed(seq)])

def revcomp(seq: str) -> str:
    """
    Compute the reverse complement of a DNA sequence.
    """
    complement = {
        "A":"T",
        "C":"G",
        "G":"C",
        "T":"A",
    }
    return "".join([complement.get(base, base) for base in reversed(seq)])