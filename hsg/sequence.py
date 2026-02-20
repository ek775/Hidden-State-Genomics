def transcribe(seq: str) -> str:
    """
    Compute the reverse complement of a DNA sequence, returning an RNA sequence.
    """
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