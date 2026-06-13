"""
BWT-based next-token probability estimation via an FM-index.

The heavy lifting is done by the ``bwt`` Rust extension (bwt/src/lib.rs),
built with maturin::

    cd bwt && maturin develop --release

Public API
----------
BwtIndex
    Re-exported Rust type.  Build with ``BwtIndex(body)`` for a single string
    or ``BwtIndex.from_sequences(seqs)`` for multiple sequences (e.g. one
    entry per chromosome loaded from SeqRepo).

build_genome_index(chroms, ...)
    Load one or more GRCh38 chromosomes from the local SeqRepo database and
    return a ``BwtIndex`` over their concatenation.  Chromosomes are separated
    by the 0x01 inter-sequence byte so that k-mer queries never span a
    chromosome boundary.

build_corpus_index(text)
    Convenience wrapper for non-genomic corpora (English text, etc.).

next_nucleotide_probs(index, k)
    Return {A: p, C: p, G: p, T: p} from the given index for context ``k``.

score_kmer_context(body, k, k_plus_1)
    Single-pair convenience: (count_k, count_k1, probability).

English-language equivalent
---------------------------
For English strings the same ``BwtIndex`` API works unchanged — it operates on
raw bytes.  The statistically comparable NLP algorithm is the **Kneser-Ney
smoothed n-gram model**:

* Same quantity: P(w_n | w_{n-k+1} … w_{n-1}), estimated from corpus counts.
* Adds lower-order back-off + continuation-probability correction, giving
  non-zero mass to unseen n-grams (the BWT approach returns 0 for them).
* Implementations: ``nltk.lm.KneserNeyInterpolated`` (small corpora) or
  ``kenlm`` (compiled C++, fast, handles billions of tokens).
* Character n-gram order in KenLM is directly analogous to k-mer length here.
"""

from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

try:
    from bwt import BwtIndex  # noqa: F401  re-export
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "bwt Rust extension not found.  Build it with:\n"
        "    cd bwt && maturin develop --release"
    ) from exc

DNA_ALPHABET: tuple[str, ...] = ("A", "C", "G", "T")
RNA_ALPHABET: tuple[str, ...] = ("A", "C", "G", "U")


# ---------------------------------------------------------------------------
# Genome loading helpers
# ---------------------------------------------------------------------------

def _load_chromosome(
    sr,
    chrom: str,
    strand: str = "+",
) -> str:
    """Fetch one chromosome from an open SeqRepo instance."""
    seq = str(sr[f"GRCh38:{chrom}"][:]).upper()
    if strand == "-":
        from hsg.sequence import revcomp
        seq = revcomp(seq)
    return seq


def build_genome_index(
    chroms: list[str],
    seqrepo_path: str | None = None,
    strand: str = "+",
    sr=None,
) -> BwtIndex:
    """Build a ``BwtIndex`` over one or more GRCh38 chromosomes.

    Each chromosome is fetched from the local SeqRepo database and passed as a
    separate sequence to ``BwtIndex.from_sequences``, so k-mer queries do not
    span chromosome boundaries.

    Parameters
    ----------
    chroms:
        UCSC-style chromosome names, e.g. ``["chr1", "chr2"]``.
    seqrepo_path:
        Path to the SeqRepo root directory.  Defaults to the ``SEQREPO_PATH``
        environment variable.  Ignored when ``sr`` is provided.
    strand:
        ``"+"`` (default) or ``"-"`` — applied to all chromosomes.
    sr:
        An already-open ``biocommons.seqrepo.SeqRepo`` instance.  When
        supplied, ``seqrepo_path`` is ignored and no second database
        connection is opened.

    Returns
    -------
    BwtIndex
    """
    from biocommons.seqrepo import SeqRepo

    if sr is None:
        path = seqrepo_path or os.environ.get("SEQREPO_PATH")
        if not path:
            raise RuntimeError(
                "A SeqRepo path is required: pass seqrepo_path=, set the "
                "SEQREPO_PATH environment variable, or pass an open sr= instance."
            )
        sr = SeqRepo(path)
    sequences = [_load_chromosome(sr, c, strand) for c in chroms]
    return BwtIndex.from_sequences(sequences)


def build_corpus_index(corpus: str) -> BwtIndex:
    """Build a ``BwtIndex`` over an arbitrary string (English text, etc.).

    Parameters
    ----------
    corpus:
        Any UTF-8 string.  For file-backed corpora, read the file first and
        pass the contents here.

    Returns
    -------
    BwtIndex
    """
    return BwtIndex(corpus)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def next_nucleotide_probs(
    index: BwtIndex,
    k: str,
    alphabet: Sequence[str] = DNA_ALPHABET,
) -> dict[str, float]:
    """Empirical next-character probabilities for context ``k``.

    Returns ``{k+a: P(k+a | k)}`` for each ``a`` in ``alphabet``.

    Parameters
    ----------
    index:
        A ``BwtIndex`` built over the corpus of interest.
    k:
        The base context string (k-mer).
    alphabet:
        Candidate next characters.  Defaults to :data:`DNA_ALPHABET`.
        Substitute any token vocabulary here for non-genomic use cases.

    Returns
    -------
    dict[str, float]
        Keys are the full extension strings (``k + a``).  Values sum to ≤ 1
        (< 1 when ``k`` is sometimes followed by characters outside the
        supplied alphabet, or occurs at the end of sequences).
    """
    extensions = [k + a for a in alphabet]
    return index.next_token_probs(k, extensions)


def score_kmer_context(
    body: str,
    k: str,
    k_plus_1: str,
) -> tuple[int, int, float]:
    """Return ``(count(k), count(k_plus_1), P(k_plus_1 | k))`` for a string body.

    .. warning::
        This function builds a new ``BwtIndex`` on every call.  For repeated
        queries over the same corpus, build the index once with
        ``build_corpus_index`` or ``build_genome_index`` and call
        ``index.next_token_prob`` directly.

    Parameters
    ----------
    body:
        Corpus string (genomic sequence, English text, etc.).
    k:
        Base context string.
    k_plus_1:
        Extended string (typically ``k`` concatenated with one token).

    Returns
    -------
    tuple[int, int, float]
    """
    index = BwtIndex(body)
    freq_k = index.count(k)
    freq_k1 = index.count(k_plus_1)
    prob = freq_k1 / freq_k if freq_k > 0 else 0.0
    return freq_k, freq_k1, prob

