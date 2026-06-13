//! BWT / FM-index frequency counting exposed as a Python extension via PyO3.
//!
//! # Algorithm
//! Builds a full FM-index over the input text.  Pattern counting uses
//! backward search (Ferragina & Manzini 2000):
//!
//!   lo = C[c] + Occ[c][lo]
//!   hi = C[c] + Occ[c][hi]
//!
//! This runs in O(|pattern|) per query after preprocessing.  The Occ table
//! rows are built in parallel across the alphabet using Rayon, reducing
//! wall-clock construction time to roughly O(n) for DNA-scale alphabets.
//!
//! # Multi-sequence indexing
//! `BwtIndex::from_sequences` concatenates multiple sequences separated by
//! byte 0x01 (sits between the NUL sentinel and any printable character).
//! Queries of printable characters will never match across a boundary,
//! preserving per-sequence k-mer semantics.
//!
//! # Memory
//! Dense Occ table: σ × (n+1) × 4 bytes.  For DNA (σ≈5) at 100 M bases ≈ 2 GB.
//! Load chromosomes in batches if whole-genome indexing is not feasible.
//!
//! # SA construction
//! Currently O(n log n) comparisons via Rust's pdqsort; each suffix comparison
//! is O(n) in the worst case (highly repetitive text), giving O(n² log n) overall.
//! For sequences > ~50 MB, replace `build_suffix_array` with an SA-IS / libdivsufsort
//! binding to guarantee O(n) construction.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Suffix array (pdqsort — O(n log n) comparisons, O(n) per comparison worst case)
// ---------------------------------------------------------------------------

fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let mut sa: Vec<usize> = (0..text.len()).collect();
    sa.sort_unstable_by(|&a, &b| text[a..].cmp(&text[b..]));
    sa
}

// ---------------------------------------------------------------------------
// BWT from suffix array
// ---------------------------------------------------------------------------

fn build_bwt(text: &[u8], sa: &[usize]) -> Vec<u8> {
    let n = text.len();
    sa.iter()
        .map(|&i| if i == 0 { text[n - 1] } else { text[i - 1] })
        .collect()
}

// ---------------------------------------------------------------------------
// FM-index core
// ---------------------------------------------------------------------------

struct FmIndex {
    n: usize,
    bwt: Vec<u8>,
    /// C[j] = number of characters in text strictly less than alphabet[j].
    c: Vec<u32>,
    /// occ[j][i] = count of alphabet[j] in BWT[0..i].  Built in parallel.
    occ: Vec<Vec<u32>>,
    char_to_idx: [u8; 256],
    alphabet: Vec<u8>,
}

impl FmIndex {
    fn new(text: &[u8]) -> Self {
        let n = text.len();

        let sa = build_suffix_array(text);
        let bwt = build_bwt(text, &sa);

        // Collect alphabet from the BWT (= permutation of text).
        let mut seen = [false; 256];
        for &b in &bwt {
            seen[b as usize] = true;
        }
        let alphabet: Vec<u8> = (0u8..=255).filter(|&b| seen[b as usize]).collect();
        let sigma = alphabet.len();

        // Guard: u8::MAX is the "not found" sentinel; it must not be a valid index.
        assert!(
            sigma < 255,
            "alphabet has {} distinct bytes; u8::MAX sentinel would collide with index {}",
            sigma,
            sigma - 1
        );
        let mut char_to_idx = [u8::MAX; 256];
        for (i, &ch) in alphabet.iter().enumerate() {
            char_to_idx[ch as usize] = i as u8;
        }

        // Character totals for C array.
        let mut counts = vec![0u32; sigma];
        for &b in &bwt {
            counts[char_to_idx[b as usize] as usize] += 1;
        }
        let mut c = vec![0u32; sigma];
        let mut cumul = 0u32;
        for i in 0..sigma {
            c[i] = cumul;
            cumul += counts[i];
        }

        // Build each character's prefix-sum row in parallel (rayon).
        // par_iter().collect() preserves alphabet order.
        let occ: Vec<Vec<u32>> = alphabet
            .par_iter()
            .map(|&ch| {
                let mut row = vec![0u32; n + 1];
                for i in 0..n {
                    row[i + 1] = row[i] + (bwt[i] == ch) as u32;
                }
                row
            })
            .collect();

        FmIndex { n, bwt, c, occ, char_to_idx, alphabet }
    }

    /// Count exact occurrences of `pattern` in the original text via
    /// FM-index backward search.  Returns 0 if any character is absent from
    /// the alphabet.
    fn count(&self, pattern: &[u8]) -> usize {
        let mut lo = 0usize;
        let mut hi = self.n;

        for &ch in pattern.iter().rev() {
            let ci = self.char_to_idx[ch as usize];
            if ci == u8::MAX {
                return 0;
            }
            let ci = ci as usize;
            lo = self.c[ci] as usize + self.occ[ci][lo] as usize;
            hi = self.c[ci] as usize + self.occ[ci][hi] as usize;
            if lo >= hi {
                return 0;
            }
        }
        hi - lo
    }
}

// ---------------------------------------------------------------------------
// Python-visible type
// ---------------------------------------------------------------------------

/// FM-index over a string corpus.
///
/// Build over a single string with ``BwtIndex(body)`` or over multiple
/// sequences (e.g. whole chromosomes) with ``BwtIndex.from_sequences(seqs)``.
/// The inter-sequence separator byte (0x01) ensures queries of printable
/// characters never match across sequence boundaries.
///
/// The Occ table is constructed in parallel across alphabet characters using
/// Rayon, so construction time scales with available CPU cores.
#[pyclass]
struct BwtIndex {
    inner: FmIndex,
}

#[pymethods]
impl BwtIndex {
    /// Build an FM-index over a single string body (genomic sequence or text).
    #[new]
    fn new(body: &str) -> Self {
        let mut text = body.as_bytes().to_vec();
        text.push(0u8);
        BwtIndex { inner: FmIndex::new(&text) }
    }

    /// Build an FM-index over multiple sequences concatenated with 0x01
    /// inter-sequence separators.  Queries of printable characters will not
    /// span sequence boundaries.
    ///
    /// Pass one entry per chromosome (or per FASTA record).  Accepts any
    /// strings, so English-language paragraphs work equally well.
    #[staticmethod]
    fn from_sequences(sequences: Vec<String>) -> Self {
        let mut text: Vec<u8> = Vec::new();
        for (i, seq) in sequences.iter().enumerate() {
            if i > 0 {
                text.push(0x01u8); // inter-sequence separator
            }
            text.extend_from_slice(seq.as_bytes());
        }
        text.push(0u8); // sentinel
        BwtIndex { inner: FmIndex::new(&text) }
    }

    /// Exact occurrence count of ``pattern`` in the indexed corpus.
    fn count(&self, pattern: &str) -> usize {
        self.inner.count(pattern.as_bytes())
    }

    /// ``count(k_plus_1) / count(k)``, or 0.0 if ``k`` is absent.
    ///
    /// ``k_plus_1`` must be the full extended string (k ++ next token).
    fn next_token_prob(&self, k: &str, k_plus_1: &str) -> f64 {
        let freq_k = self.inner.count(k.as_bytes());
        if freq_k == 0 {
            return 0.0;
        }
        self.inner.count(k_plus_1.as_bytes()) as f64 / freq_k as f64
    }

    /// For each extension string return ``count(ext) / count(k)``.
    ///
    /// ``extensions`` should be the full k+1 strings for each vocabulary token.
    fn next_token_probs(&self, k: &str, extensions: Vec<String>) -> HashMap<String, f64> {
        let freq_k = self.inner.count(k.as_bytes());
        extensions
            .into_iter()
            .map(|ext| {
                let p = if freq_k == 0 {
                    0.0
                } else {
                    self.inner.count(ext.as_bytes()) as f64 / freq_k as f64
                };
                (ext, p)
            })
            .collect()
    }

    /// Compute ``next_token_probs`` for a batch of ``(k, extensions)`` pairs
    /// in parallel via Rayon.  Each element is ``(k, [ext1, ext2, ...])``,
    /// returns a list of ``dict[str, float]`` in the same order.
    ///
    /// Use this when scoring many distinct k-mer contexts at once, e.g. all
    /// unique k-mers in a set of genomic sequences.
    fn batch_next_token_probs(
        &self,
        queries: Vec<(String, Vec<String>)>,
    ) -> Vec<HashMap<String, f64>> {
        // FmIndex is read-only after construction and all its fields implement
        // Sync, so a shared reference is safe across Rayon threads without any
        // unsafe code.
        let inner = &self.inner;
        queries
            .into_par_iter()
            .map(|(k, exts)| {
                let freq_k = inner.count(k.as_bytes());
                exts.into_iter()
                    .map(|ext| {
                        let p = if freq_k == 0 {
                            0.0
                        } else {
                            inner.count(ext.as_bytes()) as f64 / freq_k as f64
                        };
                        (ext, p)
                    })
                    .collect()
            })
            .collect()
    }

    /// Raw occurrence counts for an arbitrary list of query strings.
    fn frequencies(&self, queries: Vec<String>) -> HashMap<String, usize> {
        queries
            .into_iter()
            .map(|q| (q.clone(), self.inner.count(q.as_bytes())))
            .collect()
    }

    /// Characters present in the corpus (excludes NUL sentinel and 0x01 separator).
    fn alphabet(&self) -> Vec<String> {
        self.inner
            .alphabet
            .iter()
            .filter(|&&b| b > 0x01u8)
            .map(|&b| (b as char).to_string())
            .collect()
    }

    /// BWT string for inspection.  NUL → ``$``, separator → ``|``.
    fn bwt_string(&self) -> String {
        let display: Vec<u8> = self
            .inner
            .bwt
            .iter()
            .map(|&b| match b {
                0x00 => b'$',
                0x01 => b'|',
                other => other,
            })
            .collect();
        String::from_utf8_lossy(&display).into_owned()
    }

    fn __len__(&self) -> usize {
        self.inner.n.saturating_sub(1)
    }

    fn __repr__(&self) -> String {
        format!("BwtIndex(len={}, alphabet={:?})", self.__len__(), self.alphabet())
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn bwt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BwtIndex>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (pure Rust, no Python runtime required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn index(body: &str) -> FmIndex {
        let mut text = body.as_bytes().to_vec();
        text.push(0u8);
        FmIndex::new(&text)
    }

    #[test]
    fn count_single_char() {
        let idx = index("AACGT");
        assert_eq!(idx.count(b"A"), 2);
        assert_eq!(idx.count(b"C"), 1);
        assert_eq!(idx.count(b"T"), 1);
    }

    #[test]
    fn count_kmer() {
        // "AACAAC" — "AC" appears twice, "AAC" appears twice, "CA" once.
        let idx = index("AACAAC");
        assert_eq!(idx.count(b"AC"), 2);
        assert_eq!(idx.count(b"AAC"), 2);
        assert_eq!(idx.count(b"CA"), 1);
    }

    #[test]
    fn count_absent_pattern() {
        let idx = index("ATCG");
        assert_eq!(idx.count(b"N"), 0);
        assert_eq!(idx.count(b"ATCGG"), 0);
    }

    #[test]
    fn next_token_prob_sums_to_one() {
        // Corpus: simple DNA where every context has a deterministic successor.
        let body = "ACGTACGTACGT";
        let idx = index(body);
        let exts: Vec<_> = ["A", "C", "G", "T"]
            .iter()
            .map(|&a| format!("AC{}", a))
            .collect();
        let total: f64 = exts
            .iter()
            .map(|ext| idx.count(ext.as_bytes()) as f64)
            .sum::<f64>()
            / idx.count(b"AC") as f64;
        // All AC continuations should account for all AC occurrences.
        assert!((total - 1.0).abs() < 1e-9, "probs sum to {total}");
    }

    #[test]
    fn zero_prob_when_k_absent() {
        let idx = index("ACGT");
        assert_eq!(idx.count(b"TTT"), 0);
    }

    #[test]
    fn separator_prevents_cross_boundary_match() {
        // "AT" at end of seq1, "CG" at start of seq2 — "ATCG" must NOT match.
        let mut text = b"GGAT".to_vec();
        text.push(0x01u8);
        text.extend_from_slice(b"CGTT");
        text.push(0u8);
        let idx = FmIndex::new(&text);
        assert_eq!(idx.count(b"ATCG"), 0, "cross-boundary match must not occur");
        assert_eq!(idx.count(b"AT"), 1);
        assert_eq!(idx.count(b"CG"), 1);
    }
}
