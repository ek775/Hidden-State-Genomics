# Knowledge Graph Theory

## Formal Construction

### Graph Definition

Our knowledge graph is a **typed heterogeneous multigraph** defined as:

**G = (V, E, τ, λ)**

where:
- **V** is the vertex set
- **E** is the edge set (multiset, allowing parallel edges)
- **τ: E → Σ** is the edge type function
- **λ: V ∪ E → A** assigns attributes to vertices and edges

### Vertex Set Construction

The vertex set V comprises two disjoint types:

**V = V_seq ∪ V_feat**

where:
- **V_seq** = {token strings from genomic sequences} (sequence tokens)
- **V_feat** = {0, 1, 2, ..., d-1} (SAE feature indices)

with d = expansion_factor × hidden_dimension (typically d = 10240 for ef8).

### Edge Construction

For each genomic sequence s with identifier id(s), we construct edges as follows:

1. **Tokenization**: Sequence s is tokenized into tokens T(s) = [t₁, t₂, ..., t_n]

2. **SAE Feature Extraction**: For each token position i ∈ {1, ..., n}:
   - Compute SAE activation vector **f_i** ∈ ℝ^d
   - Select strongest feature: f*_i = argmax_j(f_i[j])

3. **Edge Addition**: For each token-feature pair (t_i, f*_i):
   - Add edge e = (t_i, f*_i) to E
   - Assign edge type: τ(e) = t_i (the token itself types the edge)
   - Assign edge attributes: λ(e) includes:
     - `sequence`: id(s) (sequence identifier)
     - `chrom`: chromosome location (if applicable)
     - `start`, `end`: genomic coordinates
     - `strand`: strand orientation (+/-)
     - `annotations`: genomic feature metadata (genes, transcripts, exons, etc.)

### Graph Properties

**Directional Structure**: The graph is directed: edges point from sequence tokens (V_seq) to feature nodes (V_feat), representing the activation relationship: "token t_i activates feature f*_i"

**Heterogeneity**: Two vertex types with fundamentally different semantics:
- Sequence vertices represent genomic context (strings)
- Feature vertices represent learned SAE components (integers)

**Edge Multiplicity**: Multiple edges can exist between the same (token, feature) pair if that token appears in different sequences or genomic contexts.

**Edge Typing**: Unlike traditional knowledge graphs where edge types represent relationship semantics (e.g., "contains", "regulates"), our edges are typed by the **genomic token** that mediates the activation. This preserves the sequence-level information at the edge level.

### Formal Construction Algorithm

```
Input: Sequences S = {s₁, s₂, ..., s_m}, SAE model M
Output: Knowledge graph G = (V, E, τ, λ)

1. Initialize: V ← ∅, E ← ∅
2. For each sequence s ∈ S:
   a. Tokenize: T ← tokenize(s)
   b. Extract features: F ← M.forward(s)
   c. For each position i ∈ {1, ..., |T|}:
      i.   Get token t_i and feature activation f_i
      ii.  Select strongest feature: f*_i ← argmax(f_i)
      iii. Add vertices: V ← V ∪ {t_i, f*_i}
      iv.  Add edge: E ← E ∪ {(t_i, f*_i)}
      v.   Set edge type: τ((t_i, f*_i)) ← t_i
      vi.  Assign attributes: λ((t_i, f*_i)) ← extract_metadata(s, i)
3. Return G = (V, E, τ, λ)
```

### Implications for Analysis

1. **Token-to-Feature Mapping**: The graph explicitly represents which genomic tokens activate which SAE features across all sequences.

2. **Feature Centrality**: Feature vertices with high in-degree represent features activated by many different tokens/contexts, indicating broad genomic patterns.

3. **Token Centrality**: Token vertices with high out-degree (to different features) represent genomic motifs that trigger diverse feature activations.

4. **Path-Based Queries**: Paths of length 2 through shared features (token₁ → feature → token₂) identify genomic contexts with similar SAE representations.

5. **Subgraph Analysis**: Community detection on feature subgraphs reveals co-activated feature clusters representing higher-level genomic patterns.

### Implementation Notes

- Implemented in `hsg/featureanalysis/featureKG.py`
- Uses NetworkX MultiDiGraph for graph representation
- Genomic annotations from NCBI RefSeq GTF enriches edge attributes
- Serialized to JSON using `networkx.readwrite.json_graph.node_link_data`
