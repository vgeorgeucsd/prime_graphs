# Prime Graph Transformation Examples

This directory contains high-impact examples demonstrating the **prime graph transformation** technique for converting directed graphs to undirected graphs while preserving all structural information.

## Research Papers

These examples are based on the following publications:

1. **"Extending Undirected Graph Techniques to Directed Graphs via Category Theory"**
   - Authors: Sebastian Pardo-Guerra, Vivek Kurien George, Vikash Morar, Joshua Roldan, Gabriel Alex Silva
   - Journal: *Mathematics* 2024, 12, 1357
   - DOI: https://doi.org/10.3390/math12091357

2. **"On the Graph Isomorphism Completeness of Directed and Multidirected Graphs"**
   - Authors: Sebastian Pardo-Guerra, Vivek Kurien George, Gabriel A. Silva
   - Journal: *Mathematics* 2025, 13, 228
   - DOI: https://doi.org/10.3390/math13020228

## Key Results

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| **Theorem 4** (Paper 1) | The categories DGraph and PGraph are isomorphic | Establishes the fundamental bridge between directed and undirected graphs |
| **Corollary 1-2** | M(L(G)) = G and L(M(H)) = H | The transformation is perfectly invertible |
| **Theorem 3** (Paper 2) | MGraph ≅ WPGraph | Extends to multidirected graphs with edge weights |
| **Theorem 1-2** (Paper 2) | Directed graphs are GI-complete | Graph isomorphism complexity result |
| **Theorem 4-5** (Paper 2) | Multidirected graphs are GI-complete | Extends GI-completeness to multigraphs |

## Files

### `prime_graph_examples.py`
A comprehensive Python module with:
- Core transformation algorithms (`directed_to_prime`, `prime_to_directed`)
- Multidirected graph support (`multidirected_to_weighted_prime`)
- Visualization utilities
- Spectral clustering implementation
- Six real-world application examples

### `prime_graph_visualization_examples.ipynb`
Interactive Jupyter notebook with:
- Step-by-step visual demonstrations
- Citation network analysis
- Gene regulatory network analysis
- Spectral clustering preservation proof
- Multidirected graph (transportation network) example

## The Prime Graph Construction

For a directed graph G_d = (V_d, E_d), the corresponding prime graph G_u = (V_u, E_u) is constructed as:

1. **Nodes**: For each node v_i in V_d, create two nodes: v_i (non-prime) and v_i' (prime)
2. **Structural edges**: For each v_i, add edge (v_i, v_i')
3. **Directional edges**: For each directed edge (v_i, v_j) in E_d, add edge (v_i, v_j')

```
Directed Graph:        Prime Graph (Bipartite):

    A ──→ B            A ─── A'
                        │ ╲
                        │   ╲
                        │     B'
                        │   ╱
                        B ───
```

### Size Relationships

- |V_prime| = 2 × |V_directed|
- |E_prime| = |E_directed| + |V_directed|

## Application Examples

### 1. Citation Network Analysis
Citation networks are naturally directed (Paper A cites Paper B). The prime graph enables:
- Undirected community detection algorithms
- Network alignment between citation networks
- Centrality analysis preserving citation direction

### 2. Gene Regulatory Networks
GRNs model gene regulation (TF activates/inhibits Gene). The prime graph enables:
- Network motif analysis with undirected algorithms
- Community detection to find regulatory modules
- Network alignment across species

### 3. Social Networks (Follower Graphs)
Social networks have asymmetric follow relationships. The prime graph enables:
- Bidirectional community analysis
- Influence propagation modeling
- Network alignment for cross-platform analysis

### 4. Spectral Clustering
The transformation preserves minimum cuts and clusters:
- **Proposition 10**: vol_u(∂(S ∪ S')) = vol_d(∂S)
- **Proposition 11**: Clusters in G_d correspond to clusters in L(G_d) with 2x nodes

### 5. Transportation Networks (Multidirected)
Transportation can have multiple routes between stations:
- Edge multiplicity becomes edge weight
- Self-loops get weight n+1 for n loops
- Theorem 3: MGraph ≅ WPGraph

## Running the Examples

### Python Module
```python
from prime_graph_examples import (
    directed_to_prime,
    prime_to_directed,
    verify_isomorphism,
    visualize_transformation
)

# Run all examples
python prime_graph_examples.py
```

### Jupyter Notebook
```bash
jupyter notebook prime_graph_visualization_examples.ipynb
```

## Dependencies

- NetworkX >= 2.6
- NumPy >= 1.20
- Matplotlib >= 3.4

## License

MIT License - See the main repository LICENSE file.
