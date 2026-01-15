# High-Impact Examples: Directed to Undirected Graph Conversion

This repository demonstrates the **prime graph technique** for converting directed graphs to undirected bipartite graphs while preserving all structural information.

## Overview

### The Prime Graph Technique

For any directed graph G, we can create an equivalent undirected bipartite graph H using "prime nodes":

```
Directed edge:  A → B

Undirected representation:  A — B' — B
                               (prime node)
```

**Key Properties:**
- ✓ **Lossless**: Original directed graph can be perfectly recovered
- ✓ **Bipartite**: Creates a two-partition graph structure
- ✓ **Preserves**: All connectivity, reachability, and path information
- ✓ **Enables**: Use of undirected graph algorithms on directed data

## Quick Start

### Prerequisites

```bash
pip install networkx matplotlib numpy
```

### Running the Examples

```bash
jupyter notebook directed_to_undirected_examples.ipynb
```

## Examples Included

### 1. Citation Network (Academic Papers)
**Real-World Use Case**: Academic citation analysis

- **Original Problem**: Papers cite other papers (directed relationship)
- **Why Convert**: Apply community detection to find research clusters
- **Benefit**: Identify influential papers using bipartite centrality measures

**Graph Properties:**
- Typically a DAG (no citation cycles)
- Power-law degree distribution
- Clear temporal ordering

### 2. Gene Regulatory Network
**Real-World Use Case**: Biological systems analysis

- **Original Problem**: Genes regulate other genes (directed regulation)
- **Why Convert**: Find gene modules using undirected clustering algorithms
- **Benefit**: Apply spectral methods for pathway analysis

**Graph Properties:**
- Complex regulatory feedback loops
- Hub-and-spoke structures (master regulators)
- Small-world network characteristics

### 3. Web Page Network
**Real-World Use Case**: Web structure analysis

- **Original Problem**: Hyperlinks are directional (Page A → Page B)
- **Why Convert**: Analyze link patterns with algorithms requiring undirected graphs
- **Benefit**: Different perspective on web topology

**Graph Properties:**
- Scale-free distribution
- Bow-tie structure
- High clustering coefficient

### 4. Workflow/Task Dependency Graph
**Real-World Use Case**: Project management and data pipelines

- **Original Problem**: Task dependencies form a DAG (Task A before Task B)
- **Why Convert**: Apply bipartite matching for resource allocation
- **Benefit**: Find critical paths using undirected algorithms

**Graph Properties:**
- Strictly acyclic (DAG)
- Clear topological ordering
- Often has layered structure

### 5. Social Network
**Real-World Use Case**: Social media analysis

- **Original Problem**: Follow relationships are directional (A follows B ≠ B follows A)
- **Why Convert**: Community detection for user clustering
- **Benefit**: Analyze influence patterns with undirected metrics

**Graph Properties:**
- High reciprocity (mutual follows)
- Power-law degree distribution
- Strong community structure

### 6. Comparative Analysis
**Real-World Use Case**: Understanding algorithmic performance

- **Demonstrates**: Technique works across different graph topologies
- **Includes**: Random (Erdős-Rényi) and Scale-Free (Barabási-Albert) graphs
- **Shows**: Preservation of structural properties

## Code Architecture

### Core Functions

```python
def directed_to_prime_graph(G):
    """
    Convert directed graph to undirected bipartite graph.

    Args:
        G: NetworkX DiGraph

    Returns:
        H: NetworkX Graph (undirected bipartite)
    """
```

```python
def prime_graph_to_directed(H):
    """
    Recover original directed graph from prime graph.

    Args:
        H: NetworkX Graph (prime graph)

    Returns:
        DG: NetworkX DiGraph
    """
```

```python
def visualize_conversion(G, H, title):
    """
    Visualize the conversion process:
    - Original directed graph
    - Prime graph (bipartite)
    - Recovered directed graph
    """
```

## Visual Examples

Each example includes three-panel visualization:

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  Original Directed  │   Prime Graph       │   Recovered Graph   │
│      Graph          │   (Bipartite)       │   (Verified)        │
│                     │                     │                     │
│    A ──→ B          │   A ─ Bp ─ B       │    A ──→ B          │
│    │     ↓          │   │   │    │       │    │     ↓          │
│    └───→ C          │   └─ Cp ─ C       │    └───→ C          │
│                     │                     │                     │
│  Blue circles       │ Blue circles (orig) │  Green circles      │
│  Directed arrows    │ Red squares (prime) │  Directed arrows    │
│                     │ Undirected edges    │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

## Applications

### When to Use This Technique

1. **Algorithm Compatibility**
   - Need to use undirected algorithms (Louvain, spectral clustering)
   - Working with libraries that only support undirected graphs
   - Comparing directed and undirected perspectives

2. **Analysis Requirements**
   - Bipartite projections
   - Spectral analysis requiring symmetric matrices
   - Community detection in directed networks

3. **Computational Benefits**
   - Some algorithms faster on undirected graphs
   - Symmetrization without information loss
   - Alternative graph embedding methods

### Specific Use Cases

| Domain | Application | Benefits |
|--------|-------------|----------|
| **Biology** | Gene regulatory networks | Module detection, pathway analysis |
| **Social Science** | Citation networks | Co-citation analysis, influence metrics |
| **Web Analytics** | Hyperlink graphs | Page importance, community structure |
| **Software Engineering** | Dependency graphs | Impact analysis, module clustering |
| **Data Science** | Workflow DAGs | Resource optimization, parallel execution |

## Mathematical Properties

### What's Preserved

The prime graph transformation preserves:

1. **Structural Isomorphism**: G ≅ prime_to_directed(directed_to_prime(G))
2. **Reachability**: If path exists from u to v in G, corresponding path exists in H
3. **Connectivity**: Strongly connected components map to connected components
4. **Degree Information**: In/out-degrees encoded in bipartite structure

### Graph Metrics

| Original Graph (Directed) | Prime Graph (Undirected) |
|--------------------------|--------------------------|
| In-degree of v | Degree of v in H |
| Out-degree of v | Number of prime nodes v' |
| PageRank | Related to bipartite centrality |
| Strongly connected components | Connected components pattern |

## Performance Characteristics

### Space Complexity

- **Nodes**: n (original) → n + m (prime), where m = number of edges
- **Edges**: m (directed) → 2m (undirected)
- **Overall**: Linear increase proportional to edge count

### Time Complexity

- **Conversion**: O(m) where m = number of edges
- **Recovery**: O(m) where m = number of edges in prime graph
- **Verification**: O(n + m) for isomorphism check

## Validation

All examples include automatic verification:

```python
# Every conversion is verified for isomorphism
is_isomorphic = nx.is_isomorphic(original_graph, recovered_graph)
# Result: True for all examples ✓
```

## Advanced Topics

### Weighted Graphs

The technique extends to weighted directed graphs:
- Store weights as edge attributes in prime graph
- Preserve during forward and reverse conversion

### Multi-Graphs

For graphs with multiple edges:
- Each directed edge gets its own prime node
- Preserves multiplicity information

### Attributed Graphs

Node and edge attributes are preserved:
- Original node attributes stay with original nodes
- Edge attributes transfer to prime nodes

## Research Applications

This technique has been used in:

1. **Network Alignment**: Comparing directed biological networks
2. **Graph Neural Networks**: Alternative encoding for directed graphs
3. **Spectral Methods**: Applying symmetric Laplacian to directed graphs
4. **Community Detection**: Finding clusters in directed social networks

## Citation

If you use this technique in your research, please cite the original paper:

```
[Add your paper citation here]
```

## Further Reading

- Fan Chung (2005). "Laplacians and the Cheeger inequality for directed graphs"
- Network alignment methods using prime graphs
- Spectral analysis of directed networks

## Contributing

Contributions welcome! Areas of interest:
- Additional real-world examples
- Performance optimizations
- Extended applications
- Visualization improvements

## License

See LICENSE file for details.

## Contact

For questions or collaborations, please open an issue in this repository.

---

## Quick Reference

### Basic Usage

```python
import networkx as nx
from directed_to_undirected_examples import directed_to_prime_graph, prime_graph_to_directed

# Create directed graph
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (1, 3)])

# Convert to prime graph
H = directed_to_prime_graph(G)

# Verify it's bipartite
print(nx.is_bipartite(H))  # True

# Recover original
G_recovered = prime_graph_to_directed(H)

# Verify isomorphism
print(nx.is_isomorphic(G, G_recovered))  # True
```

### Visualization

```python
from directed_to_undirected_examples import visualize_conversion

visualize_conversion(G, H, title="My Graph Conversion")
```

---

**Last Updated**: January 2026
