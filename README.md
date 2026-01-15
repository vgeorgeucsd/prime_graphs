# prime_graphs
Code for converting directed graphs to undirected graphs using the prime graph technique.

## Quick Start

See **[EXAMPLES_README.md](EXAMPLES_README.md)** for comprehensive examples and documentation.

## High-Impact Examples

The repository includes a detailed Jupyter notebook ([directed_to_undirected_examples.ipynb](directed_to_undirected_examples.ipynb)) with 6+ real-world examples:

1. **Citation Network** - Academic paper citations
2. **Gene Regulatory Network** - Biological gene regulation
3. **Web Page Network** - Hyperlink structures
4. **Workflow DAG** - Task dependency graphs
5. **Social Network** - Follow/follower relationships
6. **Comparative Analysis** - Random vs Scale-Free graphs

Each example demonstrates:
- The conversion process (directed → undirected bipartite)
- Full recovery verification (lossless transformation)
- Real-world use cases and applications
- Visual comparisons with three-panel layouts

## Core Technique

The **prime graph technique** converts any directed graph to an undirected bipartite graph:

```
Directed:    A ──→ B
Undirected:  A ── B' ── B  (where B' is a "prime node")
```

**Properties:**
- ✓ Lossless (perfect recovery of original graph)
- ✓ Creates bipartite structure
- ✓ Enables use of undirected algorithms
- ✓ Preserves all connectivity information

## Files

- `directed_to_undirected_examples.ipynb` - Comprehensive examples notebook
- `EXAMPLES_README.md` - Detailed documentation
- `make_prime_graphs.ipynb` - Original implementation
- `DirectedGraphLaplacian.m` - MATLAB Laplacian computation
- `spectral_graph_clustering.m` - MATLAB clustering code
- `network_alignment.ipynb` - Network alignment examples

## Citation

If you use this code, please cite our paper.
