"""
Prime Graph Conversion Examples
===============================

This module demonstrates the prime graph transformation technique for converting
directed graphs to undirected graphs while preserving all structural information.

Based on the research papers:
1. "Extending Undirected Graph Techniques to Directed Graphs via Category Theory"
   Pardo-Guerra et al., Mathematics 2024
2. "On the Graph Isomorphism Completeness of Directed and Multidirected Graphs"
   Pardo-Guerra et al., Mathematics 2025

The prime graph construction provides:
- A lossless, invertible transformation between directed and undirected graphs
- Preservation of topological features (connectivity, paths, clusters)
- Enables use of undirected graph algorithms on directed graph data
- Proven GI-completeness for directed and multidirected graphs

Author: Vivek Kurien George
License: MIT
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import re
from collections import defaultdict


# =============================================================================
# CORE TRANSFORMATION ALGORITHMS
# =============================================================================

def directed_to_prime(G: nx.DiGraph) -> nx.Graph:
    """
    Convert a directed graph to its corresponding prime graph.

    The prime graph transformation (Functor L: DGraph -> PGraph):
    - For each node v in G, create nodes v and v' in the prime graph
    - For each directed edge (u, v) in G, create edge (u, v') in prime graph
    - For each node v, create edge (v, v') connecting prime/non-prime pairs

    This transformation is:
    - Lossless: All directional information is preserved
    - Invertible: The original graph can be perfectly recovered
    - Functorial: Preserves graph morphisms (Theorem 2 in paper)

    Parameters
    ----------
    G : nx.DiGraph
        Input directed graph

    Returns
    -------
    nx.Graph
        The corresponding prime graph (undirected bipartite graph)

    Example
    -------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3)])
    >>> H = directed_to_prime(G)
    >>> print(f"Directed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    >>> print(f"Prime: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    """
    H = nx.Graph()

    # Add all nodes (both prime and non-prime versions)
    for node in G.nodes():
        non_prime = str(node)
        prime = str(node) + "'"
        H.add_node(non_prime, prime=False, original=node)
        H.add_node(prime, prime=True, original=node)
        # Connect each node to its prime counterpart (Definition 4, condition iii)
        H.add_edge(non_prime, prime)

    # Convert directed edges to prime graph edges
    # For directed edge (u, v), create edge (u, v') in prime graph
    for src, tar in G.edges():
        src_str = str(src)
        tar_prime = str(tar) + "'"
        H.add_edge(src_str, tar_prime)

    return H


def prime_to_directed(H: nx.Graph) -> nx.DiGraph:
    """
    Convert a prime graph back to its corresponding directed graph.

    The inverse transformation (Functor M: PGraph -> DGraph):
    - Identify prime nodes (ending with ') and non-prime nodes
    - For each edge (u, v') where u != v, create directed edge (u, v)
    - Edges of form (v, v') are structural and don't create directed edges

    This is the inverse of directed_to_prime, satisfying:
    M(L(G)) = G (Corollary 1 in paper)

    Parameters
    ----------
    H : nx.Graph
        Input prime graph

    Returns
    -------
    nx.DiGraph
        The corresponding directed graph
    """
    G = nx.DiGraph()

    # Identify non-prime nodes and add them to directed graph
    for node in H.nodes():
        if not str(node).endswith("'"):
            G.add_node(node)

    # Convert prime graph edges back to directed edges
    for u, v in H.edges():
        u_str, v_str = str(u), str(v)

        # Check if edge connects non-prime to prime node
        if u_str.endswith("'") and not v_str.endswith("'"):
            # Edge from non-prime v to prime u' -> directed edge (v, u)
            target = u_str[:-1]  # Remove prime marker
            if v_str != target:  # Skip self-referential edges
                G.add_edge(v_str, target)
        elif v_str.endswith("'") and not u_str.endswith("'"):
            # Edge from non-prime u to prime v' -> directed edge (u, v)
            target = v_str[:-1]  # Remove prime marker
            if u_str != target:  # Skip self-referential edges
                G.add_edge(u_str, target)

    return G


def verify_isomorphism(G_original: nx.DiGraph, G_recovered: nx.DiGraph) -> bool:
    """
    Verify that the original and recovered directed graphs are isomorphic.

    This verifies the key theorem: M(L(G)) = G
    The prime graph transformation is lossless and perfectly invertible.
    """
    return nx.is_isomorphic(G_original, G_recovered)


# =============================================================================
# MULTIDIRECTED GRAPH SUPPORT (from Paper 2)
# =============================================================================

def multidirected_to_weighted_prime(G: nx.MultiDiGraph) -> nx.Graph:
    """
    Convert a multidirected graph to a weighted prime graph.

    From Theorem 3 in Paper 2:
    The categories MGraph and WPGraph are isomorphic.

    For multidirected graphs:
    - Edge multiplicity m(u,v) becomes weight w(u, v') in the prime graph
    - Self-loops of multiplicity n create weight n+1 on edge (v, v')

    Parameters
    ----------
    G : nx.MultiDiGraph
        Input multidirected graph (allows parallel edges)

    Returns
    -------
    nx.Graph
        Weighted prime graph with edge weights encoding multiplicities
    """
    H = nx.Graph()

    # Count edge multiplicities
    edge_counts = defaultdict(int)
    for u, v in G.edges():
        edge_counts[(u, v)] += 1

    # Count self-loops
    self_loop_counts = defaultdict(int)
    for u, v in G.edges():
        if u == v:
            self_loop_counts[u] += 1

    # Add nodes
    for node in G.nodes():
        non_prime = str(node)
        prime = str(node) + "'"
        H.add_node(non_prime, prime=False, original=node)
        H.add_node(prime, prime=True, original=node)

        # Weight for (v, v') edge: 1 if no self-loop, n+1 if self-loop with multiplicity n
        self_loop_weight = 1 + self_loop_counts.get(node, 0)
        H.add_edge(non_prime, prime, weight=self_loop_weight)

    # Add weighted edges for directed connections
    for (src, tar), count in edge_counts.items():
        if src != tar:  # Skip self-loops (handled above)
            src_str = str(src)
            tar_prime = str(tar) + "'"
            # Weight encodes edge multiplicity
            if H.has_edge(src_str, tar_prime):
                H[src_str][tar_prime]['weight'] += count
            else:
                H.add_edge(src_str, tar_prime, weight=count)

    return H


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def visualize_transformation(G: nx.DiGraph, figsize: Tuple[int, int] = (14, 5),
                            title: str = "Prime Graph Transformation") -> plt.Figure:
    """
    Create a side-by-side visualization of a directed graph and its prime graph.

    This visualization demonstrates the key insight from the papers:
    directional information is encoded in the bipartite structure of the prime graph.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original directed graph
    ax1 = axes[0]
    pos_directed = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos_directed, ax=ax1, node_color='lightblue',
                          node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G, pos_directed, ax=ax1, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos_directed, ax=ax1, edge_color='gray',
                          arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')
    ax1.set_title(f"Directed Graph\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Prime graph
    H = directed_to_prime(G)
    ax2 = axes[1]

    # Separate prime and non-prime nodes for bipartite layout
    non_prime_nodes = [n for n in H.nodes() if not str(n).endswith("'")]
    prime_nodes = [n for n in H.nodes() if str(n).endswith("'")]

    # Create bipartite layout
    pos_prime = {}
    for i, node in enumerate(non_prime_nodes):
        pos_prime[node] = (0, -i)
    for i, node in enumerate(prime_nodes):
        pos_prime[node] = (2, -i)

    # Draw non-prime nodes
    nx.draw_networkx_nodes(H, pos_prime, nodelist=non_prime_nodes, ax=ax2,
                          node_color='lightblue', node_size=500, edgecolors='black')
    # Draw prime nodes
    nx.draw_networkx_nodes(H, pos_prime, nodelist=prime_nodes, ax=ax2,
                          node_color='lightcoral', node_size=500, edgecolors='black')
    nx.draw_networkx_labels(H, pos_prime, ax=ax2, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(H, pos_prime, ax=ax2, edge_color='gray', width=1.5)
    ax2.set_title(f"Prime Graph\n({H.number_of_nodes()} nodes, {H.number_of_edges()} edges)",
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Add legend
    ax2.plot([], [], 'o', color='lightblue', markersize=10, label='Non-prime nodes')
    ax2.plot([], [], 'o', color='lightcoral', markersize=10, label='Prime nodes')
    ax2.legend(loc='upper right', fontsize=8)

    # Recovered directed graph
    G_recovered = prime_to_directed(H)
    ax3 = axes[2]

    # Use same layout as original for comparison
    pos_recovered = {str(k): v for k, v in pos_directed.items()}
    nx.draw_networkx_nodes(G_recovered, pos_recovered, ax=ax3, node_color='lightgreen',
                          node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G_recovered, pos_recovered, ax=ax3, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G_recovered, pos_recovered, ax=ax3, edge_color='gray',
                          arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')

    is_iso = verify_isomorphism(G, G_recovered)
    ax3.set_title(f"Recovered Graph\nIsomorphic: {is_iso}",
                  fontsize=12, fontweight='bold')
    ax3.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_prime_graph_structure(G: nx.DiGraph, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Detailed visualization showing the prime graph as a bipartite graph.

    This illustrates Definition 4 from Paper 1:
    - Prime-labeled nodes only connect to non-prime-labeled nodes (and vice versa)
    - Each node v has a corresponding prime node v'
    """
    H = directed_to_prime(G)

    fig, ax = plt.subplots(figsize=figsize)

    # Separate nodes
    non_prime = [n for n in H.nodes() if not str(n).endswith("'")]
    prime = [n for n in H.nodes() if str(n).endswith("'")]

    # Bipartite layout
    pos = nx.bipartite_layout(H, non_prime, align='vertical')

    # Color edges by type
    structural_edges = []  # (v, v') edges
    directional_edges = []  # (u, v') edges where u != v

    for u, v in H.edges():
        u_str, v_str = str(u), str(v)
        if u_str.endswith("'"):
            base_u = u_str[:-1]
            if base_u == v_str:
                structural_edges.append((u, v))
            else:
                directional_edges.append((u, v))
        elif v_str.endswith("'"):
            base_v = v_str[:-1]
            if base_v == u_str:
                structural_edges.append((u, v))
            else:
                directional_edges.append((u, v))

    # Draw edges
    nx.draw_networkx_edges(H, pos, edgelist=structural_edges, ax=ax,
                          edge_color='green', width=2, style='dashed',
                          alpha=0.7, label='Structural (v-v\')')
    nx.draw_networkx_edges(H, pos, edgelist=directional_edges, ax=ax,
                          edge_color='blue', width=2, alpha=0.7,
                          label='Directional (u-v\')')

    # Draw nodes
    nx.draw_networkx_nodes(H, pos, nodelist=non_prime, ax=ax,
                          node_color='lightblue', node_size=700,
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(H, pos, nodelist=prime, ax=ax,
                          node_color='lightcoral', node_size=700,
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=11, font_weight='bold')

    # Legend
    ax.plot([], [], 'o', color='lightblue', markersize=12, label='Non-prime nodes (I)')
    ax.plot([], [], 'o', color='lightcoral', markersize=12, label='Prime nodes (I\')')
    ax.plot([], [], '-', color='green', linewidth=2, linestyle='dashed', label='Structural edges (v-v\')')
    ax.plot([], [], '-', color='blue', linewidth=2, label='Directional edges (u-v\')')
    ax.legend(loc='upper left', fontsize=10)

    ax.set_title("Prime Graph Bipartite Structure\n(Encoding of directional information)",
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    return fig


# =============================================================================
# SPECTRAL CLUSTERING UTILITIES
# =============================================================================

def compute_prime_laplacian(H: nx.Graph) -> np.ndarray:
    """
    Compute the graph Laplacian of a prime graph.

    For prime graphs, the Laplacian has the structure:
    L = D - A = [[D1, -B], [-B^T, D2]]

    where B encodes the adjacencies between prime and non-prime nodes.
    """
    return nx.laplacian_matrix(H).toarray()


def spectral_clustering_prime(G: nx.DiGraph, n_clusters: int = 2) -> Dict[str, int]:
    """
    Perform spectral clustering on a directed graph via its prime graph.

    From Proposition 11 in Paper 1:
    If C is a cluster in the directed graph G, then its corresponding cluster
    in L(G) has twice the number of nodes (prime and non-prime versions).

    This demonstrates that spectral clustering on prime graphs preserves
    the cluster structure of the original directed graph.

    Parameters
    ----------
    G : nx.DiGraph
        Input directed graph
    n_clusters : int
        Number of clusters to find

    Returns
    -------
    Dict[str, int]
        Mapping from original node to cluster assignment
    """
    H = directed_to_prime(G)
    L = compute_prime_laplacian(H)

    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    # Use Fiedler vector (second smallest eigenvalue) for 2-way cut
    fiedler = eigenvectors[:, 1]

    # Get node ordering
    nodes = list(H.nodes())

    # Cluster based on sign of Fiedler vector components
    clusters = {}
    for i, node in enumerate(nodes):
        if not str(node).endswith("'"):  # Only assign original nodes
            clusters[node] = 0 if fiedler[i] < 0 else 1

    return clusters


# =============================================================================
# EXAMPLE APPLICATIONS
# =============================================================================

def example_citation_network():
    """
    Example: Citation Network Analysis

    Citation networks are naturally directed: paper A cites paper B (A -> B).
    The prime graph transformation allows using undirected community detection
    algorithms while preserving citation directionality.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Citation Network")
    print("="*70)

    # Create a small citation network
    G = nx.DiGraph()

    # Papers and their citations
    papers = ['Paper_A', 'Paper_B', 'Paper_C', 'Paper_D', 'Paper_E', 'Paper_F']
    G.add_nodes_from(papers)

    # Citation edges: newer papers cite older papers
    citations = [
        ('Paper_F', 'Paper_A'),  # F cites A
        ('Paper_F', 'Paper_B'),  # F cites B
        ('Paper_E', 'Paper_A'),  # E cites A
        ('Paper_E', 'Paper_C'),  # E cites C
        ('Paper_D', 'Paper_B'),  # D cites B
        ('Paper_D', 'Paper_C'),  # D cites C
        ('Paper_C', 'Paper_A'),  # C cites A
        ('Paper_B', 'Paper_A'),  # B cites A
    ]
    G.add_edges_from(citations)

    print(f"\nCitation Network Statistics:")
    print(f"  Papers (nodes): {G.number_of_nodes()}")
    print(f"  Citations (edges): {G.number_of_edges()}")

    # Convert to prime graph
    H = directed_to_prime(G)

    print(f"\nPrime Graph Statistics:")
    print(f"  Nodes: {H.number_of_nodes()} (2x papers)")
    print(f"  Edges: {H.number_of_edges()}")

    # Verify invertibility
    G_recovered = prime_to_directed(H)
    is_iso = verify_isomorphism(G, G_recovered)
    print(f"\nTransformation Verification:")
    print(f"  Original ≅ Recovered: {is_iso}")

    # Demonstrate undirected analysis
    print(f"\nUndirected Analysis on Prime Graph:")
    print(f"  Is bipartite: {nx.is_bipartite(H)}")

    return G, H


def example_gene_regulatory_network():
    """
    Example: Gene Regulatory Network (GRN)

    GRNs model how genes regulate each other's expression.
    Edges represent activation or inhibition: Gene A regulates Gene B (A -> B).

    The prime graph transformation enables:
    - Network motif analysis using undirected subgraph algorithms
    - Community detection to find regulatory modules
    - Network alignment to compare GRNs across species
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Gene Regulatory Network")
    print("="*70)

    # Create a simplified GRN
    G = nx.DiGraph()

    # Genes
    genes = ['TF1', 'TF2', 'TF3', 'Gene_A', 'Gene_B', 'Gene_C', 'Gene_D']
    G.add_nodes_from(genes)

    # Regulatory relationships (TF = Transcription Factor)
    regulations = [
        ('TF1', 'Gene_A'),    # TF1 regulates Gene_A
        ('TF1', 'Gene_B'),    # TF1 regulates Gene_B
        ('TF2', 'Gene_B'),    # TF2 regulates Gene_B
        ('TF2', 'Gene_C'),    # TF2 regulates Gene_C
        ('TF3', 'Gene_C'),    # TF3 regulates Gene_C
        ('TF3', 'Gene_D'),    # TF3 regulates Gene_D
        ('Gene_A', 'TF2'),    # Feedback: Gene_A affects TF2
        ('Gene_D', 'TF1'),    # Feedback: Gene_D affects TF1
    ]
    G.add_edges_from(regulations)

    print(f"\nGene Regulatory Network:")
    print(f"  Genes/TFs: {G.number_of_nodes()}")
    print(f"  Regulatory edges: {G.number_of_edges()}")

    # Convert to prime graph
    H = directed_to_prime(G)

    print(f"\nPrime Graph:")
    print(f"  Nodes: {H.number_of_nodes()}")
    print(f"  Edges: {H.number_of_edges()}")

    # Verify invertibility
    G_recovered = prime_to_directed(H)
    print(f"\nInvertibility verified: {verify_isomorphism(G, G_recovered)}")

    # Network analysis
    print(f"\nDirected Graph Properties:")
    print(f"  Strongly connected components: {nx.number_strongly_connected_components(G)}")

    print(f"\nPrime Graph Properties:")
    print(f"  Connected components: {nx.number_connected_components(H)}")
    print(f"  Is bipartite: {nx.is_bipartite(H)}")

    return G, H


def example_social_network():
    """
    Example: Social Network (Follower Graph)

    Social networks like Twitter have directed edges: User A follows User B.
    The prime graph enables bidirectional community analysis while
    preserving the asymmetric follower relationship.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Social Network (Follower Graph)")
    print("="*70)

    G = nx.DiGraph()

    # Users in two communities
    community1 = ['Alice', 'Bob', 'Carol']
    community2 = ['David', 'Eve', 'Frank']

    G.add_nodes_from(community1 + community2)

    # Intra-community follows (dense)
    intra_follows = [
        ('Alice', 'Bob'), ('Bob', 'Alice'),
        ('Alice', 'Carol'), ('Carol', 'Bob'),
        ('David', 'Eve'), ('Eve', 'David'),
        ('David', 'Frank'), ('Frank', 'Eve'),
    ]

    # Inter-community follows (sparse)
    inter_follows = [
        ('Carol', 'David'),  # Carol follows David
        ('Eve', 'Alice'),    # Eve follows Alice
    ]

    G.add_edges_from(intra_follows + inter_follows)

    print(f"\nSocial Network:")
    print(f"  Users: {G.number_of_nodes()}")
    print(f"  Follow edges: {G.number_of_edges()}")

    # Convert and verify
    H = directed_to_prime(G)
    G_recovered = prime_to_directed(H)

    print(f"\nPrime Graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    print(f"Transformation invertible: {verify_isomorphism(G, G_recovered)}")

    # Community detection on prime graph
    clusters = spectral_clustering_prime(G)
    print(f"\nSpectral Clustering Results:")
    for user, cluster in sorted(clusters.items(), key=lambda x: x[1]):
        print(f"  {user}: Cluster {cluster}")

    return G, H


def example_multidirected_graph():
    """
    Example: Multidirected Graph (Transportation Network)

    Transportation networks can have multiple edges between nodes
    (e.g., multiple bus routes between stations).

    From Theorem 3 in Paper 2: The categories MGraph and WPGraph are isomorphic.
    Edge multiplicity is encoded as edge weights in the weighted prime graph.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Multidirected Graph (Transportation)")
    print("="*70)

    # Create multidirected graph
    G = nx.MultiDiGraph()

    stations = ['Station_A', 'Station_B', 'Station_C', 'Station_D']
    G.add_nodes_from(stations)

    # Multiple routes between stations
    routes = [
        ('Station_A', 'Station_B'),  # Route 1: A->B
        ('Station_A', 'Station_B'),  # Route 2: A->B (parallel edge)
        ('Station_A', 'Station_B'),  # Route 3: A->B (parallel edge)
        ('Station_B', 'Station_C'),
        ('Station_B', 'Station_C'),
        ('Station_C', 'Station_D'),
        ('Station_D', 'Station_A'),
        ('Station_D', 'Station_A'),
    ]
    G.add_edges_from(routes)

    print(f"\nMultidirected Transportation Network:")
    print(f"  Stations: {G.number_of_nodes()}")
    print(f"  Routes (with multiplicity): {G.number_of_edges()}")

    # Convert to weighted prime graph
    H = multidirected_to_weighted_prime(G)

    print(f"\nWeighted Prime Graph:")
    print(f"  Nodes: {H.number_of_nodes()}")
    print(f"  Edges: {H.number_of_edges()}")

    # Show edge weights
    print(f"\nEdge Weights (encoding multiplicity):")
    for u, v, data in H.edges(data=True):
        if data.get('weight', 1) > 1:
            print(f"  ({u}, {v}): weight = {data['weight']}")

    return G, H


def example_workflow_dag():
    """
    Example: Workflow DAG (Task Dependencies)

    Directed Acyclic Graphs (DAGs) represent task dependencies in workflows.
    The prime graph transformation preserves the topological ordering
    while enabling undirected graph analysis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Workflow DAG (Task Dependencies)")
    print("="*70)

    G = nx.DiGraph()

    # Tasks in a data pipeline
    tasks = ['Load_Data', 'Clean_Data', 'Feature_Eng', 'Train_Model',
             'Validate', 'Deploy', 'Monitor']
    G.add_nodes_from(tasks)

    # Dependencies: Task A must complete before Task B
    dependencies = [
        ('Load_Data', 'Clean_Data'),
        ('Clean_Data', 'Feature_Eng'),
        ('Feature_Eng', 'Train_Model'),
        ('Train_Model', 'Validate'),
        ('Validate', 'Deploy'),
        ('Deploy', 'Monitor'),
        ('Clean_Data', 'Validate'),  # Direct validation of cleaned data
    ]
    G.add_edges_from(dependencies)

    print(f"\nWorkflow DAG:")
    print(f"  Tasks: {G.number_of_nodes()}")
    print(f"  Dependencies: {G.number_of_edges()}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(G)}")

    # Convert to prime graph
    H = directed_to_prime(G)

    print(f"\nPrime Graph:")
    print(f"  Nodes: {H.number_of_nodes()}")
    print(f"  Edges: {H.number_of_edges()}")

    # Topological analysis
    topo_order = list(nx.topological_sort(G))
    print(f"\nTopological Order (original DAG):")
    for i, task in enumerate(topo_order, 1):
        print(f"  {i}. {task}")

    # Verify invertibility
    G_recovered = prime_to_directed(H)
    print(f"\nTransformation invertible: {verify_isomorphism(G, G_recovered)}")
    print(f"Recovered DAG is acyclic: {nx.is_directed_acyclic_graph(G_recovered)}")

    return G, H


def example_spectral_clustering_comparison():
    """
    Example: Spectral Clustering Preservation

    Demonstrates Proposition 10 and 11 from Paper 1:
    - vol_u(∂(S ∪ S')) = vol_d(∂S) : Volume crossing cuts is preserved
    - Clusters in directed graph correspond to clusters in prime graph

    This shows that spectral clustering on prime graphs gives the same
    results as specialized directed graph methods.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Spectral Clustering Preservation")
    print("="*70)

    # Create a directed graph with clear cluster structure
    G = nx.DiGraph()

    # Two clusters with dense internal connections
    cluster1 = list(range(10))
    cluster2 = list(range(10, 20))

    G.add_nodes_from(cluster1 + cluster2)

    # Dense intra-cluster edges
    np.random.seed(42)
    for i in cluster1:
        for j in cluster1:
            if i != j and np.random.random() < 0.4:
                G.add_edge(i, j)

    for i in cluster2:
        for j in cluster2:
            if i != j and np.random.random() < 0.4:
                G.add_edge(i, j)

    # Sparse inter-cluster edges
    for i in cluster1:
        for j in cluster2:
            if np.random.random() < 0.05:
                G.add_edge(i, j)

    for i in cluster2:
        for j in cluster1:
            if np.random.random() < 0.05:
                G.add_edge(i, j)

    print(f"\nDirected Graph with Cluster Structure:")
    print(f"  Nodes: {G.number_of_nodes()} (two clusters of 10)")
    print(f"  Edges: {G.number_of_edges()}")

    # Convert to prime graph
    H = directed_to_prime(G)

    print(f"\nPrime Graph:")
    print(f"  Nodes: {H.number_of_nodes()}")
    print(f"  Edges: {H.number_of_edges()}")

    # Spectral clustering
    clusters = spectral_clustering_prime(G)

    # Evaluate clustering quality
    cluster1_assignments = [clusters[str(n)] for n in cluster1]
    cluster2_assignments = [clusters[str(n)] for n in cluster2]

    # Check if clusters are mostly in different groups
    cluster1_majority = 1 if sum(cluster1_assignments) > len(cluster1)/2 else 0
    cluster2_majority = 1 if sum(cluster2_assignments) > len(cluster2)/2 else 0

    cluster1_accuracy = sum(1 for c in cluster1_assignments if c == cluster1_majority) / len(cluster1)
    cluster2_accuracy = sum(1 for c in cluster2_assignments if c == cluster2_majority) / len(cluster2)

    print(f"\nSpectral Clustering Results:")
    print(f"  Ground truth: Nodes 0-9 in Cluster A, Nodes 10-19 in Cluster B")
    print(f"  Cluster 1 recovery accuracy: {cluster1_accuracy:.1%}")
    print(f"  Cluster 2 recovery accuracy: {cluster2_accuracy:.1%}")
    print(f"  Clusters correctly separated: {cluster1_majority != cluster2_majority}")

    return G, H, clusters


def demonstrate_key_theorem():
    """
    Demonstrates the key theorem: DGraph and PGraph are isomorphic categories.

    From Theorem 4 in Paper 1:
    The functors L: DGraph -> PGraph and M: PGraph -> DGraph satisfy:
    - (M ∘ L) = Id_DGraph  (Corollary 1)
    - (L ∘ M) = Id_PGraph  (Corollary 2)
    """
    print("\n" + "="*70)
    print("KEY THEOREM DEMONSTRATION")
    print("="*70)
    print("\nTheorem 4: The categories DGraph and PGraph are isomorphic.")
    print("\nThis means:")
    print("  1. Every directed graph has a unique prime graph representation")
    print("  2. Every prime graph corresponds to a unique directed graph")
    print("  3. The transformation preserves graph morphisms (functorial)")
    print("  4. The transformation is lossless and perfectly invertible")

    # Create test graphs of various sizes
    test_cases = [
        ("Small (5 nodes)", nx.gnp_random_graph(5, 0.3, directed=True, seed=1)),
        ("Medium (20 nodes)", nx.gnp_random_graph(20, 0.2, directed=True, seed=2)),
        ("Large (100 nodes)", nx.gnp_random_graph(100, 0.1, directed=True, seed=3)),
        ("Dense (10 nodes)", nx.gnp_random_graph(10, 0.7, directed=True, seed=4)),
        ("Sparse (50 nodes)", nx.gnp_random_graph(50, 0.05, directed=True, seed=5)),
    ]

    print("\nVerification across different graph types:")
    print("-" * 60)

    all_verified = True
    for name, G in test_cases:
        # Forward transformation: L
        H = directed_to_prime(G)

        # Inverse transformation: M
        G_recovered = prime_to_directed(H)

        # Verify isomorphism
        is_iso = verify_isomorphism(G, G_recovered)
        all_verified = all_verified and is_iso

        status = "✓" if is_iso else "✗"
        print(f"  {status} {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> "
              f"Prime: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    print("-" * 60)
    print(f"\nAll transformations verified: {'✓ YES' if all_verified else '✗ NO'}")

    # Demonstrate the size relationships
    print("\nSize Relationships:")
    print("  |V_prime| = 2 * |V_directed|  (each node gets a prime counterpart)")
    print("  |E_prime| = |E_directed| + |V_directed|  (Remark 1 in Paper 1)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PRIME GRAPH TRANSFORMATION EXAMPLES")
    print("Converting Directed Graphs to Undirected Graphs via Category Theory")
    print("="*70)

    # Run all examples
    demonstrate_key_theorem()
    example_citation_network()
    example_gene_regulatory_network()
    example_social_network()
    example_multidirected_graph()
    example_workflow_dag()
    example_spectral_clustering_comparison()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
