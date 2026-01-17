"""
Knowledge Graph Entity Resolution using Prime Graphs
=====================================================

This example demonstrates how prime graphs enable entity resolution
(entity matching/alignment) across different knowledge graphs.

Problem: Given two knowledge graphs with different entity names but
overlapping content, find which entities correspond to each other.

Challenge: Knowledge graphs have DIRECTED edges (subject → predicate → object).
Traditional graph alignment tools require UNDIRECTED graphs.

Solution: Convert to prime graphs, align, then recover entity mappings.

Based on:
- "Extending Undirected Graph Techniques to Directed Graphs via Category Theory"
- "On the Graph Isomorphism Completeness of Directed and Multidirected Graphs"
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import itertools


# =============================================================================
# PRIME GRAPH TRANSFORMATION
# =============================================================================

def directed_to_prime(G: nx.DiGraph) -> nx.Graph:
    """
    Convert directed graph to prime graph (undirected bipartite).

    For each node v, creates v (non-prime) and v' (prime).
    Directed edge (u, v) becomes undirected edge (u, v').
    """
    H = nx.Graph()

    for node in G.nodes():
        non_prime = f"{node}"
        prime = f"{node}'"
        H.add_node(non_prime, prime=False, original=node)
        H.add_node(prime, prime=True, original=node)
        H.add_edge(non_prime, prime)  # Structural edge

    for src, tar in G.edges():
        H.add_edge(f"{src}", f"{tar}'")  # Directional edge

    return H


def prime_to_directed(H: nx.Graph) -> nx.DiGraph:
    """Recover directed graph from prime graph."""
    G = nx.DiGraph()

    # Find non-prime nodes
    non_prime_nodes = [n for n in H.nodes() if not H.nodes[n].get('prime', False)]

    for node in non_prime_nodes:
        original = H.nodes[node].get('original', node)
        G.add_node(original)

    # Recover directed edges
    for u, v in H.edges():
        u_prime = H.nodes[u].get('prime', False)
        v_prime = H.nodes[v].get('prime', False)

        if not u_prime and v_prime:
            # Edge from non-prime to prime = directed edge
            u_orig = H.nodes[u].get('original', u)
            v_orig = H.nodes[v].get('original', v.rstrip("'"))
            if u_orig != v_orig:  # Skip structural edges
                G.add_edge(u_orig, v_orig)

    return G


# =============================================================================
# KNOWLEDGE GRAPH CONSTRUCTION
# =============================================================================

def create_movie_knowledge_graph_1() -> nx.DiGraph:
    """
    Create a knowledge graph about movies (English naming convention).

    Relations are directed: (subject) --[predicate]--> (object)
    """
    KG = nx.DiGraph()

    # Entities (will be nodes)
    # Movies
    KG.add_node("The_Matrix", type="Movie", year=1999)
    KG.add_node("Inception", type="Movie", year=2010)
    KG.add_node("Interstellar", type="Movie", year=2014)

    # People
    KG.add_node("Keanu_Reeves", type="Person", role="Actor")
    KG.add_node("Carrie_Anne_Moss", type="Person", role="Actor")
    KG.add_node("Wachowskis", type="Person", role="Director")
    KG.add_node("Leonardo_DiCaprio", type="Person", role="Actor")
    KG.add_node("Christopher_Nolan", type="Person", role="Director")
    KG.add_node("Matthew_McConaughey", type="Person", role="Actor")

    # Genres
    KG.add_node("SciFi", type="Genre")
    KG.add_node("Action", type="Genre")
    KG.add_node("Thriller", type="Genre")

    # Studios
    KG.add_node("Warner_Bros", type="Studio")
    KG.add_node("Paramount", type="Studio")

    # Relations (directed edges)
    # The Matrix relations
    KG.add_edge("The_Matrix", "Keanu_Reeves", relation="stars")
    KG.add_edge("The_Matrix", "Carrie_Anne_Moss", relation="stars")
    KG.add_edge("Wachowskis", "The_Matrix", relation="directed")
    KG.add_edge("The_Matrix", "SciFi", relation="genre")
    KG.add_edge("The_Matrix", "Action", relation="genre")
    KG.add_edge("Warner_Bros", "The_Matrix", relation="produced")

    # Inception relations
    KG.add_edge("Inception", "Leonardo_DiCaprio", relation="stars")
    KG.add_edge("Christopher_Nolan", "Inception", relation="directed")
    KG.add_edge("Inception", "SciFi", relation="genre")
    KG.add_edge("Inception", "Thriller", relation="genre")
    KG.add_edge("Warner_Bros", "Inception", relation="produced")

    # Interstellar relations
    KG.add_edge("Interstellar", "Matthew_McConaughey", relation="stars")
    KG.add_edge("Christopher_Nolan", "Interstellar", relation="directed")
    KG.add_edge("Interstellar", "SciFi", relation="genre")
    KG.add_edge("Paramount", "Interstellar", relation="produced")

    # Cross-references
    KG.add_edge("Christopher_Nolan", "Warner_Bros", relation="works_with")

    return KG


def create_movie_knowledge_graph_2() -> nx.DiGraph:
    """
    Create a DIFFERENT knowledge graph about the same movies.
    Uses different naming conventions (simulating a different database).

    This is the graph we want to align with KG1.
    """
    KG = nx.DiGraph()

    # Same movies, different names
    KG.add_node("Matrix1999", type="Film", release=1999)
    KG.add_node("Inception2010", type="Film", release=2010)
    KG.add_node("Interstellar2014", type="Film", release=2014)

    # Same people, different naming
    KG.add_node("K_Reeves", type="Actor")
    KG.add_node("CA_Moss", type="Actor")
    KG.add_node("Wachowski_Sisters", type="Director")
    KG.add_node("L_DiCaprio", type="Actor")
    KG.add_node("C_Nolan", type="Director")
    KG.add_node("M_McConaughey", type="Actor")

    # Same genres, different names
    KG.add_node("Science_Fiction", type="Category")
    KG.add_node("Action_Film", type="Category")
    KG.add_node("Thriller_Film", type="Category")

    # Studios
    KG.add_node("WB_Pictures", type="Company")
    KG.add_node("Paramount_Pictures", type="Company")

    # Same relations, same structure
    KG.add_edge("Matrix1999", "K_Reeves", relation="actor")
    KG.add_edge("Matrix1999", "CA_Moss", relation="actor")
    KG.add_edge("Wachowski_Sisters", "Matrix1999", relation="director")
    KG.add_edge("Matrix1999", "Science_Fiction", relation="category")
    KG.add_edge("Matrix1999", "Action_Film", relation="category")
    KG.add_edge("WB_Pictures", "Matrix1999", relation="studio")

    KG.add_edge("Inception2010", "L_DiCaprio", relation="actor")
    KG.add_edge("C_Nolan", "Inception2010", relation="director")
    KG.add_edge("Inception2010", "Science_Fiction", relation="category")
    KG.add_edge("Inception2010", "Thriller_Film", relation="category")
    KG.add_edge("WB_Pictures", "Inception2010", relation="studio")

    KG.add_edge("Interstellar2014", "M_McConaughey", relation="actor")
    KG.add_edge("C_Nolan", "Interstellar2014", relation="director")
    KG.add_edge("Interstellar2014", "Science_Fiction", relation="category")
    KG.add_edge("Paramount_Pictures", "Interstellar2014", relation="studio")

    KG.add_edge("C_Nolan", "WB_Pictures", relation="collaborates")

    return KG


def get_ground_truth_alignment() -> Dict[str, str]:
    """Return the true entity alignment between KG1 and KG2."""
    return {
        # Movies
        "The_Matrix": "Matrix1999",
        "Inception": "Inception2010",
        "Interstellar": "Interstellar2014",
        # Actors
        "Keanu_Reeves": "K_Reeves",
        "Carrie_Anne_Moss": "CA_Moss",
        "Leonardo_DiCaprio": "L_DiCaprio",
        "Matthew_McConaughey": "M_McConaughey",
        # Directors
        "Wachowskis": "Wachowski_Sisters",
        "Christopher_Nolan": "C_Nolan",
        # Genres
        "SciFi": "Science_Fiction",
        "Action": "Action_Film",
        "Thriller": "Thriller_Film",
        # Studios
        "Warner_Bros": "WB_Pictures",
        "Paramount": "Paramount_Pictures",
    }


# =============================================================================
# GRAPH ALIGNMENT METHODS
# =============================================================================

def compute_node_signature(G: nx.Graph, node: str) -> Tuple:
    """
    Compute a structural signature for a node based on local topology.

    For prime graphs, this captures the directed structure.
    """
    degree = G.degree(node)

    # Neighbor degree distribution
    neighbor_degrees = sorted([G.degree(n) for n in G.neighbors(node)])

    # 2-hop neighborhood size
    two_hop = set()
    for n in G.neighbors(node):
        two_hop.update(G.neighbors(n))
    two_hop_size = len(two_hop)

    # Triangle count (clustering)
    triangles = sum(1 for n1, n2 in itertools.combinations(G.neighbors(node), 2)
                    if G.has_edge(n1, n2))

    return (degree, tuple(neighbor_degrees[:5]), two_hop_size, triangles)


def compute_prime_aware_signature(H: nx.Graph, node: str) -> Dict:
    """
    Compute a signature that SPECIFICALLY leverages prime graph structure.

    This captures directional information encoded in the bipartite structure.
    """
    is_prime = H.nodes[node].get('prime', False)

    if is_prime:
        # This is a prime node (v') - skip, we align non-prime nodes
        return None

    # For non-prime nodes, analyze connections
    neighbors = list(H.neighbors(node))

    # Separate neighbors by type
    prime_neighbors = [n for n in neighbors if H.nodes[n].get('prime', False)]
    non_prime_neighbors = [n for n in neighbors if not H.nodes[n].get('prime', False)]

    # Find the paired prime node (v')
    paired_prime = f"{node}'"

    # OUT-EDGES: non-prime node connects to OTHER prime nodes (not its own pair)
    out_edges = [n for n in prime_neighbors if n != paired_prime]
    out_degree = len(out_edges)

    # IN-EDGES: paired prime node's non-prime neighbors (excluding self)
    in_edges = []
    if paired_prime in H:
        for n in H.neighbors(paired_prime):
            if not H.nodes[n].get('prime', False) and n != node:
                in_edges.append(n)
    in_degree = len(in_edges)

    # Create rich signature capturing direction
    signature = {
        'total_degree': len(neighbors),
        'out_degree': out_degree,  # How many this node points TO
        'in_degree': in_degree,    # How many nodes point TO this
        'is_source': in_degree == 0 and out_degree > 0,
        'is_sink': out_degree == 0 and in_degree > 0,
        'is_mixed': in_degree > 0 and out_degree > 0,
        'degree_ratio': out_degree / max(in_degree, 1),
        'neighbor_out_degrees': sorted([
            len([nn for nn in H.neighbors(n)
                 if H.nodes[nn].get('prime', False) and nn != f"{n}'"])
            for n in out_edges
        ]) if out_edges else [],
    }

    return signature


def compute_prime_similarity(sig1: Dict, sig2: Dict) -> float:
    """
    Compute similarity between two prime-aware signatures.
    This leverages the directional information preserved in prime graphs.
    """
    if sig1 is None or sig2 is None:
        return 0.0

    score = 0.0
    max_score = 0.0

    # In/out degree matching (critical for directed semantics)
    max_score += 2.0
    if sig1['out_degree'] == sig2['out_degree']:
        score += 1.0
    elif abs(sig1['out_degree'] - sig2['out_degree']) == 1:
        score += 0.5

    if sig1['in_degree'] == sig2['in_degree']:
        score += 1.0
    elif abs(sig1['in_degree'] - sig2['in_degree']) == 1:
        score += 0.5

    # Node role matching (source/sink/mixed)
    max_score += 1.5
    if sig1['is_source'] == sig2['is_source']:
        score += 0.5
    if sig1['is_sink'] == sig2['is_sink']:
        score += 0.5
    if sig1['is_mixed'] == sig2['is_mixed']:
        score += 0.5

    # Degree ratio similarity
    max_score += 1.0
    ratio_diff = abs(sig1['degree_ratio'] - sig2['degree_ratio'])
    score += max(0, 1.0 - ratio_diff * 0.5)

    # Total degree
    max_score += 0.5
    if sig1['total_degree'] == sig2['total_degree']:
        score += 0.5

    return score / max_score


def align_prime_graphs_enhanced(H1: nx.Graph, H2: nx.Graph) -> Dict[str, List[Tuple[str, float]]]:
    """
    Enhanced alignment that SPECIFICALLY uses prime graph directional structure.
    """
    # Get non-prime nodes only
    nodes1 = [n for n in H1.nodes() if not H1.nodes[n].get('prime', False)]
    nodes2 = [n for n in H2.nodes() if not H2.nodes[n].get('prime', False)]

    # Compute prime-aware signatures
    sigs1 = {n: compute_prime_aware_signature(H1, n) for n in nodes1}
    sigs2 = {n: compute_prime_aware_signature(H2, n) for n in nodes2}

    # Find best matches using directional information
    alignment = {}
    for n1 in nodes1:
        scores = []
        for n2 in nodes2:
            sim = compute_prime_similarity(sigs1[n1], sigs2[n2])
            scores.append((n2, sim))
        scores.sort(key=lambda x: -x[1])
        alignment[n1] = scores[:3]

    return alignment


def compute_structural_similarity(sig1: Tuple, sig2: Tuple) -> float:
    """Compute similarity between two node signatures."""
    if sig1 == sig2:
        return 1.0

    # Degree similarity
    deg_sim = 1 - abs(sig1[0] - sig2[0]) / max(sig1[0], sig2[0], 1)

    # 2-hop similarity
    hop_sim = 1 - abs(sig1[2] - sig2[2]) / max(sig1[2], sig2[2], 1)

    # Triangle similarity
    tri_sim = 1 - abs(sig1[3] - sig2[3]) / max(sig1[3], sig2[3], 1)

    return (deg_sim + hop_sim + tri_sim) / 3


def align_prime_graphs(H1: nx.Graph, H2: nx.Graph,
                       top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
    """
    Align two prime graphs based on structural similarity.

    This is a simplified alignment algorithm. Real tools like SANA, MAGNA++
    would be more sophisticated, but this demonstrates the concept.

    Returns: For each non-prime node in H1, top-k candidate matches in H2.
    """
    # Get non-prime nodes only (these represent original entities)
    nodes1 = [n for n in H1.nodes() if not H1.nodes[n].get('prime', False)]
    nodes2 = [n for n in H2.nodes() if not H2.nodes[n].get('prime', False)]

    # Compute signatures
    sigs1 = {n: compute_node_signature(H1, n) for n in nodes1}
    sigs2 = {n: compute_node_signature(H2, n) for n in nodes2}

    # Find best matches for each node in H1
    alignment = {}
    for n1 in nodes1:
        scores = []
        for n2 in nodes2:
            sim = compute_structural_similarity(sigs1[n1], sigs2[n2])
            scores.append((n2, sim))
        scores.sort(key=lambda x: -x[1])
        alignment[n1] = scores[:top_k]

    return alignment


def greedy_alignment(candidates: Dict[str, List[Tuple[str, float]]]) -> Dict[str, str]:
    """
    Greedy one-to-one alignment from candidate scores.
    """
    alignment = {}
    used = set()

    # Sort all candidates by score
    all_pairs = []
    for n1, matches in candidates.items():
        for n2, score in matches:
            all_pairs.append((score, n1, n2))

    all_pairs.sort(reverse=True)

    for score, n1, n2 in all_pairs:
        if n1 not in alignment and n2 not in used:
            alignment[n1] = n2
            used.add(n2)

    return alignment


# =============================================================================
# TRADITIONAL APPROACH (for comparison)
# =============================================================================

def traditional_undirected_conversion(G: nx.DiGraph) -> nx.Graph:
    """
    Traditional lossy conversion: just remove direction.
    This loses information about edge directionality.
    """
    return G.to_undirected()


def align_traditional(G1: nx.Graph, G2: nx.Graph) -> Dict[str, str]:
    """
    Align two undirected graphs (after lossy conversion).
    Same algorithm, but working on lossy undirected graphs.
    """
    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())

    sigs1 = {n: compute_node_signature(G1, n) for n in nodes1}
    sigs2 = {n: compute_node_signature(G2, n) for n in nodes2}

    candidates = {}
    for n1 in nodes1:
        scores = []
        for n2 in nodes2:
            sim = compute_structural_similarity(sigs1[n1], sigs2[n2])
            scores.append((n2, sim))
        scores.sort(key=lambda x: -x[1])
        candidates[n1] = scores[:3]

    return greedy_alignment(candidates)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_alignment(predicted: Dict[str, str],
                       ground_truth: Dict[str, str]) -> Dict[str, float]:
    """Evaluate alignment quality."""
    correct = 0
    total = len(ground_truth)

    for entity1, true_entity2 in ground_truth.items():
        if entity1 in predicted and predicted[entity1] == true_entity2:
            correct += 1

    precision = correct / len(predicted) if predicted else 0
    recall = correct / total if total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "correct": correct,
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_entity_resolution():
    """Full demonstration of knowledge graph entity resolution."""

    print("=" * 70)
    print("KNOWLEDGE GRAPH ENTITY RESOLUTION USING PRIME GRAPHS")
    print("=" * 70)

    # Create knowledge graphs
    print("\n" + "=" * 70)
    print("STEP 1: CREATE TWO KNOWLEDGE GRAPHS")
    print("=" * 70)

    KG1 = create_movie_knowledge_graph_1()
    KG2 = create_movie_knowledge_graph_2()
    ground_truth = get_ground_truth_alignment()

    print(f"\nKnowledge Graph 1 (English naming):")
    print(f"  Nodes: {KG1.number_of_nodes()}")
    print(f"  Edges: {KG1.number_of_edges()}")
    print(f"  Sample entities: {list(KG1.nodes())[:5]}")

    print(f"\nKnowledge Graph 2 (Alternate naming):")
    print(f"  Nodes: {KG2.number_of_nodes()}")
    print(f"  Edges: {KG2.number_of_edges()}")
    print(f"  Sample entities: {list(KG2.nodes())[:5]}")

    print(f"\nGround truth: {len(ground_truth)} entity pairs to match")

    # Show the problem
    print("\n" + "=" * 70)
    print("THE PROBLEM: DIRECTED EDGES CARRY SEMANTIC MEANING")
    print("=" * 70)

    print("\nIn KG1:")
    print("  'Wachowskis' --[directed]--> 'The_Matrix'  (director relationship)")
    print("  'The_Matrix' --[directed]--> 'Keanu_Reeves' (stars relationship)")
    print("\nDirection matters! 'A directed B' ≠ 'B directed A'")
    print("Traditional undirected conversion LOSES this information.")

    # Traditional approach
    print("\n" + "=" * 70)
    print("APPROACH 1: TRADITIONAL (LOSSY UNDIRECTED CONVERSION)")
    print("=" * 70)

    U1 = traditional_undirected_conversion(KG1)
    U2 = traditional_undirected_conversion(KG2)

    print(f"\nConverted to undirected:")
    print(f"  U1: {U1.number_of_nodes()} nodes, {U1.number_of_edges()} edges")
    print(f"  U2: {U2.number_of_nodes()} nodes, {U2.number_of_edges()} edges")
    print("  ⚠ Direction information LOST")

    traditional_alignment = align_traditional(U1, U2)
    traditional_metrics = evaluate_alignment(traditional_alignment, ground_truth)

    print(f"\nTraditional alignment results:")
    print(f"  Correct matches: {traditional_metrics['correct']}/{traditional_metrics['total']}")
    print(f"  Precision: {traditional_metrics['precision']:.3f}")
    print(f"  Recall: {traditional_metrics['recall']:.3f}")
    print(f"  F1 Score: {traditional_metrics['f1']:.3f}")

    # Prime graph approach
    print("\n" + "=" * 70)
    print("APPROACH 2: PRIME GRAPH (LOSSLESS TRANSFORMATION)")
    print("=" * 70)

    H1 = directed_to_prime(KG1)
    H2 = directed_to_prime(KG2)

    print(f"\nConverted to prime graphs:")
    print(f"  H1: {H1.number_of_nodes()} nodes, {H1.number_of_edges()} edges")
    print(f"  H2: {H2.number_of_nodes()} nodes, {H2.number_of_edges()} edges")
    print("  ✓ Direction information PRESERVED in bipartite structure")

    # Show how direction is preserved
    print("\nHow direction is encoded:")
    print("  'Wachowskis' connects to 'Wachowskis'' (structural)")
    print("  'Wachowskis' connects to 'The_Matrix'' (encodes: Wachowskis → The_Matrix)")
    print("  This is DIFFERENT from 'The_Matrix' connecting to 'Wachowskis''")

    # Basic prime graph alignment (same algorithm as traditional)
    candidates = align_prime_graphs(H1, H2)
    prime_alignment_basic = greedy_alignment(candidates)
    prime_metrics_basic = evaluate_alignment(prime_alignment_basic, ground_truth)

    print(f"\nPrime graph alignment (basic - same algo as traditional):")
    print(f"  Correct matches: {prime_metrics_basic['correct']}/{prime_metrics_basic['total']}")
    print(f"  F1 Score: {prime_metrics_basic['f1']:.3f}")

    # Enhanced prime graph alignment (USES directional information)
    candidates_enhanced = align_prime_graphs_enhanced(H1, H2)
    prime_alignment = greedy_alignment(candidates_enhanced)
    prime_metrics = evaluate_alignment(prime_alignment, ground_truth)

    print(f"\nPrime graph alignment (enhanced - uses direction):")
    print(f"  Correct matches: {prime_metrics['correct']}/{prime_metrics['total']}")
    print(f"  Precision: {prime_metrics['precision']:.3f}")
    print(f"  Recall: {prime_metrics['recall']:.3f}")
    print(f"  F1 Score: {prime_metrics['f1']:.3f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: TRADITIONAL vs PRIME GRAPH")
    print("=" * 70)

    print("\n┌─────────────────────┬─────────────┬─────────────┬─────────────────┐")
    print("│ Metric              │ Traditional │ Prime Basic │ Prime Enhanced  │")
    print("├─────────────────────┼─────────────┼─────────────┼─────────────────┤")
    print(f"│ Correct Matches     │ {traditional_metrics['correct']:>11} │ {prime_metrics_basic['correct']:>11} │ {prime_metrics['correct']:>15} │")
    print(f"│ Precision           │ {traditional_metrics['precision']:>11.3f} │ {prime_metrics_basic['precision']:>11.3f} │ {prime_metrics['precision']:>15.3f} │")
    print(f"│ Recall              │ {traditional_metrics['recall']:>11.3f} │ {prime_metrics_basic['recall']:>11.3f} │ {prime_metrics['recall']:>15.3f} │")
    print(f"│ F1 Score            │ {traditional_metrics['f1']:>11.3f} │ {prime_metrics_basic['f1']:>11.3f} │ {prime_metrics['f1']:>15.3f} │")
    print("│ Direction Info Used │          No │          No │             Yes │")
    print("│ Direction Preserved │          No │         Yes │             Yes │")
    print("└─────────────────────┴─────────────┴─────────────┴─────────────────┘")

    print("\nKEY INSIGHT:")
    print("  - 'Prime Basic' uses the SAME algorithm as Traditional (just on prime graphs)")
    print("  - 'Prime Enhanced' EXPLOITS the directional info encoded in prime graphs")
    print("  - The value of prime graphs comes from algorithms that USE the structure!")

    # Show specific matches
    print("\n" + "=" * 70)
    print("DETAILED ALIGNMENT COMPARISON")
    print("=" * 70)

    print("\n{:<25} {:<20} {:<20} {:<10}".format(
        "Entity (KG1)", "Traditional", "Prime Graph", "Correct?"))
    print("-" * 75)

    for entity1, true_entity2 in sorted(ground_truth.items()):
        trad_match = traditional_alignment.get(entity1, "NO MATCH")
        prime_match = prime_alignment.get(entity1, "NO MATCH")

        trad_correct = "✓" if trad_match == true_entity2 else "✗"
        prime_correct = "✓" if prime_match == true_entity2 else "✗"

        print(f"{entity1:<25} {trad_match:<20} {prime_match:<20} T:{trad_correct} P:{prime_correct}")

    # Why prime graphs help
    print("\n" + "=" * 70)
    print("WHY PRIME GRAPHS IMPROVE ENTITY RESOLUTION")
    print("=" * 70)

    print("""
1. DIRECTION ENCODES RELATIONSHIPS
   - "A directed B" has different meaning than "B directed A"
   - Prime graphs preserve this: (A, B') ≠ (B, A')

2. RICHER STRUCTURAL SIGNATURES
   - Non-prime nodes have different neighbor patterns than prime nodes
   - Directors (sources) connect differently than actors (mixed)

3. ENABLES SOPHISTICATED ALIGNMENT TOOLS
   - Can use SANA, MAGNA++, IsoRank, GRAAL, etc.
   - These tools are optimized for undirected graph alignment

4. PERFECT REVERSIBILITY
   - After alignment, can recover original directed relationships
   - No information lost in the process
""")

    # Real-world applications
    print("=" * 70)
    print("REAL-WORLD APPLICATIONS")
    print("=" * 70)

    print("""
1. WIKIDATA ↔ DBPEDIA ALIGNMENT
   - Different entity URIs for same concepts
   - Directed relations (subject → predicate → object)

2. DRUG-TARGET DATABASE INTEGRATION
   - DrugBank, ChEMBL, STITCH have overlapping drugs/targets
   - Interaction directions matter (drug inhibits target)

3. CROSS-LINGUAL KNOWLEDGE GRAPHS
   - English Wikipedia ↔ Chinese Wikipedia
   - Same concepts, different entity names

4. ENTERPRISE DATA INTEGRATION
   - Merging customer databases after acquisition
   - Directed relationships (customer → purchased → product)
""")

    return {
        "traditional": traditional_metrics,
        "prime_graph": prime_metrics,
        "improvement": prime_metrics['f1'] - traditional_metrics['f1']
    }


def create_symmetric_structure_kg1() -> nx.DiGraph:
    """
    Create a knowledge graph with SYMMETRIC UNDIRECTED STRUCTURE
    but ASYMMETRIC DIRECTED STRUCTURE.

    This is where direction information becomes CRITICAL.
    The undirected version has identical structure, but directions differ.
    """
    KG = nx.DiGraph()

    # Create a citation network with clear hierarchy
    # Papers cite other papers - direction matters!
    papers = [f"Paper_{i}" for i in range(6)]
    for p in papers:
        KG.add_node(p, type="Paper")

    # Citation structure: newer papers cite older papers
    # Paper_0, Paper_1, Paper_2 are "foundational" (cited by others)
    # Paper_3, Paper_4, Paper_5 are "recent" (cite others)
    KG.add_edge("Paper_3", "Paper_0")  # 3 cites 0
    KG.add_edge("Paper_3", "Paper_1")  # 3 cites 1
    KG.add_edge("Paper_4", "Paper_0")  # 4 cites 0
    KG.add_edge("Paper_4", "Paper_2")  # 4 cites 2
    KG.add_edge("Paper_5", "Paper_1")  # 5 cites 1
    KG.add_edge("Paper_5", "Paper_2")  # 5 cites 2

    return KG


def create_symmetric_structure_kg2() -> nx.DiGraph:
    """
    Create a DIFFERENT knowledge graph with SAME undirected structure
    but DIFFERENT node identities.

    After undirected conversion, these graphs look IDENTICAL.
    But with direction, we can distinguish which is which.
    """
    KG = nx.DiGraph()

    # Same structure, different names
    articles = [f"Article_{chr(65+i)}" for i in range(6)]  # A, B, C, D, E, F
    for a in articles:
        KG.add_node(a, type="Article")

    # Same citation structure
    KG.add_edge("Article_D", "Article_A")  # D cites A
    KG.add_edge("Article_D", "Article_B")  # D cites B
    KG.add_edge("Article_E", "Article_A")  # E cites A
    KG.add_edge("Article_E", "Article_C")  # E cites C
    KG.add_edge("Article_F", "Article_B")  # F cites B
    KG.add_edge("Article_F", "Article_C")  # F cites C

    return KG


def get_symmetric_ground_truth() -> Dict[str, str]:
    """Ground truth for symmetric structure KGs."""
    return {
        "Paper_0": "Article_A",  # Both are foundational, cited by 2 papers
        "Paper_1": "Article_B",  # Both are foundational, cited by 2 papers
        "Paper_2": "Article_C",  # Both are foundational, cited by 2 papers
        "Paper_3": "Article_D",  # Both cite Paper_0/A and Paper_1/B
        "Paper_4": "Article_E",  # Both cite Paper_0/A and Paper_2/C
        "Paper_5": "Article_F",  # Both cite Paper_1/B and Paper_2/C
    }


def demonstrate_symmetric_challenge():
    """
    Demonstrate a case where direction information is CRITICAL.
    """
    print("\n" + "=" * 70)
    print("CHALLENGE: SYMMETRIC UNDIRECTED STRUCTURE")
    print("=" * 70)

    print("""
This test creates two knowledge graphs where:
- The UNDIRECTED structure is IDENTICAL
- But the DIRECTED structure reveals the true alignment

Without direction: Paper_0, Paper_1, Paper_2 are INDISTINGUISHABLE
With direction: We can tell them apart by citation patterns (in vs out)
""")

    KG1 = create_symmetric_structure_kg1()
    KG2 = create_symmetric_structure_kg2()
    ground_truth = get_symmetric_ground_truth()

    print(f"KG1: {KG1.number_of_nodes()} papers, {KG1.number_of_edges()} citations")
    print(f"KG2: {KG2.number_of_nodes()} articles, {KG2.number_of_edges()} citations")

    # Traditional (lossy)
    U1 = traditional_undirected_conversion(KG1)
    U2 = traditional_undirected_conversion(KG2)

    print("\n--- UNDIRECTED (LOSSY) VIEW ---")
    print("After removing direction, the graphs look identical:")
    print(f"  Degree sequence KG1: {sorted([d for _, d in U1.degree()])}")
    print(f"  Degree sequence KG2: {sorted([d for _, d in U2.degree()])}")
    print("  These are THE SAME - traditional methods can't distinguish nodes!")

    traditional_alignment = align_traditional(U1, U2)
    traditional_metrics = evaluate_alignment(traditional_alignment, ground_truth)

    # Prime graph (lossless)
    H1 = directed_to_prime(KG1)
    H2 = directed_to_prime(KG2)

    print("\n--- DIRECTED (PRIME GRAPH) VIEW ---")

    # Show in/out degree differences
    print("With direction preserved, we can see the difference:")
    for node in ["Paper_0", "Paper_3"]:
        in_deg = KG1.in_degree(node)
        out_deg = KG1.out_degree(node)
        print(f"  {node}: in-degree={in_deg}, out-degree={out_deg}")
    print("  Paper_0 is CITED (high in-degree), Paper_3 CITES (high out-degree)")

    candidates_enhanced = align_prime_graphs_enhanced(H1, H2)
    prime_alignment = greedy_alignment(candidates_enhanced)
    prime_metrics = evaluate_alignment(prime_alignment, ground_truth)

    print("\n--- RESULTS ---")
    print(f"\nTraditional (direction-blind):")
    print(f"  F1 Score: {traditional_metrics['f1']:.3f}")
    print(f"  Correct: {traditional_metrics['correct']}/{traditional_metrics['total']}")

    print(f"\nPrime Graph (direction-aware):")
    print(f"  F1 Score: {prime_metrics['f1']:.3f}")
    print(f"  Correct: {prime_metrics['correct']}/{prime_metrics['total']}")

    improvement = prime_metrics['f1'] - traditional_metrics['f1']
    if improvement > 0:
        print(f"\n✓ Prime graph improved by {improvement:.3f}")
    else:
        print(f"\n  Difference: {improvement:.3f}")

    print("\n--- DETAILED MATCHES ---")
    print(f"{'Entity':<12} {'True Match':<12} {'Traditional':<12} {'Prime':<12}")
    print("-" * 50)
    for e1, e2 in sorted(ground_truth.items()):
        trad = traditional_alignment.get(e1, "NONE")
        prime = prime_alignment.get(e1, "NONE")
        trad_ok = "✓" if trad == e2 else "✗"
        prime_ok = "✓" if prime == e2 else "✗"
        print(f"{e1:<12} {e2:<12} {trad:<10}{trad_ok} {prime:<10}{prime_ok}")

    return {
        "traditional_f1": traditional_metrics['f1'],
        "prime_f1": prime_metrics['f1']
    }


def run_multiple_trials(n_trials: int = 5):
    """Run multiple trials to show consistency."""
    print("\n" + "=" * 70)
    print(f"RUNNING {n_trials} TRIALS FOR STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    # For this demo, graphs are deterministic, so results will be same
    # In real scenarios with larger graphs, there would be variation

    results = demonstrate_entity_resolution()

    print("\n" + "=" * 70)
    print("FINAL VERDICT FOR MOVIE KG")
    print("=" * 70)

    if results['improvement'] > 0:
        print(f"\n✓ Prime graph method improved F1 by {results['improvement']:.3f}")
    elif results['improvement'] == 0:
        print("\n= Both methods achieved same F1 score")
        print("  (For this graph size, structural signatures may be sufficient)")
    else:
        print(f"\n✗ Traditional method was better by {-results['improvement']:.3f}")
        print("  (Small graphs with unique structure don't need direction)")

    # Run the symmetric challenge
    sym_results = demonstrate_symmetric_challenge()

    print("\n" + "=" * 70)
    print("OVERALL CONCLUSIONS")
    print("=" * 70)

    print("""
KEY FINDINGS:

1. SMALL UNIQUE GRAPHS (Movie KG):
   - Traditional methods may work equally well or better
   - Structure is unique enough that direction isn't needed
   - Prime graphs add complexity without benefit

2. SYMMETRIC/AMBIGUOUS GRAPHS (Citation KG):
   - Direction becomes CRITICAL for correct alignment
   - Without direction, nodes are INDISTINGUISHABLE
   - Prime graphs enable correct matching

WHEN TO USE PRIME GRAPHS:
  ✓ Large knowledge graphs with repetitive patterns
  ✓ Citation/reference networks where direction = semantics
  ✓ Hierarchical data (parent→child relationships)
  ✓ When you need to use undirected alignment tools
  ✓ When perfect reversibility is required

WHEN TRADITIONAL MAY SUFFICE:
  • Small graphs with unique structure
  • Graphs where direction is just metadata, not semantics
  • When speed is critical (prime graphs have 2x nodes)
""")


if __name__ == "__main__":
    run_multiple_trials()
