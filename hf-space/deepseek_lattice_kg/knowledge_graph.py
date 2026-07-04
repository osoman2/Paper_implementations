"""
AI Engineering Knowledge Graph — simulates the Neo4j graph from the paper.

Nodes are AI/RAG concepts; edges carry typed relationships
(uses, improves, extends, part_of, competes_with).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Node:
    name: str
    category: str          # e.g. "retrieval", "evaluation", "model"
    description: str
    properties: dict = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    rel: str               # relationship type
    dst: str
    weight: float = 1.0
    note: str = ""


# ---------------------------------------------------------------------------
# Knowledge graph definition
# ---------------------------------------------------------------------------

NODES: list[Node] = [
    Node("RAG", "pipeline", "Retrieval-Augmented Generation — augments LLM generation with retrieved context"),
    Node("BM25", "retrieval", "Sparse probabilistic retrieval model based on term frequency and inverse document frequency"),
    Node("Dense Retrieval", "retrieval", "Encodes query and documents into embeddings; retrieves by cosine similarity"),
    Node("Hybrid Search", "retrieval", "Fuses sparse (BM25) and dense retrieval scores; improves recall+precision"),
    Node("RRF", "fusion", "Reciprocal Rank Fusion — rank-based score merging for multiple retrieval results"),
    Node("Reranking", "retrieval", "Cross-encoder re-scores top-k candidates; improves precision at the cost of latency"),
    Node("Knowledge Graph", "storage", "Graph of entities and typed relationships; enables relationship-aware retrieval"),
    Node("KG-RAG", "pipeline", "RAG augmented with knowledge graph retrieval; preserves entity relationships and provenance"),
    Node("Cypher", "query", "Graph query language for Neo4j; matches patterns of nodes and edges"),
    Node("Neo4j", "storage", "Graph database; stores and queries knowledge graphs via Cypher"),
    Node("Fine-tuning", "training", "Continues pre-training on domain data to specialise LLM weights"),
    Node("PEFT", "training", "Parameter-Efficient Fine-Tuning — freezes most weights, trains small adapters"),
    Node("LoRA", "training", "Low-Rank Adaptation — PEFT variant that inserts low-rank weight matrices"),
    Node("LLM", "model", "Large Language Model — pretrained on broad text, generates natural language"),
    Node("Vector Database", "storage", "Stores and indexes dense embeddings for approximate nearest-neighbour lookup"),
    Node("Embedding Model", "model", "Encodes text into dense vectors capturing semantic meaning"),
    Node("Chunking", "preprocessing", "Splits long documents into retrievable units; chunk size affects recall"),
    Node("Entity Extraction", "preprocessing", "Identifies named entities in text; key step for KG population"),
    Node("Context Window", "model", "Maximum tokens an LLM can process; limits how much retrieved context fits"),
    Node("Provenance", "evaluation", "Tracking which source each claim comes from; crucial for RAG citation quality"),
    Node("Hallucination", "evaluation", "LLM generates plausible-sounding but factually wrong content"),
    Node("CAD Score", "evaluation", "Context-Aware Diversity metric; evaluates LLM output semantic diversity"),
    Node("IDF", "retrieval", "Inverse Document Frequency — measures term rarity; used for alpha in hybrid search"),
    Node("Alpha", "retrieval", "Weight balancing sparse vs dense score in hybrid search; IDF-adaptive is key insight"),
    Node("Subgraph", "retrieval", "Portion of the knowledge graph relevant to a query; retrieved via Cypher pattern match"),
]

EDGES: list[Edge] = [
    Edge("RAG", "uses", "BM25", note="sparse retrieval leg"),
    Edge("RAG", "uses", "Dense Retrieval", note="dense retrieval leg"),
    Edge("RAG", "uses", "Chunking", note="preprocessing step"),
    Edge("RAG", "uses", "Reranking", note="post-retrieval refinement"),
    Edge("RAG", "suffers_from", "Hallucination", note="LLM ignores retrieved context"),
    Edge("RAG", "needs", "Provenance", note="attribution for retrieved facts"),

    Edge("KG-RAG", "extends", "RAG", note="adds structured graph retrieval"),
    Edge("KG-RAG", "uses", "Knowledge Graph", note="entity-relationship store"),
    Edge("KG-RAG", "uses", "Cypher", note="NL → Cypher query generation"),
    Edge("KG-RAG", "uses", "Subgraph", note="retrieves relevant graph neighborhood"),
    Edge("KG-RAG", "improves", "Provenance", note="graph edges encode source relationships"),
    Edge("KG-RAG", "reduces", "Hallucination", note="structured context is harder to ignore"),

    Edge("Hybrid Search", "uses", "BM25"),
    Edge("Hybrid Search", "uses", "Dense Retrieval"),
    Edge("Hybrid Search", "uses", "RRF", note="default fusion method"),
    Edge("Hybrid Search", "uses", "Alpha", note="balance sparse/dense contribution"),
    Edge("Hybrid Search", "improves", "RAG"),

    Edge("Alpha", "computed_from", "IDF", note="IDF-adaptive alpha = key insight from DAT paper"),

    Edge("Dense Retrieval", "uses", "Embedding Model"),
    Edge("Dense Retrieval", "uses", "Vector Database"),

    Edge("Fine-tuning", "specialises", "LLM"),
    Edge("PEFT", "is_a", "Fine-tuning"),
    Edge("LoRA", "is_a", "PEFT"),

    Edge("Cypher", "queries", "Neo4j"),
    Edge("Neo4j", "stores", "Knowledge Graph"),
    Edge("Knowledge Graph", "built_from", "Entity Extraction"),

    Edge("Subgraph", "retrieved_from", "Knowledge Graph"),

    Edge("Reranking", "uses", "Embedding Model"),
    Edge("CAD Score", "evaluates", "LLM"),

    Edge("LLM", "limited_by", "Context Window"),
    Edge("LLM", "generates", "Hallucination"),

    Edge("Chunking", "affects", "Context Window", note="chunk size vs context budget tradeoff"),
]


def build_graph() -> tuple[dict[str, Node], list[Edge]]:
    node_map = {n.name: n for n in NODES}
    return node_map, EDGES


def get_neighbors(node_name: str, edges: list[Edge], hops: int = 1) -> set[str]:
    """Return names of all nodes reachable within `hops` from node_name."""
    frontier = {node_name}
    visited = {node_name}
    for _ in range(hops):
        next_frontier: set[str] = set()
        for e in edges:
            if e.src in frontier and e.dst not in visited:
                next_frontier.add(e.dst)
            if e.dst in frontier and e.src not in visited:
                next_frontier.add(e.src)
        visited.update(next_frontier)
        frontier = next_frontier
    return visited - {node_name}


def cypher_query(entity_name: str) -> str:
    """Return the simulated Cypher query for an entity."""
    return (
        f"MATCH (n {{name: '{entity_name}'}})-[r]-(m)\n"
        f"RETURN n, type(r) AS rel, m\n"
        f"LIMIT 25"
    )


def retrieve_subgraph(
    entity_names: list[str],
    edges: list[Edge],
    hops: int = 1,
) -> tuple[set[str], list[Edge]]:
    """
    Returns (all_node_names, relevant_edges) for the subgraph around the
    given entities within `hops` steps.
    """
    all_nodes: set[str] = set(entity_names)
    for name in entity_names:
        all_nodes.update(get_neighbors(name, edges, hops))

    sub_edges = [
        e for e in edges
        if e.src in all_nodes and e.dst in all_nodes
    ]
    return all_nodes, sub_edges


def format_subgraph_context(
    node_names: set[str],
    sub_edges: list[Edge],
    node_map: dict[str, Node],
) -> str:
    """Format the retrieved subgraph as structured text context (like the paper injects into the LLM)."""
    lines = ["### Retrieved Knowledge Graph Context\n"]

    lines.append("**Entities:**")
    for name in sorted(node_names):
        if name in node_map:
            n = node_map[name]
            lines.append(f"- **{name}** ({n.category}): {n.description}")

    lines.append("\n**Relationships:**")
    for e in sub_edges:
        note = f" — _{e.note}_" if e.note else ""
        lines.append(f"- `{e.src}` --[{e.rel}]--> `{e.dst}`{note}")

    return "\n".join(lines)
