"""
KG-RAG query engine — mirrors the paper's pipeline:
  NL query → entity extraction → Cypher generation → subgraph retrieval → context injection

Also includes a flat BM25 baseline to show the contrast.
"""
from __future__ import annotations

import re
import math
from collections import Counter

from .knowledge_graph import (
    Node, Edge, build_graph,
    cypher_query, retrieve_subgraph, format_subgraph_context,
)


# ---------------------------------------------------------------------------
# Entity extractor — keyword matching against KG node names (simulates NER)
# ---------------------------------------------------------------------------

def extract_entities(query: str, node_map: dict[str, Node]) -> list[str]:
    """
    Find node names mentioned in the query (case-insensitive substring match).
    Returns matched node names sorted by match specificity (longer name first).
    """
    q = query.lower()
    matched = []
    for name, node in node_map.items():
        aliases = [name.lower()]
        # add common shorthand aliases
        if name == "Knowledge Graph":
            aliases += ["kg", "knowledge graph"]
        elif name == "KG-RAG":
            aliases += ["kg-rag", "graph rag", "graph-rag"]
        elif name == "BM25":
            aliases += ["bm25", "bm 25", "sparse"]
        elif name == "Dense Retrieval":
            aliases += ["dense", "semantic search", "embedding search"]
        elif name == "Hybrid Search":
            aliases += ["hybrid"]
        elif name == "Large Language Model" or name == "LLM":
            aliases += ["llm", "language model", "gpt", "deepseek", "claude"]
        elif name == "Fine-tuning":
            aliases += ["fine-tun", "finetuning", "fine tuning"]
        elif name == "Hallucination":
            aliases += ["hallucin", "hallucination", "confabulation"]
        elif name == "Provenance":
            aliases += ["attribution", "citation", "source"]
        elif name == "Vector Database":
            aliases += ["vector db", "vectordb", "qdrant", "pinecone", "weaviate", "faiss"]
        elif name == "Embedding Model":
            aliases += ["embedding", "encoder", "sentence-transformer"]
        elif name == "Retrieval-Augmented Generation" or name == "RAG":
            aliases += ["rag", "retrieval augmented", "retrieval-augmented"]
        elif name == "Reranking":
            aliases += ["rerank", "cross-encoder", "reranker"]
        elif name == "Chunking":
            aliases += ["chunk", "splitting", "segmentation"]

        if any(alias in q for alias in aliases):
            matched.append(name)

    # if nothing matched, fall back to the 2 most relevant general nodes
    if not matched:
        matched = ["RAG", "LLM"]

    # deduplicate and sort longest first (more specific match wins)
    return sorted(set(matched), key=len, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Flat BM25 baseline — retrieves from plain text documents
# ---------------------------------------------------------------------------

FLAT_DOCS = [
    ("BM25",
     "BM25 is a sparse retrieval model. It scores documents by term frequency and inverse document frequency. "
     "Used as the sparse leg in hybrid RAG pipelines."),
    ("Dense Retrieval",
     "Dense retrieval encodes queries and documents into vectors using an embedding model. "
     "Retrieval is by cosine similarity in a vector database. Semantic search."),
    ("Hybrid Search",
     "Hybrid search combines BM25 and dense retrieval. Uses RRF or linear interpolation to fuse scores. "
     "Improves recall and precision over either alone. Alpha controls the balance."),
    ("RAG",
     "Retrieval-Augmented Generation fetches relevant documents and injects them into the LLM context. "
     "Reduces hallucination by grounding the model in retrieved facts."),
    ("KG-RAG",
     "Knowledge Graph RAG retrieves structured entity-relationship subgraphs instead of raw text chunks. "
     "Uses Cypher queries to traverse the graph. Preserves provenance and entity relationships."),
    ("Fine-tuning",
     "Fine-tuning continues LLM training on domain data. PEFT methods like LoRA update only small adapter matrices. "
     "Improves accuracy on narrow tasks without full retraining."),
    ("Hallucination",
     "LLMs generate plausible but wrong content when lacking relevant context. "
     "RAG and KG-RAG reduce hallucination by injecting retrieved facts."),
    ("Reranking",
     "A cross-encoder reranker re-scores top-k candidates from first-stage retrieval. "
     "More accurate than bi-encoder but slower. Used for precision-critical queries."),
    ("Knowledge Graph",
     "A knowledge graph stores entities and typed relationships. Built via entity extraction from documents. "
     "Queried with Cypher in Neo4j. Enables relationship-aware retrieval."),
    ("Embedding Model",
     "An embedding model (e.g. sentence-transformers) encodes text into dense vectors. "
     "Used for semantic search, dense retrieval, and reranking."),
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _build_bm25_index(docs: list[tuple[str, str]]) -> dict:
    k1, b = 1.5, 0.75
    corpus = [_tokenize(text) for _, text in docs]
    N = len(corpus)
    avgdl = sum(len(d) for d in corpus) / N
    df: Counter = Counter()
    for d in corpus:
        df.update(set(d))
    idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}
    return {"corpus": corpus, "idf": idf, "k1": k1, "b": b, "avgdl": avgdl}


def bm25_search(query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    """Returns list of (title, text, score) sorted descending."""
    idx = _build_bm25_index(FLAT_DOCS)
    tokens = _tokenize(query)
    k1, b, avgdl = idx["k1"], idx["b"], idx["avgdl"]
    scores = []
    for i, (title, text) in enumerate(FLAT_DOCS):
        dl = len(idx["corpus"][i])
        score = sum(
            idx["idf"].get(t, 0) * (
                (freq := idx["corpus"][i].count(t)) * (k1 + 1)
                / (freq + k1 * (1 - b + b * dl / avgdl))
            )
            for t in set(tokens)
        )
        scores.append((title, text, score))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# KG-RAG engine
# ---------------------------------------------------------------------------

def kg_rag_search(
    query: str, hops: int = 1
) -> tuple[list[str], list[str], set[str], list[Edge], str]:
    """
    Full KG-RAG pipeline:
      1. Extract entities from query
      2. Generate Cypher queries (one per entity)
      3. Retrieve subgraph
      4. Format context

    Returns (entities, cypher_queries, subgraph_nodes, subgraph_edges, context_text)
    """
    node_map, edges = build_graph()

    entities = extract_entities(query, node_map)
    cyphers = [cypher_query(e) for e in entities]
    sub_nodes, sub_edges = retrieve_subgraph(entities, edges, hops=hops)
    context = format_subgraph_context(sub_nodes, sub_edges, node_map)

    return entities, cyphers, sub_nodes, sub_edges, context


def build_graph_data() -> tuple[dict[str, Node], list[Edge]]:
    return build_graph()
