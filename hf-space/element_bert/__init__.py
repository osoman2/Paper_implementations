"""
ElementBERT POC — Domain-specific element embeddings for materials inference.

Based on: "Semantic Embeddings of Chemical Elements for Enhanced Materials
Inference and Discovery" (Jia et al., 2026, Materials Genome Engineering Advances).

The key idea: train a BERT-style encoder on a corpus of alloy science abstracts
so it learns latent relationships between elements, phases, processing conditions,
and performance metrics — then use those embeddings as features for downstream
property prediction, phase classification, and Bayesian composition search.
"""

from .embeddings import (
    ELEMENTS,
    get_element_embedding,
    get_composition_embedding,
    element_similarity_matrix,
    pca_2d,
)
from .predictor import AlloyPredictor, EXAMPLE_ALLOYS

__all__ = [
    "ELEMENTS",
    "get_element_embedding",
    "get_composition_embedding",
    "element_similarity_matrix",
    "pca_2d",
    "AlloyPredictor",
    "EXAMPLE_ALLOYS",
]
