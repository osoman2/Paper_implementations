"""
Simulated ElementBERT embeddings for 22 common engineering alloy elements.

In the original paper, embeddings come from a BERT-style encoder trained on
>1M alloy science abstracts. Here we approximate that learned space using a
combination of:
  - Normalized periodic-table properties (atomic number, radius, electronegativity,
    melting point, density, ionization energy, valence electrons)
  - Domain co-occurrence signals derived from common alloy families and phases
    (lightweight alloys, superalloys, shape-memory, high-entropy alloys, steels)

This approximation preserves the key structural relationships the paper reports:
Ti and Zr cluster together (both lightweight, HCP→BCC allotropes); Ni and Co
cluster near superalloy elements; Fe, Cr, Mn cluster in the steel/stainless region.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Raw periodic-table properties (normalized to [0,1] across the 22 elements)
# Columns: [Z, r_atom, X_pauling, T_melt, density, IE1, valence_e]
# ---------------------------------------------------------------------------
_RAW: dict[str, list[float]] = {
    #        Z    r    X     Tm     ρ    IE1   Ve
    "Ti": [22, 1.40, 1.54, 1668, 4.51, 6.82, 4],
    "Ni": [28, 1.24, 1.91, 1455, 8.91, 7.64, 10],
    "Al": [13, 1.43, 1.61, 660,  2.70, 5.99, 3],
    "Fe": [26, 1.26, 1.83, 1538, 7.87, 7.90, 8],
    "Cr": [24, 1.28, 1.66, 1907, 7.19, 6.77, 6],
    "Co": [27, 1.25, 1.88, 1495, 8.90, 7.88, 9],
    "Mo": [42, 1.39, 2.16, 2623, 10.2, 7.09, 6],
    "V":  [23, 1.35, 1.63, 1910, 6.11, 6.75, 5],
    "W":  [74, 1.39, 2.36, 3422, 19.3, 7.86, 6],
    "Nb": [41, 1.46, 1.60, 2477, 8.57, 6.76, 5],
    "Zr": [40, 1.60, 1.33, 1855, 6.51, 6.63, 4],
    "Cu": [29, 1.28, 1.90, 1085, 8.96, 7.73, 11],
    "Mn": [25, 1.27, 1.55, 1246, 7.47, 7.43, 7],
    "Si": [14, 1.17, 1.90, 1414, 2.33, 8.15, 4],
    "Ta": [73, 1.46, 1.50, 3017, 16.4, 7.55, 5],
    "Hf": [72, 1.59, 1.30, 2233, 13.3, 6.83, 4],
    "Re": [75, 1.37, 1.90, 3186, 21.0, 7.83, 7],
    "Ru": [44, 1.34, 2.20, 2334, 12.4, 7.36, 8],
    "B":  [5,  0.87, 2.04, 2075, 2.34, 8.30, 3],
    "C":  [6,  0.77, 2.55, 3827, 2.20, 11.3, 4],
    "Sn": [50, 1.45, 1.96, 232,  7.31, 7.34, 4],
    "Zn": [30, 1.34, 1.65, 420,  7.13, 9.39, 12],
}

# Domain co-occurrence vectors (8 alloy families / phases):
# [lightweight, superalloy, shape_memory, HEA, steel/stainless,
#  refractory, semiconductor, other]
_DOMAIN: dict[str, list[float]] = {
    "Ti": [1.0, 0.3, 0.6, 0.5, 0.1, 0.2, 0.0, 0.2],
    "Ni": [0.1, 1.0, 0.9, 0.6, 0.4, 0.1, 0.0, 0.3],
    "Al": [1.0, 0.5, 0.2, 0.7, 0.3, 0.1, 0.1, 0.4],
    "Fe": [0.1, 0.2, 0.1, 0.5, 1.0, 0.1, 0.0, 0.2],
    "Cr": [0.2, 0.6, 0.1, 0.6, 0.9, 0.2, 0.0, 0.2],
    "Co": [0.1, 0.9, 0.4, 0.5, 0.3, 0.1, 0.0, 0.2],
    "Mo": [0.1, 0.7, 0.1, 0.4, 0.6, 0.4, 0.0, 0.2],
    "V":  [0.3, 0.4, 0.2, 0.5, 0.5, 0.3, 0.0, 0.2],
    "W":  [0.1, 0.5, 0.1, 0.4, 0.4, 0.9, 0.0, 0.1],
    "Nb": [0.3, 0.5, 0.5, 0.5, 0.3, 0.5, 0.0, 0.2],
    "Zr": [0.8, 0.3, 0.4, 0.4, 0.1, 0.2, 0.2, 0.3],
    "Cu": [0.2, 0.2, 0.2, 0.5, 0.3, 0.1, 0.1, 0.8],
    "Mn": [0.1, 0.3, 0.1, 0.5, 0.7, 0.1, 0.0, 0.3],
    "Si": [0.4, 0.3, 0.1, 0.3, 0.4, 0.1, 0.9, 0.3],
    "Ta": [0.1, 0.5, 0.2, 0.4, 0.2, 0.8, 0.0, 0.1],
    "Hf": [0.3, 0.5, 0.1, 0.4, 0.1, 0.6, 0.1, 0.2],
    "Re": [0.1, 0.7, 0.1, 0.3, 0.2, 0.7, 0.0, 0.1],
    "Ru": [0.1, 0.6, 0.1, 0.3, 0.2, 0.5, 0.1, 0.2],
    "B":  [0.2, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3, 0.4],
    "C":  [0.1, 0.2, 0.1, 0.2, 0.6, 0.1, 0.4, 0.3],
    "Sn": [0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.2, 0.7],
    "Zn": [0.3, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.5],
}

# Build normalized embedding matrix
def _build_embeddings() -> tuple[list[str], np.ndarray]:
    symbols = list(_RAW.keys())
    prop_mat = np.array([_RAW[s] for s in symbols], dtype=float)
    dom_mat  = np.array([_DOMAIN[s] for s in symbols], dtype=float)

    # min-max normalize property columns
    prop_min = prop_mat.min(axis=0)
    prop_max = prop_mat.max(axis=0)
    prop_rng = np.where(prop_max - prop_min > 0, prop_max - prop_min, 1.0)
    prop_norm = (prop_mat - prop_min) / prop_rng

    # Concatenate: 7 property dims + 8 domain dims = 15-dim embedding
    emb = np.concatenate([prop_norm, dom_mat], axis=1)

    # L2-normalize each row so cosine similarity == dot product
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)
    return symbols, emb


ELEMENTS, _EMB_MATRIX = _build_embeddings()
_SYM_IDX = {s: i for i, s in enumerate(ELEMENTS)}
DIM = _EMB_MATRIX.shape[1]


def get_element_embedding(symbol: str) -> np.ndarray:
    """Return the 15-dim ElementBERT-style embedding for a chemical element."""
    if symbol not in _SYM_IDX:
        raise KeyError(f"Unknown element: {symbol!r}. Supported: {ELEMENTS}")
    return _EMB_MATRIX[_SYM_IDX[symbol]].copy()


def get_composition_embedding(composition: dict[str, float]) -> np.ndarray:
    """
    Composition-weighted average of element embeddings.

    composition: {symbol: weight_fraction, ...}  (fractions need not sum to 1)
    Returns a 15-dim vector (L2-normalised).
    """
    total = sum(composition.values())
    if total <= 0:
        raise ValueError("Composition fractions must sum to a positive number.")
    vec = np.zeros(DIM, dtype=float)
    for sym, frac in composition.items():
        vec += (frac / total) * get_element_embedding(sym)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def element_similarity_matrix(symbols: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Cosine similarity matrix for the given (or all) elements."""
    syms = symbols if symbols is not None else ELEMENTS
    idx  = [_SYM_IDX[s] for s in syms]
    sub  = _EMB_MATRIX[idx]
    sim  = sub @ sub.T  # dot product of L2-normalised rows = cosine similarity
    return syms, np.clip(sim, -1.0, 1.0)


def pca_2d(symbols: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Project element embeddings to 2-D with PCA (manual, no sklearn needed)."""
    syms = symbols if symbols is not None else ELEMENTS
    idx  = [_SYM_IDX[s] for s in syms]
    X    = _EMB_MATRIX[idx].copy()

    # Center
    X -= X.mean(axis=0)

    # SVD-based PCA (equivalent to sklearn PCA)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    coords   = X @ Vt[:2].T  # project onto first 2 principal components
    return syms, coords
