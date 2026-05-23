"""
Label Scarcity POC — semi-supervised pseudo-labeling for RAG query routing.

Inspired by: "Overcoming labelled data scarcity for defect classification
in scanning tunnelling microscopy" (score 8/10).

Core idea: use a handful of labeled examples to pseudo-label a large pool
of unlabeled items via k-NN in TF-IDF embedding space, then expand the
training set iteratively — directly applicable to intent classification
and query routing with limited annotations.
"""

from .pseudo_labeler import SemiSupervisedLabeler, LabeledExample, PseudoLabelResult

__all__ = ["SemiSupervisedLabeler", "LabeledExample", "PseudoLabelResult"]
