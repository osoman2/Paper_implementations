"""
Alloy property predictor using ElementBERT-style composition embeddings.

In the original paper, downstream models (Random Forest, Gaussian Process, etc.)
are trained on top of ElementBERT composition features.  Here we use a small
collection of well-known engineering alloys with reported properties to train
a simple sklearn Ridge regressor — the point being to show the pipeline:
  composition → ElementBERT embedding → property model → prediction.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .embeddings import get_composition_embedding

# ---------------------------------------------------------------------------
# Known alloys with approximate properties
# Sources: ASM International, Matweb, literature averages
# ---------------------------------------------------------------------------
EXAMPLE_ALLOYS: list[dict] = [
    {
        "name": "Ti-6Al-4V (Grade 5)",
        "composition": {"Ti": 0.90, "Al": 0.06, "V": 0.04},
        "UTS_MPa": 950, "hardness_HV": 349, "density_gcc": 4.43,
    },
    {
        "name": "Inconel 718",
        "composition": {"Ni": 0.52, "Fe": 0.19, "Cr": 0.19, "Nb": 0.05, "Mo": 0.03, "Al": 0.005, "Ti": 0.009},
        "UTS_MPa": 1375, "hardness_HV": 400, "density_gcc": 8.19,
    },
    {
        "name": "316L Stainless Steel",
        "composition": {"Fe": 0.68, "Cr": 0.17, "Ni": 0.12, "Mo": 0.025, "Mn": 0.02},
        "UTS_MPa": 570, "hardness_HV": 175, "density_gcc": 7.99,
    },
    {
        "name": "7075-T6 Aluminium",
        "composition": {"Al": 0.88, "Zn": 0.06, "Mg": 0.025, "Cu": 0.016},
        "UTS_MPa": 572, "hardness_HV": 175, "density_gcc": 2.81,
    },
    {
        "name": "Nitinol 55 (Ni-Ti SMA)",
        "composition": {"Ni": 0.55, "Ti": 0.45},
        "UTS_MPa": 900, "hardness_HV": 260, "density_gcc": 6.45,
    },
    {
        "name": "CoCrFeMnNi (Cantor HEA)",
        "composition": {"Co": 0.20, "Cr": 0.20, "Fe": 0.20, "Mn": 0.20, "Ni": 0.20},
        "UTS_MPa": 660, "hardness_HV": 168, "density_gcc": 7.97,
    },
    {
        "name": "Hastelloy C-276",
        "composition": {"Ni": 0.57, "Mo": 0.16, "Cr": 0.15, "Fe": 0.057, "W": 0.035},
        "UTS_MPa": 785, "hardness_HV": 220, "density_gcc": 8.89,
    },
    {
        "name": "Ti-6Al-2Sn-4Zr-2Mo",
        "composition": {"Ti": 0.86, "Al": 0.06, "Zr": 0.04, "Sn": 0.02, "Mo": 0.02},
        "UTS_MPa": 1000, "hardness_HV": 310, "density_gcc": 4.54,
    },
    {
        "name": "WC-Co (cemented carbide, 10% Co)",
        "composition": {"W": 0.856, "C": 0.044, "Co": 0.10},
        "UTS_MPa": 2100, "hardness_HV": 1500, "density_gcc": 14.5,
    },
    {
        "name": "Pure Copper (annealed)",
        "composition": {"Cu": 1.0},
        "UTS_MPa": 220, "hardness_HV": 48, "density_gcc": 8.96,
    },
    {
        "name": "Maraging Steel 250",
        "composition": {"Fe": 0.68, "Ni": 0.18, "Co": 0.08, "Mo": 0.05, "Ti": 0.004},
        "UTS_MPa": 1720, "hardness_HV": 514, "density_gcc": 8.00,
    },
    {
        "name": "René 95 (PM superalloy)",
        "composition": {"Ni": 0.61, "Co": 0.08, "Cr": 0.14, "Mo": 0.035, "W": 0.035,
                        "Nb": 0.035, "Al": 0.035, "Ti": 0.025, "Hf": 0.005},
        "UTS_MPa": 1310, "hardness_HV": 393, "density_gcc": 8.26,
    },
]

# Filter to elements that exist in our embedding vocabulary
_VOCAB = {
    "Ti", "Ni", "Al", "Fe", "Cr", "Co", "Mo", "V", "W", "Nb",
    "Zr", "Cu", "Mn", "Si", "Ta", "Hf", "Re", "Ru", "B", "C", "Sn", "Zn",
}


def _filter_composition(comp: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in comp.items() if k in _VOCAB}


PROPERTIES = ["UTS_MPa", "hardness_HV", "density_gcc"]
_PROP_LABELS = {
    "UTS_MPa":       "Ultimate Tensile Strength (MPa)",
    "hardness_HV":   "Vickers Hardness (HV)",
    "density_gcc":   "Density (g/cc)",
}


class AlloyPredictor:
    """
    Thin sklearn Ridge wrapper trained on the example alloy library.

    Mirrors the downstream model pattern from the paper:
      composition → composition embedding (weighted average of element embeddings)
      → property prediction.
    """

    def __init__(self) -> None:
        self._scalers: dict[str, StandardScaler] = {}
        self._models:  dict[str, Ridge]          = {}
        self._fitted = False
        self._fit()

    def _fit(self) -> None:
        X: list[np.ndarray] = []
        ys: dict[str, list[float]] = {p: [] for p in PROPERTIES}

        for alloy in EXAMPLE_ALLOYS:
            filt = _filter_composition(alloy["composition"])
            if not filt:
                continue
            X.append(get_composition_embedding(filt))
            for p in PROPERTIES:
                ys[p].append(float(alloy[p]))

        X_arr = np.array(X)
        for p in PROPERTIES:
            sc  = StandardScaler()
            Xs  = sc.fit_transform(X_arr)
            mdl = Ridge(alpha=1.0)
            mdl.fit(Xs, np.array(ys[p]))
            self._scalers[p] = sc
            self._models[p]  = mdl

        self._fitted = True

    def predict(self, composition: dict[str, float]) -> dict[str, float]:
        """Return predicted properties for a given alloy composition."""
        filt = _filter_composition(composition)
        if not filt:
            return {p: 0.0 for p in PROPERTIES}
        emb = get_composition_embedding(filt).reshape(1, -1)
        return {
            p: float(self._models[p].predict(self._scalers[p].transform(emb))[0])
            for p in PROPERTIES
        }

    @property
    def property_labels(self) -> dict[str, str]:
        return _PROP_LABELS
