"""OLR-WAA: Adaptive and Drift-Resilient Online Regression with Dynamic Weighted Averaging."""

from .olr_waa import OLRWAARegressor, DriftDetector, StreamSimulator

__all__ = ["OLRWAARegressor", "DriftDetector", "StreamSimulator"]
