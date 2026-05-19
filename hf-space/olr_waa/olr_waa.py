"""
OLR-WAA: Online Linear Regression with Dynamic Weighted Averaging.

Core concepts from the paper:
- Maintains a pool of online linear regressors (experts) trained on sliding windows
- Weights each expert using exponentially discounted prediction errors (WAA)
- When drift is detected, older experts are down-weighted more aggressively
- Final prediction = weighted average of all expert predictions

Applicable to:
- Adaptive retrieval score calibration in RAG pipelines
- Real-time user preference drift in recommendation systems
- Embedding drift monitoring in vector search
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Expert:
    """A single online linear regression model (one expert in the ensemble)."""
    weights: np.ndarray
    bias: float = 0.0
    age: int = 0
    cum_loss: float = 0.0
    lr: float = 0.01

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.weights, x) + self.bias)

    def update(self, x: np.ndarray, y: float) -> float:
        pred = self.predict(x)
        error = pred - y
        self.weights -= self.lr * error * x
        self.bias   -= self.lr * error
        self.age    += 1
        self.cum_loss += error ** 2
        return error


class OLRWAARegressor:
    """
    Online Regression with Dynamic Weighted Averaging (OLR-WAA).

    Maintains `n_experts` sliding-window regressors. After each step:
    1. Each expert predicts.
    2. WAA updates each expert's log-weight: w_i -= eta * loss_i
    3. Final prediction = softmax(weights) · predictions
    4. Each expert is updated with the true label.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_experts: int = 5,
        eta: float = 0.5,
        lr: float = 0.02,
        forgetting_factor: float = 0.98,
    ):
        self.n_features = n_features
        self.n_experts  = n_experts
        self.eta        = eta           # WAA learning rate (controls how fast weight shifts)
        self.forgetting_factor = forgetting_factor  # discount old losses per step

        # Stagger expert initialisation so they have diverse starting points
        self.experts = [
            Expert(
                weights=np.random.randn(n_features) * 0.1 * (i + 1),
                bias=np.random.randn() * 0.1,
                lr=lr * (0.5 + 0.25 * i),
            )
            for i in range(n_experts)
        ]
        self.log_weights = np.zeros(n_experts)   # unnormalized WAA weights

        self.history: list[dict] = []

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def step(self, x: np.ndarray, y: float) -> dict:
        """One online step: predict → aggregate → update experts & weights."""
        x = np.asarray(x, dtype=float).reshape(self.n_features)

        # 1. Each expert predicts
        preds  = np.array([e.predict(x) for e in self.experts])
        w_norm = self._softmax(self.log_weights)

        # 2. Ensemble prediction
        ensemble_pred = float(np.dot(w_norm, preds))

        # 3. Update WAA weights using squared loss (before expert parameter update)
        losses = (preds - y) ** 2
        # Apply forgetting factor and subtract eta * loss (WAA mirror-descent step)
        self.log_weights = self.log_weights * self.forgetting_factor - self.eta * losses

        # 4. Update each expert
        errors = [e.update(x, y) for e in self.experts]

        record = {
            "y_true":      y,
            "y_pred":      ensemble_pred,
            "error":       ensemble_pred - y,
            "abs_error":   abs(ensemble_pred - y),
            "weights":     w_norm.copy(),
            "expert_preds": preds.copy(),
        }
        self.history.append(record)
        return record

    def reset(self):
        self.__init__(
            n_features=self.n_features,
            n_experts=self.n_experts,
            eta=self.eta,
            lr=self.experts[0].lr,
            forgetting_factor=self.forgetting_factor,
        )


class DriftDetector:
    """
    Page-Hinkley drift detector adapted for regression.

    Signals drift when the cumulative deviation of errors exceeds a threshold,
    indicating a change in the underlying data distribution.
    """

    def __init__(self, threshold: float = 50.0, delta: float = 0.005):
        self.threshold = threshold
        self.delta     = delta
        self._sum      = 0.0
        self._min_sum  = 0.0
        self._n        = 0
        self.drift_points: list[int] = []

    def update(self, error: float, step: int) -> bool:
        self._n   += 1
        self._sum += abs(error) - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        if ph > self.threshold:
            self.drift_points.append(step)
            self._sum, self._min_sum = 0.0, 0.0   # reset after detecting drift
            return True
        return False

    def reset(self):
        self._sum = self._min_sum = self._n = 0
        self.drift_points = []


class StreamSimulator:
    """
    Generates synthetic data streams with optional concept drift.

    Supports three regimes:
    - 'linear'  : y = a·x + b + noise
    - 'abrupt'  : sudden slope change at drift_point
    - 'gradual' : slope linearly shifts over a transition window
    """

    def __init__(
        self,
        n_steps: int = 200,
        drift_type: str = "abrupt",
        drift_point: int = 100,
        noise_std: float = 0.5,
        a_before: float = 1.0,
        a_after: float = -2.0,
        b: float = 0.0,
        transition_width: int = 20,
        seed: int = 42,
    ):
        self.n_steps         = n_steps
        self.drift_type      = drift_type
        self.drift_point     = drift_point
        self.noise_std       = noise_std
        self.a_before        = a_before
        self.a_after         = a_after
        self.b               = b
        self.transition_width = transition_width
        self.rng             = np.random.default_rng(seed)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        X = self.rng.standard_normal(self.n_steps)
        Y = np.zeros(self.n_steps)
        noise = self.rng.normal(0, self.noise_std, self.n_steps)

        for t in range(self.n_steps):
            if self.drift_type == "linear" or t < self.drift_point:
                a = self.a_before
            elif self.drift_type == "abrupt":
                a = self.a_after
            elif self.drift_type == "gradual":
                progress = min(1.0, (t - self.drift_point) / self.transition_width)
                a = self.a_before + (self.a_after - self.a_before) * progress
            else:
                a = self.a_before
            Y[t] = a * X[t] + self.b + noise[t]

        return X.reshape(-1, 1), Y
