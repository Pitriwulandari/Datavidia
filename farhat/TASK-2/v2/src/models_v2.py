"""Model classes for v2 ensemble (module-level for pickling)."""
from __future__ import annotations

from sklearn.neural_network import MLPClassifier


class ResidualMLP:
    """Simple MLP for ensemble (picklable at module level)."""
    def __init__(self, input_dim: int, hidden: int = 256, n_classes: int = 6, seed: int = 42):
        self.n_classes = n_classes
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(hidden, hidden),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

    def fit(self, X, y):
        self.mlp.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.mlp.predict_proba(X)