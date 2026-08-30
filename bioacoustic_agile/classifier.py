"""Transfer learning classification heads for target species detection."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class BioacousticProbe:
    """Lightweight classifier head built on pre-trained embeddings."""

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(C=C, class_weight="balanced")

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "BioacousticProbe":
        self.model.fit(embeddings, labels)
        return self

    def predict_confidence(self, embeddings: np.ndarray) -> np.ndarray:
        """Returns target species presence confidence score [0, 1]."""
        return self.model.predict_proba(embeddings)[:, 1]

    def save(self, filepath: str) -> None:
        """Export serialized weights for deployment on edge hardware."""
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BioacousticProbe":
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance