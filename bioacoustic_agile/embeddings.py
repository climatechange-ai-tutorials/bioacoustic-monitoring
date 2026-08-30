"""Embedding extraction utilities for bioacoustic models."""

from typing import Protocol, ArrayLike
import numpy as np


class AudioEmbedder(Protocol):
    """Protocol for bioacoustic embedding models."""

    def embed_window(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract embeddings from an audio window."""
        ...


class PerchEmbedder:
    """Adapter for Google's Perch bioacoustic classifier model."""

    def __init__(self, model_url: str | None = None):
        self.model_url = model_url
        self._model = None  # Lazy loading for hardware efficiency

    def load_model(self) -> None:
        if self._model is None:
            # Load TF Hub / PyTorch module here
            pass

    def embed_window(self, audio_data: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
        self.load_model()
        # Preprocessing & inference logic
        return np.zeros((1, 128))  # Placeholder vector