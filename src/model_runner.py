"""Audio loading, embedding backbone and classifier-head inference.

The tutorial embeds 5-second windows of 32 kHz audio with the pre-trained Perch
v8 model (``perch-hoplite`` preset ``perch_8``; the model computes its own
log-mel spectrogram internally) and trains a linear classifier on those
embeddings. That classifier is saved by the notebook as
``<db_path>/agile_classifier.pt`` -- despite the extension, a JSON file holding
base64-encoded weights. This module reloads it *without* needing
``perch-hoplite`` and applies it to new audio.

Per-file decision rule (the classifier is multi-label: one sigmoid per class):

* Every window gets a probability per class: ``sigmoid(embedding @ beta + bias)``.
* Each class's file-level probability is its maximum over windows.
* If the best class reaches 0.5 the file is labelled with that class and
  ``confidence_score`` is its probability. Otherwise the file is labelled
  ``no_detection`` and ``confidence_score`` is ``1 - p_best``.

So ``confidence_score`` is always the confidence *in the predicted label*, and
values near 0.5 mark the clips the model is least sure about -- exactly the
ones worth sending to a human (uncertainty sampling).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

NO_DETECTION_LABEL = "no_detection"
DEFAULT_MODEL_CHOICE = "perch_8"
DECISION_THRESHOLD = 0.5


class AudioLoadError(RuntimeError):
    """Raised when an audio file is missing, corrupted, empty or unsupported."""


class InferenceError(RuntimeError):
    """Raised when the model or classifier cannot produce a prediction."""


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal, preferring librosa and falling back to scipy."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g)


def load_audio(path: str | os.PathLike[str], target_sample_rate: int) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sample_rate``.

    Args:
        path: A local ``.wav`` or ``.flac`` file.
        target_sample_rate: Sample rate expected by the embedding model.

    Returns:
        1-D float32 array.

    Raises:
        AudioLoadError: If the file cannot be decoded or contains no usable audio.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AudioLoadError("soundfile is required to read audio: pip install soundfile") from exc

    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File not found: {path}")
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises several types
        raise AudioLoadError(f"Corrupted or unsupported audio ({type(exc).__name__}: {exc})") from exc

    if audio.size == 0:
        raise AudioLoadError("Audio file contains no samples")
    audio = audio.mean(axis=1)  # downmix to mono
    if not np.all(np.isfinite(audio)):
        raise AudioLoadError("Audio contains NaN or infinite samples")
    try:
        audio = _resample(audio, int(sr), int(target_sample_rate))
    except Exception as exc:  # noqa: BLE001
        raise AudioLoadError(f"Resampling from {sr} Hz failed: {exc}") from exc
    return np.asarray(audio, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Embedding backbone
# --------------------------------------------------------------------------- #
class EmbeddingBackbone(Protocol):
    """Anything that turns mono audio into one embedding per window."""

    sample_rate: int
    window_size_s: float
    hop_size_s: float

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Return embeddings of shape ``[num_windows, embedding_dim]``."""
        ...


class PerchBackbone:
    """The tutorial's pre-trained backbone, loaded through ``perch-hoplite``.

    The model is downloaded from Kaggle Models on first use and cached by
    ``kagglehub``. Loading is deferred until :meth:`load` or the first
    :meth:`embed` call so that importing this module stays cheap.
    """

    def __init__(self, model_choice: str = DEFAULT_MODEL_CHOICE) -> None:
        try:
            from perch_hoplite.zoo import model_configs
        except ImportError as exc:
            raise InferenceError(
                "The Perch backbone needs perch-hoplite: "
                "pip install git+https://github.com/google-research/perch-hoplite.git"
            ) from exc
        self.model_choice = model_choice
        self._preset = model_configs.get_preset_model_config(model_choice)
        cfg = self._preset.model_config
        self.sample_rate = int(cfg.sample_rate)
        self.window_size_s = float(cfg.window_size_s)
        self.hop_size_s = float(cfg.hop_size_s)
        self.embedding_dim = int(self._preset.embedding_dim)
        self._model: Any = None

    def load(self) -> None:
        """Download (if needed) and load the model."""
        if self._model is None:
            self._model = self._preset.load_model()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self.load()
        out = self._model.embed(audio).embeddings  # [frames, channels, dim]
        if out is None:
            raise InferenceError("Backbone returned no embeddings")
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3:
            out = out.mean(axis=1)  # Perch v8 has a single channel
        return out


# --------------------------------------------------------------------------- #
# Classifier head
# --------------------------------------------------------------------------- #
@dataclass
class LinearHead:
    """The tutorial's linear classifier: ``logits = embeddings @ beta + beta_bias``.

    Attributes:
        beta: Weights, shape ``[embedding_dim, num_classes]``.
        beta_bias: Bias, shape ``[num_classes]``.
        classes: Class labels (eBird codes such as ``woothr``).
        model_config: Embedding-model config stored with the classifier, if any.
    """

    beta: np.ndarray
    beta_bias: np.ndarray
    classes: tuple[str, ...]
    model_config: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        return int(self.beta.shape[0])

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        """Return logits of shape ``[num_windows, num_classes]``."""
        return embeddings @ self.beta + self.beta_bias

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """Return per-class sigmoid probabilities, shape ``[num_windows, num_classes]``."""
        z = np.clip(self.logits(embeddings), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LinearHead":
        """Load a classifier saved by the notebook (``linear_classifier.save(...)``).

        Raises:
            InferenceError: If the file is missing or not a perch-hoplite classifier.
        """
        path = Path(path).expanduser()
        if not path.is_file():
            raise InferenceError(f"Classifier file not found: {path}")
        try:
            cfg = json.loads(path.read_text())
            classes = tuple(str(c) for c in cfg["classes"])
            beta = np.frombuffer(base64.b64decode(cfg["beta"]), dtype=np.float32)
            beta = beta.reshape(-1, len(classes)).copy()
            bias = np.frombuffer(base64.b64decode(cfg["beta_bias"]), dtype=np.float32).copy()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InferenceError(
                f"{path} is not a perch-hoplite LinearClassifier file "
                f"(expected the JSON written by linear_classifier.save): {exc}"
            ) from exc
        if bias.shape != (len(classes),):
            raise InferenceError(f"Classifier bias has shape {bias.shape}, expected ({len(classes)},)")
        return cls(beta, bias, classes, dict(cfg.get("model_config") or {}))

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the classifier in the same JSON format perch-hoplite uses."""
        payload = {
            "model_config": self.model_config,
            "classes": list(self.classes),
            "beta": base64.b64encode(np.float32(self.beta).tobytes()).decode("ascii"),
            "beta_bias": base64.b64encode(np.float32(self.beta_bias).tobytes()).decode("ascii"),
        }
        Path(path).write_text(json.dumps(payload))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
@dataclass
class FilePrediction:
    """File-level prediction.

    Attributes:
        predicted_species_label: Best class, or ``no_detection``.
        confidence_score: Confidence in ``predicted_species_label`` (0-1).
        top_class: Highest-scoring class even when below the decision threshold.
        top_class_probability: Its probability (max over windows).
        best_window_start_s / best_window_end_s: Where ``top_class`` peaks.
        num_windows: Number of windows embedded.
    """

    predicted_species_label: str
    confidence_score: float
    top_class: str
    top_class_probability: float
    best_window_start_s: float
    best_window_end_s: float
    num_windows: int


class ModelRunner:
    """Run backbone + classifier head over audio files."""

    def __init__(self, backbone: EmbeddingBackbone, head: LinearHead) -> None:
        self.backbone = backbone
        self.head = head
        dim = getattr(backbone, "embedding_dim", None)
        if dim is not None and dim != head.embedding_dim:
            raise InferenceError(
                f"Classifier expects {head.embedding_dim}-d embeddings but the backbone "
                f"produces {dim}-d. Was the classifier trained with a different model?"
            )

    @classmethod
    def from_classifier_file(
        cls, classifier_path: str | os.PathLike[str], model_choice: str = DEFAULT_MODEL_CHOICE
    ) -> "ModelRunner":
        """Build the standard runner: Perch backbone + the notebook's saved classifier."""
        return cls(PerchBackbone(model_choice), LinearHead.load(classifier_path))

    @property
    def classes(self) -> tuple[str, ...]:
        return self.head.classes

    def predict_audio(self, audio: np.ndarray) -> FilePrediction:
        """Classify an already-loaded mono signal at the backbone's sample rate."""
        try:
            emb = self.backbone.embed(audio)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Embedding failed: {exc}") from exc
        if emb.ndim != 2 or emb.shape[0] == 0:
            raise InferenceError(f"Unexpected embedding shape {emb.shape}")
        if emb.shape[1] != self.head.embedding_dim:
            raise InferenceError(
                f"Embedding dim {emb.shape[1]} does not match classifier dim {self.head.embedding_dim}"
            )

        probs = self.head.probabilities(emb)  # [W, C]
        per_class = probs.max(axis=0)
        top = int(per_class.argmax())
        p_top = float(per_class[top])
        win = int(probs[:, top].argmax())
        start = win * self.backbone.hop_size_s
        end = start + self.backbone.window_size_s

        if p_top >= DECISION_THRESHOLD:
            label, conf = self.head.classes[top], p_top
        else:
            label, conf = NO_DETECTION_LABEL, 1.0 - p_top
        return FilePrediction(
            predicted_species_label=label,
            confidence_score=round(conf, 6),
            top_class=self.head.classes[top],
            top_class_probability=round(p_top, 6),
            best_window_start_s=round(start, 3),
            best_window_end_s=round(end, 3),
            num_windows=int(emb.shape[0]),
        )

    def predict_file(self, path: str | os.PathLike[str]) -> FilePrediction:
        """Load and classify one file.

        Raises:
            AudioLoadError: For unreadable audio.
            InferenceError: For model/classifier failures.
        """
        audio = load_audio(path, self.backbone.sample_rate)
        return self.predict_audio(audio)
