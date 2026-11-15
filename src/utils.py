"""CSV logging, confidence-threshold queue splitting and label export.

File formats
------------
``predictions_manifest.csv`` -- one row per audio file. The first five columns
are the stable contract (``filename, file_path, predicted_species_label,
confidence_score, status``); the rest add context for review and retraining.

``review_queue.csv`` -- the subset of successfully processed rows with
``confidence_score < threshold``, least confident first.

``verified_labels.csv`` -- expert-verified labels from the review queue, one row
per (window, label). Columns ``filename, start_time_s, end_time_s, label`` use
the same names as the annotation CSVs ``perch-hoplite`` ingests, and
``label_type`` is ``positive`` / ``negative``. :func:`ingest_verified_labels`
loads it into the tutorial's embedding database for retraining.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

DEFAULT_CONFIDENCE_THRESHOLD = 0.75
MANIFEST_FILENAME = "predictions_manifest.csv"
REVIEW_QUEUE_FILENAME = "review_queue.csv"
VERIFIED_LABELS_FILENAME = "verified_labels.csv"

STATUS_OK = "ok"
STATUS_ERROR = "error"

MANIFEST_COLUMNS: list[str] = [
    "filename",
    "file_path",
    "predicted_species_label",
    "confidence_score",
    "status",
    # --- context columns ---
    "relative_path",
    "local_path",
    "top_class",
    "top_class_probability",
    "best_window_start_s",
    "best_window_end_s",
    "num_windows",
    "error_message",
    "processed_at",
]

VERIFIED_LABEL_COLUMNS: list[str] = [
    "filename",
    "start_time_s",
    "end_time_s",
    "label",
    "label_type",
    "file_path",
    "model_label",
    "model_confidence",
    "annotator",
    "verified_at",
]


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Manifest logging
# --------------------------------------------------------------------------- #
class ManifestWriter:
    """Append prediction rows to a CSV as they are produced.

    Each row is flushed immediately, so if a long batch is interrupted the rows
    processed so far are already on disk.

    Usage::

        with ManifestWriter("outputs/predictions_manifest.csv") as log:
            log.write({...})
    """

    def __init__(self, path: str | os.PathLike[str], columns: list[str] = MANIFEST_COLUMNS) -> None:
        self.path = Path(path)
        self.columns = columns
        self._fh: Any = None
        self._writer: csv.DictWriter | None = None
        self.rows_written = 0

    def __enter__(self) -> "ManifestWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        """Write one row; missing columns are left blank."""
        if self._writer is None:
            raise RuntimeError("ManifestWriter must be used as a context manager")
        self._writer.writerow({c: row.get(c, "") for c in self.columns})
        self._fh.flush()
        self.rows_written += 1

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            self._fh.close()


def read_manifest(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a predictions manifest, coercing numeric columns.

    Returns an empty frame with the manifest columns if the file is missing.
    """
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    df = pd.read_csv(path, dtype={"filename": str, "file_path": str, "status": str})
    for col in ("confidence_score", "top_class_probability", "best_window_start_s", "best_window_end_s"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Threshold filtering
# --------------------------------------------------------------------------- #
def validate_threshold(threshold: float) -> float:
    """Return ``threshold`` as float, raising ``ValueError`` if outside [0, 1]."""
    t = float(threshold)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"Confidence threshold must be between 0 and 1, got {threshold}")
    return t


def split_by_confidence(
    df: pd.DataFrame, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split successful predictions into ``(confident, review_queue)``.

    Rows whose ``status`` is not ``ok`` (e.g. corrupted files) are in neither:
    there is nothing to listen to. The review queue is sorted least confident
    first so experts see the most informative clips first.
    """
    t = validate_threshold(threshold)
    ok = df[df["status"] == STATUS_OK]
    low_mask = ok["confidence_score"] < t
    confident = ok[~low_mask]
    queue = ok[low_mask].sort_values("confidence_score", kind="stable")
    return confident, queue


def write_review_queue(
    manifest_path: str | os.PathLike[str],
    queue_path: str | os.PathLike[str],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Re-derive the review queue from a manifest at ``threshold`` and save it.

    This lets the threshold be changed after a run without re-running inference.
    """
    df = read_manifest(manifest_path)
    _, queue = split_by_confidence(df, threshold)
    Path(queue_path).parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(queue_path, index=False)
    return queue


@dataclass
class BatchSummary:
    """Headline numbers for a processed batch."""

    total_files: int
    processed_ok: int
    failed: int
    high_confidence: int
    review_queue: int
    mean_confidence: float | None
    detections_by_species: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def summarize(df: pd.DataFrame, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> BatchSummary:
    """Compute :class:`BatchSummary` for a manifest frame."""
    confident, queue = split_by_confidence(df, threshold)
    ok = df[df["status"] == STATUS_OK]
    from .model_runner import NO_DETECTION_LABEL  # local import keeps utils light

    species = confident[confident["predicted_species_label"] != NO_DETECTION_LABEL]
    counts = species["predicted_species_label"].value_counts()
    return BatchSummary(
        total_files=int(len(df)),
        processed_ok=int(len(ok)),
        failed=int(len(df) - len(ok)),
        high_confidence=int(len(confident)),
        review_queue=int(len(queue)),
        mean_confidence=float(ok["confidence_score"].mean()) if len(ok) else None,
        detections_by_species={str(k): int(v) for k, v in counts.items()},
    )


# --------------------------------------------------------------------------- #
# Verified labels (active-learning export)
# --------------------------------------------------------------------------- #
def verification_to_label_rows(
    row: Mapping[str, Any],
    verified_label: str,
    annotator: str,
    no_detection_label: str = "no_detection",
) -> list[dict[str, Any]]:
    """Turn one expert decision on a review-queue row into label rows.

    Rules (``top`` is the class the model scored highest for this clip):

    * verified == a species             -> ``positive`` for that species, and
      ``negative`` for ``top`` if it is a different species.
    * verified == ``no_detection``      -> ``negative`` for ``top``.

    The window is the one where ``top`` peaked (``best_window_*``).
    """
    top = str(row.get("top_class") or row.get("predicted_species_label") or "")
    base = {
        "filename": row.get("relative_path") or row.get("filename"),
        "start_time_s": row.get("best_window_start_s"),
        "end_time_s": row.get("best_window_end_s"),
        "file_path": row.get("file_path"),
        "model_label": row.get("predicted_species_label"),
        "model_confidence": row.get("confidence_score"),
        "annotator": annotator,
        "verified_at": utc_now(),
    }
    out: list[dict[str, Any]] = []
    if verified_label != no_detection_label:
        out.append({**base, "label": verified_label, "label_type": "positive"})
    if top and top != no_detection_label and top != verified_label:
        out.append({**base, "label": top, "label_type": "negative"})
    return out


def append_verified_labels(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Append label rows to ``verified_labels.csv``, replacing older rows for the
    same ``(filename, start_time_s, label)`` so re-verifying a clip updates it.

    Returns:
        Total number of rows in the file afterwards.
    """
    path = Path(path)
    new = pd.DataFrame(list(rows), columns=VERIFIED_LABEL_COLUMNS)
    if path.is_file():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    key = ["filename", "start_time_s", "label"]
    combined = combined.drop_duplicates(subset=key, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return int(len(combined))


def ingest_verified_labels(
    csv_path: str | os.PathLike[str],
    db: Any,
    dataset_name: str,
    provenance: str = "review_queue",
) -> dict[str, int]:
    """Insert ``verified_labels.csv`` into a perch-hoplite embedding database.

    Intended to run *inside the tutorial notebook* (where ``db`` exists and the
    same recordings have been embedded), just before "Prepare the labeled
    data". Each label is attached to every stored window of the recording that
    overlaps ``[start_time_s, end_time_s)``, mirroring perch-hoplite's own
    annotation ingestor. The recordings must have been embedded from the same
    folder layout, so that ``filename`` matches the database's recording names.

    Returns:
        Count of inserted labels per ``"<label>/<label_type>"``.
    """
    from ml_collections import config_dict
    from perch_hoplite.db import datatypes

    df = pd.read_csv(csv_path)
    kinds = {"positive": datatypes.LabelType.POSITIVE, "negative": datatypes.LabelType.NEGATIVE}
    counts: dict[str, int] = {}
    missing: set[str] = set()
    for filename, annos in df.groupby("filename"):
        window_ids = db.match_window_ids(
            deployments_filter=config_dict.create(eq=dict(project=dataset_name)),
            recordings_filter=config_dict.create(eq=dict(filename=str(filename))),
        )
        if len(window_ids) == 0:
            missing.add(str(filename))
            continue
        for wid in window_ids:
            window = db.get_window(wid)
            start, end = float(window.offsets[0]), float(window.offsets[1])
            hits = annos[(annos["start_time_s"] < end) & (annos["end_time_s"] > start)]
            for _, a in hits.iterrows():
                db.insert_annotation(
                    recording_id=window.recording_id,
                    offsets=window.offsets,
                    label=str(a["label"]),
                    label_type=kinds[str(a["label_type"]).lower()],
                    provenance=provenance,
                    handle_duplicates="skip",
                )
                key = f"{a['label']}/{a['label_type']}"
                counts[key] = counts.get(key, 0) + 1
    db.commit()
    if missing:
        print(f"{len(missing)} recording(s) not found in the database, e.g. {sorted(missing)[:3]}")
    return counts
