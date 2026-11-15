"""Batch inference and active-learning tooling for the bioacoustic tutorial.

This package sits *beside* the tutorial notebook and never modifies it. It wraps
the same pre-trained embedding backbone (Perch v8 via ``perch-hoplite``) and the
linear classifier the notebook trains, so that a classifier saved from the
notebook (``agile_classifier.pt``) can be run over new recordings from the
command line or the Streamlit app.

Modules:
    ingestion:    Discover and fetch audio from local folders, S3/GCS and Google Drive.
    model_runner: Audio loading, embedding backbone and classifier head inference.
    utils:        CSV logging, confidence-threshold queue splitting, label export.
    pipeline:     The end-to-end batch job and its CLI (``python -m src.pipeline``).
"""

__version__ = "0.2.0"
