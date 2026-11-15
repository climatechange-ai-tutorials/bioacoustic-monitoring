"""Multi-source batch inference pipeline and its command-line interface.

Example::

    python -m src.pipeline \
        --input gs://chirp-public-bucket/soundscapes/powdermill/Recording_2 \
        --classifier path/to/agile_classifier.pt \
        --output-dir outputs --threshold 0.75

Produces ``<output-dir>/predictions_manifest.csv`` and
``<output-dir>/review_queue.csv``. A file that cannot be downloaded, decoded or
classified is logged with ``status=error`` and the batch carries on.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .ingestion import DEFAULT_CACHE_DIR, IngestionError, SourceType, iter_fetched, open_source
from .model_runner import DEFAULT_MODEL_CHOICE, AudioLoadError, InferenceError, ModelRunner
from .utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MANIFEST_FILENAME,
    REVIEW_QUEUE_FILENAME,
    STATUS_ERROR,
    STATUS_OK,
    BatchSummary,
    ManifestWriter,
    read_manifest,
    summarize,
    utc_now,
    validate_threshold,
    write_review_queue,
)

logger = logging.getLogger("bioacoustic.pipeline")

#: ``callback(files_done, files_total, message)`` -- used by the CLI and Streamlit.
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class PipelineConfig:
    """Settings for one batch run.

    Attributes:
        input_uri: Local folder/file, ``s3://``/``gs://`` prefix, or Drive link/ID.
        classifier_path: The notebook's saved classifier (``agile_classifier.pt``).
        output_dir: Where the manifest and review queue are written.
        source_type: ``auto``, ``local``, ``cloud`` or ``gdrive``.
        threshold: Rows with ``confidence_score`` below this go to the review queue.
        model_choice: perch-hoplite preset; must match the one used for training.
        cache_dir: Download cache for remote sources.
        limit: Process at most this many files (useful for a quick trial run).
    """

    input_uri: str
    classifier_path: str
    output_dir: str = "outputs"
    source_type: str = SourceType.AUTO.value
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    model_choice: str = DEFAULT_MODEL_CHOICE
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    limit: int | None = None

    @property
    def manifest_path(self) -> Path:
        return Path(self.output_dir) / MANIFEST_FILENAME

    @property
    def review_queue_path(self) -> Path:
        return Path(self.output_dir) / REVIEW_QUEUE_FILENAME


@dataclass
class BatchResult:
    """What a run produced."""

    manifest_path: Path
    review_queue_path: Path
    summary: BatchSummary
    elapsed_s: float
    errors: list[tuple[str, str]] = field(default_factory=list)


def run_batch(
    config: PipelineConfig,
    runner: ModelRunner | None = None,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    """Discover audio, classify each file, and write the manifest and review queue.

    Args:
        config: Run settings.
        runner: A pre-built :class:`ModelRunner` (lets callers cache the loaded
            model between runs, or inject a test double). Built from
            ``config.classifier_path`` when omitted.
        progress: Optional progress callback.

    Raises:
        IngestionError: If the source cannot be opened/listed or has no audio.
        InferenceError: If the model or classifier cannot be loaded.
        ValueError: For an invalid threshold.
    """
    threshold = validate_threshold(config.threshold)
    report = progress or (lambda done, total, msg: None)
    t0 = time.perf_counter()

    source = open_source(config.input_uri, config.source_type, cache_dir=config.cache_dir)
    report(0, 0, f"Scanning {source.root} for .wav/.flac files...")
    items = source.list_audio()
    if config.limit:
        items = items[: config.limit]
    if not items:
        raise IngestionError(f"No .wav or .flac files found under {source.root}")
    total = len(items)
    report(0, total, f"Found {total} audio file(s).")

    if runner is None:
        report(0, total, "Loading embedding model and classifier (first run downloads the model)...")
        runner = ModelRunner.from_classifier_file(config.classifier_path, config.model_choice)

    errors: list[tuple[str, str]] = []
    with ManifestWriter(config.manifest_path) as log:
        for i, (item, fetched) in enumerate(iter_fetched(source, items), start=1):
            row: dict[str, object] = {
                "filename": item.filename,
                "file_path": item.uri,
                "relative_path": item.relative_path,
                "processed_at": utc_now(),
            }
            try:
                if isinstance(fetched, Exception):
                    raise fetched
                row["local_path"] = str(fetched)
                pred = runner.predict_file(fetched)
                row.update(pred.__dict__)
                row["status"] = STATUS_OK
                msg = f"{item.filename}: {pred.predicted_species_label} ({pred.confidence_score:.2f})"
            except (AudioLoadError, InferenceError, IngestionError) as exc:
                row.update(status=STATUS_ERROR, error_message=str(exc))
                errors.append((item.uri, str(exc)))
                msg = f"{item.filename}: ERROR {exc}"
                logger.warning("Skipping %s: %s", item.uri, exc)
            log.write(row)
            report(i, total, msg)

    queue = write_review_queue(config.manifest_path, config.review_queue_path, threshold)
    summary = summarize(read_manifest(config.manifest_path), threshold)
    report(total, total, f"Done. {summary.processed_ok} ok, {summary.failed} failed, {len(queue)} queued for review.")
    return BatchResult(
        manifest_path=config.manifest_path,
        review_queue_path=config.review_queue_path,
        summary=summary,
        elapsed_s=time.perf_counter() - t0,
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m src.pipeline``."""
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Batch species classification with the tutorial's Perch backbone and linear classifier.",
    )
    p.add_argument("-i", "--input", required=True, help="Local folder/file, s3:// or gs:// URI, or Google Drive link/ID.")
    p.add_argument("-c", "--classifier", required=True, help="Classifier saved by the notebook (agile_classifier.pt).")
    p.add_argument("-o", "--output-dir", default="outputs", help="Output folder (default: outputs).")
    p.add_argument(
        "-s", "--source-type", default="auto", choices=[s.value for s in SourceType],
        help="Input source type (default: auto-detect).",
    )
    p.add_argument(
        "-t", "--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Review-queue cutoff: rows with confidence below this are queued (default: 0.75).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_CHOICE, help="perch-hoplite model preset (default: perch_8).")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Download cache for remote sources.")
    p.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    p.add_argument("-q", "--quiet", action="store_true", help="Only print the final summary.")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.ERROR if args.quiet else logging.WARNING, format="%(levelname)s %(message)s")
    config = PipelineConfig(
        input_uri=args.input,
        classifier_path=args.classifier,
        output_dir=args.output_dir,
        source_type=args.source_type,
        threshold=args.threshold,
        model_choice=args.model,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )

    def cli_progress(done: int, total: int, msg: str) -> None:
        if not args.quiet:
            prefix = f"[{done}/{total}] " if total else ""
            print(prefix + msg, flush=True)

    try:
        result = run_batch(config, progress=cli_progress)
    except (IngestionError, InferenceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    s = result.summary
    print(
        f"\nProcessed {s.total_files} file(s) in {result.elapsed_s:.1f}s: "
        f"{s.processed_ok} ok, {s.failed} failed, {s.high_confidence} confident, "
        f"{s.review_queue} queued for review."
    )
    if s.mean_confidence is not None:
        print(f"Mean confidence: {s.mean_confidence:.3f}")
    print(f"Manifest     : {result.manifest_path}")
    print(f"Review queue : {result.review_queue_path}")
    return 0 if s.processed_ok else 1


if __name__ == "__main__":
    sys.exit(main())
