"""Tests for src/ that run without perch-hoplite, a GPU or network access.

The Perch backbone is replaced by a tiny deterministic fake, so these check the
plumbing (ingestion, error handling, CSV outputs, queue splitting, label
export) rather than model quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.ingestion import IngestionError, LocalSource, SourceType, detect_source_type, open_source, parse_drive_id
from src.model_runner import NO_DETECTION_LABEL, AudioLoadError, LinearHead, ModelRunner, load_audio
from src.pipeline import PipelineConfig, main, run_batch
from src.utils import (
    append_verified_labels,
    read_manifest,
    split_by_confidence,
    summarize,
    verification_to_label_rows,
)

SR = 16_000


class FakeBackbone:
    """2-d 'embedding' per 1 s window: [RMS * 10, 1]."""

    sample_rate = SR
    window_size_s = 1.0
    hop_size_s = 1.0
    embedding_dim = 2

    def embed(self, audio: np.ndarray) -> np.ndarray:
        n = max(1, len(audio) // SR)
        frames = np.array_split(audio[: n * SR], n)
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        return np.stack([rms * 10, np.ones_like(rms)], axis=1).astype(np.float32)


def make_head() -> LinearHead:
    # logit = 4 * (10 * rms) - 2  ->  loud windows are 'woothr'
    return LinearHead(np.array([[4.0], [0.0]], np.float32), np.array([-2.0], np.float32), ("woothr",))


@pytest.fixture()
def audio_dir(tmp_path: Path) -> Path:
    root = tmp_path / "clips"
    (root / "day1").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, amp in [("loud.wav", 0.3), ("quiet.wav", 0.001), ("edge.wav", 0.05)]:
        sf.write(root / "day1" / name, (amp * rng.standard_normal(SR * 3)).astype(np.float32), SR)
    sf.write(root / "other.flac", (0.3 * rng.standard_normal(SR * 2)).astype(np.float32), SR)
    (root / "corrupt.wav").write_bytes(b"RIFF\x00\x00not really audio")
    (root / "notes.txt").write_text("ignore me")
    return root


def test_local_source_scans_recursively(audio_dir: Path) -> None:
    items = LocalSource(audio_dir).list_audio()
    assert [i.relative_path for i in items] == ["corrupt.wav", "day1/edge.wav", "day1/loud.wav", "day1/quiet.wav", "other.flac"]


def test_local_source_follows_symlinked_folders(audio_dir: Path, tmp_path: Path) -> None:
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "day1").symlink_to(audio_dir / "day1", target_is_directory=True)
    assert [i.relative_path for i in LocalSource(linked).list_audio()] == ["day1/edge.wav", "day1/loud.wav", "day1/quiet.wav"]


def test_detect_source_type(audio_dir: Path) -> None:
    assert detect_source_type("s3://bucket/x") is SourceType.CLOUD
    assert detect_source_type("gs://bucket/x/") is SourceType.CLOUD
    assert detect_source_type("https://drive.google.com/drive/folders/1AbC_dEf") is SourceType.GDRIVE
    assert detect_source_type(str(audio_dir)) is SourceType.LOCAL
    assert detect_source_type("1A2b3C4d5E6f7G8h9I0jKlMnOpQ") is SourceType.GDRIVE


def test_parse_drive_id() -> None:
    assert parse_drive_id("https://drive.google.com/drive/folders/abc123_-X?usp=sharing") == ("abc123_-X", True)
    assert parse_drive_id("https://drive.google.com/file/d/xyz789/view") == ("xyz789", False)
    with pytest.raises(IngestionError):
        parse_drive_id("not a link")


def test_missing_local_path_raises() -> None:
    with pytest.raises(IngestionError):
        open_source("/definitely/not/here", "local")


def test_load_audio_rejects_corrupt_file(audio_dir: Path) -> None:
    with pytest.raises(AudioLoadError):
        load_audio(audio_dir / "corrupt.wav", SR)


def test_load_audio_resamples(audio_dir: Path) -> None:
    assert load_audio(audio_dir / "day1" / "loud.wav", 8_000).shape == (8_000 * 3,)


def test_linear_head_roundtrip(tmp_path: Path) -> None:
    head = make_head()
    head.save(tmp_path / "clf.pt")
    loaded = LinearHead.load(tmp_path / "clf.pt")
    assert loaded.classes == ("woothr",)
    np.testing.assert_allclose(loaded.beta, head.beta)
    np.testing.assert_allclose(loaded.beta_bias, head.beta_bias)


def test_predictions_and_no_detection(audio_dir: Path) -> None:
    runner = ModelRunner(FakeBackbone(), make_head())
    loud = runner.predict_file(audio_dir / "day1" / "loud.wav")
    quiet = runner.predict_file(audio_dir / "day1" / "quiet.wav")
    assert loud.predicted_species_label == "woothr" and loud.confidence_score > 0.9
    assert quiet.predicted_species_label == NO_DETECTION_LABEL and quiet.confidence_score > 0.8
    assert loud.num_windows == 3


def test_run_batch_end_to_end(audio_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cfg = PipelineConfig(input_uri=str(audio_dir), classifier_path="unused", output_dir=str(out), threshold=0.75)
    result = run_batch(cfg, runner=ModelRunner(FakeBackbone(), make_head()))

    manifest = read_manifest(result.manifest_path)
    assert list(manifest.columns[:5]) == ["filename", "file_path", "predicted_species_label", "confidence_score", "status"]
    assert len(manifest) == 5
    bad = manifest[manifest["filename"] == "corrupt.wav"].iloc[0]
    assert bad["status"] == "error" and "Corrupted" in bad["error_message"]

    queue = pd.read_csv(result.review_queue_path)
    assert (queue["confidence_score"] < 0.75).all()
    assert "edge.wav" in set(queue["filename"])
    assert result.summary.failed == 1
    assert result.summary.processed_ok == 4
    assert result.summary.detections_by_species.get("woothr") == 2


def test_split_by_confidence_threshold() -> None:
    df = pd.DataFrame(
        {"status": ["ok", "ok", "ok", "error"], "confidence_score": [0.9, 0.6, 0.75, np.nan],
         "predicted_species_label": ["a", "b", "a", ""]}
    )
    confident, queue = split_by_confidence(df, 0.75)
    assert list(confident["confidence_score"]) == [0.9, 0.75]
    assert list(queue["confidence_score"]) == [0.6]
    with pytest.raises(ValueError):
        split_by_confidence(df, 1.5)
    assert summarize(df, 0.75).failed == 1


def test_verified_label_export(tmp_path: Path) -> None:
    row = {"relative_path": "day1/edge.wav", "filename": "edge.wav", "file_path": "/x/edge.wav",
           "best_window_start_s": 0.0, "best_window_end_s": 5.0, "top_class": "woothr",
           "predicted_species_label": "woothr", "confidence_score": 0.6}
    assert [(r["label"], r["label_type"]) for r in verification_to_label_rows(row, "woothr", "me")] == [("woothr", "positive")]
    assert [(r["label"], r["label_type"]) for r in verification_to_label_rows(row, NO_DETECTION_LABEL, "me")] == [("woothr", "negative")]
    swapped = verification_to_label_rows(row, "amerob", "me")
    assert {(r["label"], r["label_type"]) for r in swapped} == {("amerob", "positive"), ("woothr", "negative")}

    path = tmp_path / "verified_labels.csv"
    append_verified_labels(path, verification_to_label_rows(row, NO_DETECTION_LABEL, "me"))
    assert append_verified_labels(path, verification_to_label_rows(row, "woothr", "me")) == 1  # re-verify replaces
    assert pd.read_csv(path).iloc[0]["label_type"] == "positive"


def test_cli_reports_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--input", str(tmp_path / "nope"), "--classifier", "x.pt", "--source-type", "local", "-o", str(tmp_path)])
    assert code == 2
    assert "does not exist" in capsys.readouterr().err
