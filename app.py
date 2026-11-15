"""Streamlit dashboard for batch inference and active-learning review.

Run with::

    streamlit run app.py

Sidebar: choose a source, classifier and confidence threshold, then press
"Run Batch Inference". Results are read back from the output folder, so the
dashboard also works for a run made earlier from the command line.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ingestion import IngestionError, SourceType
from src.model_runner import NO_DETECTION_LABEL, InferenceError, LinearHead, ModelRunner
from src.pipeline import PipelineConfig, run_batch
from src.utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MANIFEST_FILENAME,
    REVIEW_QUEUE_FILENAME,
    VERIFIED_LABELS_FILENAME,
    STATUS_OK,
    append_verified_labels,
    read_manifest,
    split_by_confidence,
    summarize,
    verification_to_label_rows,
)

SOURCE_OPTIONS: dict[str, SourceType] = {
    "Local Folder": SourceType.LOCAL,
    "Cloud Storage (S3/GCS)": SourceType.CLOUD,
    "Google Drive": SourceType.GDRIVE,
}
PATH_HINTS: dict[str, str] = {
    "Local Folder": "/path/to/audio_clips/",
    "Cloud Storage (S3/GCS)": "gs://chirp-public-bucket/soundscapes/powdermill/Recording_2",
    "Google Drive": "https://drive.google.com/drive/folders/<folder-id>",
}
SKIP_OPTION = "— skip / unsure —"
PAGE_SIZE = 10

st.set_page_config(page_title="Bioacoustic Batch Inference", page_icon="🐦", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and classifier...")
def get_runner(classifier_path: str, model_choice: str) -> ModelRunner:
    """Load the backbone + classifier once per (classifier, model) and reuse it."""
    runner = ModelRunner.from_classifier_file(classifier_path, model_choice)
    runner.backbone.load()  # type: ignore[attr-defined]
    return runner


@st.cache_data(show_spinner=False)
def classifier_classes(classifier_path: str) -> list[str]:
    """Class labels in a saved classifier, or [] if it cannot be read."""
    try:
        return list(LinearHead.load(classifier_path).classes)
    except InferenceError:
        return []


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Input")
    source_label = st.radio("Source type", list(SOURCE_OPTIONS), key="source_label")
    input_uri = st.text_input("Path / URI / Drive link", placeholder=PATH_HINTS[source_label])
    classifier_path = st.text_input(
        "Classifier file",
        value="agile_classifier.pt",
        help="The file saved by the notebook's 'Save the classifier' step (linear_classifier.save).",
    )

    st.header("Output")
    output_dir = st.text_input("Output directory", value="outputs")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.5,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.01,
        help="Predictions with confidence below this go to the human review queue. "
        "Changing it re-filters the existing results; no re-run needed.",
    )
    annotator = st.text_input("Annotator ID", value="field_researcher")

    with st.expander("Advanced"):
        model_choice = st.text_input("Model preset", value="perch_8", help="Must match the preset used in the notebook.")
        limit = st.number_input("Max files (0 = all)", min_value=0, value=0, step=1)

out_dir = Path(output_dir)
manifest_path = out_dir / MANIFEST_FILENAME
queue_path = out_dir / REVIEW_QUEUE_FILENAME
verified_path = out_dir / VERIFIED_LABELS_FILENAME

st.title("🐦 Bioacoustic Batch Inference")
st.caption(
    "Classify recordings with the tutorial's Perch embedding model and linear classifier, "
    "then review the clips the model is least sure about."
)

# --------------------------------------------------------------------------- #
# Execution panel
# --------------------------------------------------------------------------- #
run_clicked = st.button("▶ Run Batch Inference", type="primary", disabled=not input_uri.strip())
if run_clicked:
    config = PipelineConfig(
        input_uri=input_uri,
        classifier_path=classifier_path,
        output_dir=output_dir,
        source_type=SOURCE_OPTIONS[source_label].value,
        threshold=threshold,
        model_choice=model_choice,
        limit=int(limit) or None,
    )
    progress_bar = st.progress(0.0, text="Starting...")
    with st.status("Running batch inference...", expanded=True) as status:
        log_box = st.empty()
        log_lines: list[str] = []

        def on_progress(done: int, total: int, message: str) -> None:
            log_lines.append(message)
            log_box.code("\n".join(log_lines[-15:]), language=None)
            frac = done / total if total else 0.0
            progress_bar.progress(min(frac, 1.0), text=f"{done}/{total} files" if total else message)

        try:
            on_progress(0, 0, "Loading model...")
            runner = get_runner(classifier_path, model_choice)
            result = run_batch(config, runner=runner, progress=on_progress)
        except (IngestionError, InferenceError, ValueError) as exc:
            status.update(label="Run failed", state="error")
            st.error(str(exc))
        else:
            s = result.summary
            status.update(
                label=f"Done in {result.elapsed_s:.0f}s — {s.processed_ok} ok, {s.failed} failed, "
                f"{s.review_queue} queued for review",
                state="complete",
                expanded=False,
            )
            st.session_state["review_page"] = 0

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
manifest = read_manifest(manifest_path)
if manifest.empty:
    st.info(f"No results yet in `{manifest_path}`. Choose a source in the sidebar and run batch inference.")
    st.stop()

summary = summarize(manifest, threshold)
_, queue = split_by_confidence(manifest, threshold)
queue.to_csv(queue_path, index=False)  # keep the CSV in sync with the slider

tab_summary, tab_review = st.tabs(["📊 Batch Summary", f"🎧 Human Review Queue ({len(queue)})"])

with tab_summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files processed", summary.total_files)
    c2.metric("High-confidence", summary.high_confidence)
    c3.metric("Needs review", summary.review_queue)
    c4.metric(
        "Mean confidence",
        f"{summary.mean_confidence:.3f}" if summary.mean_confidence is not None else "—",
    )

    st.subheader("High-confidence detections by species")
    if summary.detections_by_species:
        chart_df = pd.DataFrame(
            {"species": list(summary.detections_by_species), "files": list(summary.detections_by_species.values())}
        ).set_index("species")
        st.bar_chart(chart_df, y="files")
    else:
        st.write(f"No species detections at or above confidence {threshold:.2f}.")

    ok = manifest[manifest["status"] == STATUS_OK]
    if not ok.empty:
        st.subheader("Mean confidence by predicted label")
        st.dataframe(
            ok.groupby("predicted_species_label")["confidence_score"]
            .agg(files="count", mean_confidence="mean", min_confidence="min")
            .round(3),
        )

    if summary.failed:
        with st.expander(f"⚠️ {summary.failed} file(s) could not be processed"):
            st.dataframe(
                manifest[manifest["status"] != STATUS_OK][["filename", "file_path", "error_message"]],
                hide_index=True,
            )

    with st.expander("Full predictions manifest"):
        st.dataframe(manifest, hide_index=True)
    st.download_button("Download predictions_manifest.csv", manifest.to_csv(index=False), MANIFEST_FILENAME, "text/csv")

with tab_review:
    classes = classifier_classes(classifier_path) or sorted(
        set(manifest["top_class"].dropna().astype(str)) - {NO_DETECTION_LABEL}
    )
    label_options = [SKIP_OPTION, *classes, NO_DETECTION_LABEL]

    verified = pd.read_csv(verified_path) if verified_path.is_file() else pd.DataFrame()
    done_keys = (
        set(zip(verified["file_path"].astype(str), verified["start_time_s"].astype(float)))
        if not verified.empty
        else set()
    )

    st.write(
        f"Clips with confidence below **{threshold:.2f}**, least confident first. "
        "Listen, pick the correct label and save. Verified labels are written to "
        f"`{verified_path}` for retraining in the notebook."
    )
    if queue.empty:
        st.success("Nothing to review at this threshold.")
    else:
        hide_done = st.toggle("Hide already-verified clips", value=True)
        view = queue
        if hide_done and done_keys:
            mask = [
                (str(r.file_path), float(r.best_window_start_s)) not in done_keys for r in queue.itertuples()
            ]
            view = queue[mask]

        n_pages = max(1, -(-len(view) // PAGE_SIZE))
        page = min(st.session_state.get("review_page", 0), n_pages - 1)
        page = st.number_input(f"Page (of {n_pages})", 1, n_pages, page + 1, key="page_input") - 1
        st.session_state["review_page"] = page

        for idx, row in view.iloc[page * PAGE_SIZE : (page + 1) * PAGE_SIZE].iterrows():
            with st.container(border=True):
                left, right = st.columns([3, 2])
                start, end = float(row["best_window_start_s"]), float(row["best_window_end_s"])
                with left:
                    st.markdown(f"**{row['filename']}** · window {start:.0f}–{end:.0f}s")
                    st.caption(
                        f"Model: `{row['predicted_species_label']}` (confidence {row['confidence_score']:.2f}; "
                        f"top class `{row['top_class']}` p={row['top_class_probability']:.2f})"
                    )
                    audio_path = Path(str(row.get("local_path") or ""))
                    if audio_path.is_file():
                        st.audio(str(audio_path), start_time=int(start), end_time=int(end) or None)
                    else:
                        st.warning("Audio not available locally (download cache may have been cleared).")
                with right:
                    with st.form(key=f"verify_{idx}"):
                        default = row["top_class"] if row["top_class"] in label_options else SKIP_OPTION
                        choice = st.selectbox("Correct label", label_options, index=label_options.index(default))
                        if st.form_submit_button("Save label"):
                            if choice == SKIP_OPTION:
                                st.info("Skipped.")
                            else:
                                rows = verification_to_label_rows(row.to_dict(), choice, annotator)
                                total = append_verified_labels(verified_path, rows)
                                st.success(f"Saved ({total} verified label rows).")

    if verified_path.is_file():
        verified = pd.read_csv(verified_path)
        st.divider()
        st.subheader(f"Verified labels ({len(verified)})")
        st.dataframe(verified, hide_index=True)
        st.download_button(
            "Download verified_labels.csv", verified.to_csv(index=False), VERIFIED_LABELS_FILENAME, "text/csv"
        )
