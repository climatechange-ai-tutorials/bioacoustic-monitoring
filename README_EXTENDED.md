# Extended tooling: batch inference and active-learning review

This adds a command-line batch pipeline and a Streamlit dashboard next to the
tutorial notebook. **The notebook is unchanged and does not depend on any of
this.** Use the notebook to train a classifier, then use these tools to run
that classifier over new recordings and send the uncertain clips to a person
for review.

```
src/
  ingestion.py     local folders, s3:// / gs:// buckets, public Google Drive folders
  model_runner.py  audio loading, Perch v8 embeddings, the notebook's linear classifier
  utils.py         CSV logging, confidence-threshold queue, verified-label export
  pipeline.py      the batch job + CLI (python -m src.pipeline)
app.py             Streamlit dashboard
outputs/           default output folder
tests/             tests that use a fake model (no GPU, network or perch-hoplite needed)
```

## 1. Install

Python 3.10–3.14, plus `libsndfile1` (and `ffmpeg` for the model) on Linux.

```bash
pip install -r requirements.txt
```

To install only what you need:

```bash
pip install -e ".[model,app]"          # local folders + dashboard
pip install -e ".[model,app,cloud,gdrive]"   # everything
```

The `model` extra installs `perch-hoplite` (TensorFlow). PyTorch and torchaudio are
**not** needed: the tutorial's backbone is a TensorFlow model, and audio is read
with `soundfile`. The model downloads from Kaggle Models on the first run and is
cached afterwards. Inference runs on CPU; a GPU only makes it faster.

## 2. Get a classifier from the notebook

The notebook's "Show the metrics" cell saves the trained classifier:

```python
classifier_path = os.path.join(db_path, 'agile_classifier.pt')
linear_classifier.save(classifier_path)
```

On Colab, download `/content/powdermill_db/agile_classifier.pt` (despite the
`.pt` extension it is a small JSON file). The pipeline must use the same model
preset as the notebook (`perch_8`, the default).

## 3. Run from the command line

```bash
python -m src.pipeline --input /path/to/audio_clips/ --classifier agile_classifier.pt
```

```bash
python -m src.pipeline -i gs://chirp-public-bucket/soundscapes/powdermill/ -c agile_classifier.pt --limit 10
```

```bash
python -m src.pipeline -i "https://drive.google.com/drive/folders/<folder-id>" -c agile_classifier.pt -t 0.8
```

| Option | Default | Meaning |
|---|---|---|
| `-i, --input` | required | Local folder or file, `s3://` / `gs://` prefix, or Google Drive link or ID |
| `-c, --classifier` | required | `agile_classifier.pt` from the notebook |
| `-o, --output-dir` | `outputs` | Where the CSV files go |
| `-s, --source-type` | `auto` | `auto`, `local`, `cloud`, `gdrive` |
| `-t, --threshold` | `0.75` | Rows below this confidence go to the review queue |
| `--model` | `perch_8` | perch-hoplite preset; must match training |
| `--limit` | all | Process only the first N files |
| `--cache-dir` | system temp | Where remote files are downloaded |

The scan is recursive and picks up `.wav` and `.flac` files. A file that can't
be downloaded or decoded, or a corrupted one, is logged with `status=error` and
an `error_message`, and the batch continues. The exit code is 0 if at least one
file succeeded, 1 if none did, and 2 if the run could not start.

**Sources**
- **Local**: read in place.
- **S3 / GCS**: listed with `fsspec` and downloaded one file at a time into the
  cache. Standard credentials apply (`AWS_PROFILE`,
  `GOOGLE_APPLICATION_CREDENTIALS`, …). Public buckets also work without credentials.
- **Google Drive**: the folder must be shared as *Anyone with the link*. It is
  downloaded once with `gdown`, which lists at most 50 files per folder. For
  larger collections, use sub-folders or a bucket.

## 4. Outputs

`outputs/predictions_manifest.csv` has one row per file:

| Column | Meaning |
|---|---|
| `filename`, `file_path` | File name and its original location (path or URI) |
| `predicted_species_label` | Best class, or `no_detection` |
| `confidence_score` | Confidence in that label (0–1, see below) |
| `status` | `ok` or `error` |
| `relative_path`, `local_path` | Path under the source root, and the local or cached copy |
| `top_class`, `top_class_probability` | Highest-scoring class, even when below 0.5 |
| `best_window_start_s`, `best_window_end_s` | The 5 s window where that class peaks |
| `num_windows`, `error_message`, `processed_at` | Diagnostics |

**How confidence is computed.** The notebook trains one sigmoid output per class.
Each 5 s window gets a probability per class, and each class keeps its highest
window probability. If the best class reaches 0.5, the file gets that label and
the confidence is that probability. If not, the label is `no_detection` and the
confidence is `1 − p`. Either way, values near 0.5 mean the model is unsure.
Those clips are the most useful ones for a person to label (uncertainty sampling).

`outputs/review_queue.csv` has the `ok` rows with `confidence_score < threshold`,
least confident first. The dashboard rebuilds this file whenever you move the
threshold slider.

Rows are written to the manifest as soon as each file finishes, so an
interrupted run keeps everything processed up to that point.

## 5. Use the dashboard

```bash
streamlit run app.py
```

- **Sidebar**: pick the source type, path or URI, classifier file, output
  folder, confidence threshold and annotator ID.
- **Run Batch Inference**: shows a progress bar and a live log. The model is
  loaded once and reused between runs.
- **Batch Summary tab**: files processed, high-confidence count, review-queue
  size, mean confidence, a bar chart of confident detections by species, the
  failed files, and the full manifest for download.
- **Human Review Queue tab**: each uncertain clip has an audio player, starting
  at the window that drove the prediction. Choose the correct label and press
  *Save label*. Verified labels are added to `outputs/verified_labels.csv`.

The dashboard reads from the output folder, so it also shows the results of
runs started from the command line.

## 6. Retrain with the verified labels

`verified_labels.csv` has one row per labelled window: `filename, start_time_s,
end_time_s, label, label_type` (`positive` or `negative`), plus provenance
columns. Each choice in the review form becomes rows like this:

| Expert choice | Rows written |
|---|---|
| Same species as the model | `positive` for that species |
| Another species | `positive` for that species, `negative` for the model's species |
| `no_detection` | `negative` for the model's species |

To use the labels in the tutorial, copy `verified_labels.csv` into Colab and run
this cell before **Prepare the labeled data**. The recordings must already be
embedded in the notebook's database:

```python
!pip install -q -e /path/to/this/repo   # or add the repo to sys.path
from src.utils import ingest_verified_labels

counts = ingest_verified_labels('/content/verified_labels.csv', db,
                                dataset_name='powdermill')
print(counts)
```

Then continue with **Prepare the labeled data** and **Train the linear classifier**
as usual. Labels are matched to stored windows by `filename`, which is the path
relative to the source root (for example `Recording_2/Recording_2_Segment_01.wav`).
To keep the names consistent, run the pipeline on the same root folder the
notebook embedded (`/content/powdermill` or
`gs://chirp-public-bucket/soundscapes/powdermill/`).

## 7. Tests

```bash
pytest
```

The tests swap the Perch model for a small fake, so they run in seconds with no
GPU, network or `perch-hoplite`. They cover recursive scanning, source
detection, corrupted-file handling, classifier save/load, CSV outputs, queue
splitting and label export.
