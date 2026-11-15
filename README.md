# Agile Modeling for Bioacoustic Monitoring
This tutorial presents an "agile modeling" approach that enables users to build custom classifier systems efficiently for species of interest using transfer learning, audio search, and human-in-the-loop active learning.

Authors (Equal Contribution):
* Jenny Hamer, Google DeepMind, hamer@google.com
* Rob Laber, Google Cloud, roblaber@google.com
* Tom Denton, Google Research, tomdenton@google.com

Revised by:
*   Tariq Shahzad, COMSATS University Islamabad, tariqshahzadd@gmail.com

Originally presented at NeurIPS 2023 (view the poster [here](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1-FHYqaY4Y0vA0P3lvcyINFDFlKWuKMw4%2Fview%3Fusp%3Ddrive_link))

Revised for 2026 Summer School

## Access this tutorial

We recommend executing this notebook in a Colab environment to gain access to GPUs and to manage all necessary dependencies. <a target="_blank" href="https://colab.research.google.com/github/climatechange-ai-tutorials/bioacoustic-monitoring/blob/main/Agile_Modeling_for_Bioacoustic_Monitoring.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Estimated time to execute end-to-end: 45 minutes 

## Beyond the notebook: batch inference and review

The repository also includes optional tools for using a classifier you trained in the notebook on new recordings. The notebook works without them. Full details are in [README_EXTENDED.md](README_EXTENDED.md).

1. **Install** (not needed for the notebook itself):
   ```bash
   pip install -r requirements.txt
   ```
2. **Save your classifier.** The notebook's "Show the metrics" cell writes it to `<db_path>/agile_classifier.pt`. On Colab, download that file.
3. **Run batch inference** on a local folder, an `s3://` or `gs://` bucket, or a public Google Drive folder:
   ```bash
   python -m src.pipeline --input gs://chirp-public-bucket/soundscapes/powdermill/ --classifier agile_classifier.pt
   ```
   This writes `outputs/predictions_manifest.csv` (one row per file) and `outputs/review_queue.csv` (files with confidence below `--threshold`, default 0.75). Corrupted or unreadable files are marked `status=error`, and the rest of the batch continues.
4. **Review uncertain clips in the dashboard**:
   ```bash
   streamlit run app.py
   ```
   Choose the source and threshold, then run the batch. Listen to each queued clip and save the correct label. Verified labels go to `outputs/verified_labels.csv`.
5. **Retrain.** In the notebook, before "Prepare the labeled data", load the verified labels into the database and train as usual:
   ```python
   from src.utils import ingest_verified_labels
   ingest_verified_labels('verified_labels.csv', db, dataset_name='powdermill')
   ```

## Contribute to this tutorial

Please refer to these [GitHub instructions](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#about-forking) to open a pull request via the "fork and pull request" workflow. 

Pull requests will be reviewed by members of the Climate Change AI Tutorials team for relevance, accuracy, and conciseness.

## Climate Change AI Tutorials
Check out the [tutorials page](https://www.climatechange.ai/tutorials?) on our website for a full list of tutorials demonstrating how AI can be used to tackle problems related to climate change.

## License
Usage of this tutorial is subject to the MIT License.

## Cite

### Plain Text
Hamer, J., Laber, R., Denton, T., & Shahzad, T. (2026). Agile Modeling for Bioacoustic Monitoring [Tutorial]. In Climate Change AI Summer School. Climate Change AI. https://doi.org/10.5281/zenodo.21955055

### BibTeX

```
@misc{hamer2026agile,
  title={Agile Modeling for Bioacoustic Monitoring},
  author={Hamer, Jenny and Laber, Rob and Denton, Tom and Shahzad, Tariq},
  year={2026},
  organization={Climate Change AI},
  type={Tutorial},
  doi={https://doi.org/10.5281/zenodo.21955055},
  booktitle={Climate Change AI Summer School},
  howpublished={\url{https://github.com/climatechange-ai-tutorials/bioacoustic-monitoring}}
}
```
