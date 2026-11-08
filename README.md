---
title: Crop Disease Classifier
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: streamlit
---

# Crop Disease Classifier

A photo-based crop disease classifier for smallholder farmers. The current app is
a **Streamlit web app**: upload or take a photo of one leaf to see its most likely
class and confidence.

The repository also contains a separate **FastAPI/Twilio WhatsApp prototype**.
It is not the app launched by the instructions below and needs additional security
and operational work before public deployment.

## Run the app locally

The app needs a trained model artifact. Model weights are not included in this
repository.

1. Install Python 3.13 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/).
   From the repository root, install the locked dependencies:

   ```bash
   uv sync --locked
   ```

2. Set `MODEL_ARTIFACT` to the path of your trained `.pt` artifact if it is not
   at the app's default location (`output/models/checkpoints/best_model.pt`):

   ```powershell
   # Windows PowerShell
   $env:MODEL_ARTIFACT = "models/checkpoints/best_model.pt"
   ```

   ```bash
   # macOS/Linux
   export MODEL_ARTIFACT=models/checkpoints/best_model.pt
   ```

3. Start the app from the repository root:

   ```bash
   uv run streamlit run app.py
   ```

Open the local URL printed by Streamlit. The app accepts JPG and PNG images up
to 10 MB.

## Train a model

The training data is not included. To download PlantVillage, configure Kaggle
access and run `bash scripts/download_data.sh` in Bash, WSL, or a similar
environment. Alternatively, use an existing class-folder dataset.

Before training, check `data_root` in `configs/train_config.yaml`. It currently
points to a Kaggle dataset path; change it to the location of your dataset when
training locally. Then run:

```bash
uv run python -m main
```

To use another configuration file:

```bash
uv run python -m main --config path/to/train_config.yaml
```

Training saves the artifact to the configured `checkpoint_path` (by default
`models/checkpoints/best_model.pt`). Set `MODEL_ARTIFACT` to that file before
starting the app. Checkpoints are ignored by Git and must be provided separately.

## Train on Kaggle with the CLI

Use Kaggle's hosted notebook compute if you do not have room for PlantVillage or
a GPU locally. The dataset stays on Kaggle; only the trained model needs to be
downloaded to run the app on your computer. Kaggle GPU availability and usage
quotas depend on your account.

1. Install the CLI and authenticate with your Kaggle account:

   ```bash
   pip install kaggle
   kaggle auth login
   ```

   Alternatively, create an API token in your Kaggle account settings and follow
   Kaggle's instructions to configure it locally. Never commit the token or put
   it in a dataset.

2. Create a clean staging folder (for example, `kaggle-source/` at the
   repository root) for the project source dataset. Copy only `main.py`,
   `configs/`, and `src/` into it; do not upload `.env`, `.venv`, `data/`,
   `output/`, model checkpoints, or other private/local files. From that folder,
   initialize and create the dataset:

   ```bash
   kaggle datasets init -p .
   ```

   Edit the generated `dataset-metadata.json` with your Kaggle account and a
   dataset slug, then publish it:

   ```bash
   kaggle datasets create -p .
   ```

   Keep this folder for later updates. After changing the source files, publish
   a new version with `kaggle datasets version -p . -m "Update project code"`.

3. Make a separate folder (for example, `kaggle-kernel/` at the repository
   root), copy `run_kaggle.ipynb` into it, and initialize its metadata:

   ```bash
   kaggle kernels init -p .
   ```

   Edit `kernel-metadata.json`: set `id` to
   `your-kaggle-username/your-kernel-slug`, set `code_file` to
   `run_kaggle.ipynb`, add both your project dataset and
   `abdallahalidev/plantvillage-dataset` to `dataset_sources`, and set
   `enable_gpu` to `"true"`. The checked-in metadata uses this repository
   owner's Kaggle IDs; replace them if you publish under your own account.

   Before pushing, update the notebook's `base` path to your mounted project
   dataset directory (`/kaggle/input/<your-project-dataset-slug>`). In the
   notebook cell that writes `/kaggle/working/train_config.yaml`, set
   `cfg["checkpoint_path"]` to
   `"/kaggle/working/models/checkpoints/best_model.pt"` before saving the
   config. Kaggle input datasets are read-only, so the checkpoint must be
   written under `/kaggle/working` for it to be available as a kernel output.
   The notebook already locates the PlantVillage `color` directory in Kaggle's
   mounted inputs.

4. From the kernel folder, push the notebook to Kaggle and check its run:

   ```bash
   kaggle kernels push -p .
   kaggle kernels status your-kaggle-username/your-kernel-slug
   ```

   Once the run has completed successfully, download its output files to the
   repo's `output/kaggle/` folder (run this command from `kaggle-kernel/`):

   ```bash
   kaggle kernels output your-kaggle-username/your-kernel-slug -p ../output/kaggle
   ```

   Find `best_model.pt` under `output/kaggle/` and set `MODEL_ARTIFACT` to its
   location before running the app. For example, in Windows PowerShell:

   ```powershell
   $env:MODEL_ARTIFACT = "output/kaggle/models/checkpoints/best_model.pt"
   uv run streamlit run app.py
   ```

   If you update the source dataset later, publish its new version first, then
   push the kernel again. Ensure your Kaggle account has access to the
   PlantVillage dataset before running the notebook.

## Deploy on Hugging Face Spaces

1. Create a Space and select the **Streamlit** SDK. This README declares the
   Streamlit SDK for a Space created from this repository.
2. Add a root-level `requirements.txt` to the Space repository; Spaces installs
   dependencies from that file. For this app, include:

   ```text
   streamlit>=1.66.1
   torch>=2.2
   torchvision>=0.17
   pillow>=10.2
   ```

   These are minimum versions, not a fully pinned production lockfile. Pin and
   test the exact versions you deploy.
3. Provide the trained `.pt` artifact to the Space. It is not tracked in this
   repository. In the Space settings, set the `MODEL_ARTIFACT` variable to its
   path in the Space.
4. Restart the Space and check its build logs and app page.

Do not put access tokens or other secrets in this README or commit them to the
repository. Use Space secrets for credentials.

## Production readiness

This repository is suitable for local experimentation and a hosted demo, but it
is **not yet ready for full production use**:

- The configured training data path is Kaggle-specific and must be changed for
  local training.
- PlantVillage images use controlled backgrounds. Test results on that dataset
  do not establish accuracy on real farm photos; field testing is needed.
- The model artifact and training data are not included, so deployment must
  provide them separately.
- The WhatsApp webhook is a separate prototype. Before exposing it publicly,
  add Twilio request-signature verification and safer, bounded media downloads,
  and test its failure handling and capacity.
- Pin and test deployment dependencies, and add deployment monitoring and
  operational checks before serving real users.

The classifier is a guide, not a substitute for advice from an agricultural
expert.

## Tests

Run the test suite from the repository root:

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
