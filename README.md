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
