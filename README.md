# Crop Disease Classifier via WhatsApp for Smallholder Farmers

## The Problem
Smallholder farmers across Africa lose significant yields to crop diseases they cannot
identify early. Existing tools require internet access and literacy levels most farmers
do not have.

## What We're Building
A computer vision model trained on the **PlantVillage dataset** that classifies crop
disease from a photo, integrated into a **WhatsApp bot** so any farmer with a basic
smartphone can use it — no app install, no literacy requirement, works over a simple
chat interface.

## Expected Output
One of:
- A working WhatsApp bot
- A web app
- A research paper documenting model performance and deployment approach

## Tech Stack
- **PyTorch** + **Hugging Face `timm`** for the classification model (transfer learning
  on a lightweight backbone — MobileNetV3 / EfficientNet-Lite)
- **Hugging Face Hub** for model versioning/hosting
- **Twilio WhatsApp Business API** (or Meta Cloud API) for the bot webhook
- **FastAPI** for the inference/webhook server

## Repo Structure
```
crop-disease-whatsapp/
├── data/               # dataset (gitignored) — see data/README.md
├── notebooks/          # exploration notebooks
├── src/
│   ├── data/            # Dataset / DataLoader / transforms
│   ├── models/          # model definition + training loop
│   ├── inference/        # single-image prediction
│   └── bot/              # WhatsApp webhook + message handlers
├── models/checkpoints/  # trained weights (gitignored — use HF Hub instead)
├── tests/               # unit tests
├── configs/             # training config (YAML)
├── docs/                # deployment notes
└── scripts/             # data download, demo scripts
```

## Getting Started

```bash
# clone
git clone https://github.com/<your-username>/crop-disease-whatsapp.git
cd crop-disease-whatsapp

# environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# get the dataset
bash scripts/download_data.sh

# train (once src/models/train.py is implemented)
python -m src.models.train --config configs/train_config.yaml

# run inference demo
python scripts/run_inference_demo.py --image path/to/leaf.jpg
```

## Team
| Name | Role | Responsibilities |
|------|------|-------------------|
| Silas | Team Lead | |
| | | |
| | | |

## 4-Week Plan
See the [Notion project page](#) for the full week-by-week plan and progress tracking.

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Kickoff, approach, roles, shared workspace | One-paragraph project plan |
| 2 | Dataset collection & exploration | Progress update + data quality notes |
| 3 | Build & iterate on model | Preliminary results |
| 4 | Finalize & present | Working app / model / paper |

## License
MIT — see [LICENSE](LICENSE)
