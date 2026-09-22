#!/usr/bin/env bash
# Downloads the PlantVillage dataset into data/raw/.
#
# Recommended: use the Kaggle CLI (requires `pip install kaggle` and a
# ~/.kaggle/kaggle.json API token — see https://www.kaggle.com/docs/api).

set -euo pipefail

DEST="data/raw"
mkdir -p "$DEST"

echo "Downloading PlantVillage dataset via Kaggle..."
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p "$DEST" --unzip

echo "Done. Dataset available at $DEST/"
