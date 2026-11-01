# Dataset

This project uses the **PlantVillage** dataset (public, ~54,000 labeled leaf images
across 38 crop/disease classes).

## Getting the data
```bash
bash scripts/download_data.sh
```
This downloads and extracts the dataset into `data/raw/`.

## Structure after download
```
data/raw/
└── PlantVillage/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Corn___Common_rust/
    ├── Tomato___Late_blight/
    └── ... (38 classes total)
```

## Train/validation/test splits
The notebook (`notebooks/plant-disease-classifier.ipynb`) creates reproducible,
stratified train/validation/test subsets directly from the raw class folders. It
does not copy or modify image files. The training configuration uses the same
source directory by default; set `PLANTVILLAGE_PATH` to override it.

## Notes
- Document class imbalance, image quality issues, and any mislabeled samples found
  during Week 2 exploration directly in the notebook.
- Keep raw data untouched; preprocessing and splits are applied when images are loaded.
