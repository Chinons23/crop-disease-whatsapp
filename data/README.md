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

## Processed splits
Run the EDA notebook (`notebooks/01_eda.ipynb`) to generate train/val/test splits
into `data/processed/`. This directory is gitignored — regenerate it locally rather
than committing images.

## Notes
- Document class imbalance, image quality issues, and any mislabeled samples found
  during Week 2 exploration directly in the EDA notebook.
- Keep raw data untouched; all cleaning happens on the way into `data/processed/`.
