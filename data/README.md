# Data Folder

## Purpose
Stores the image dataset used for training and evaluating the forgery detection model.

## Structure

| Subfolder | Contents |
|-----------|----------|
| `raw/` | Original CASIA v2.0 dataset images (authentic + tampered) |
| `processed/` | ELA-processed versions of the images |
| `splits/` | Train/validation/test split CSVs and extracted features |

## Important Notes
- The `raw/` and `processed/` folders are excluded from Git (see `.gitignore`) because they are too large
- To set up the dataset, run: `python scripts/prepare_dataset.py`
- Dataset split ratio: 70% train / 15% validation / 15% test (configurable in `config/config.yaml`)
