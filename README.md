# ANN for Binary Classification (Pima Indians Diabetes)

This repository contains helper files and a training script for an artificial neural network (ANN) binary classification example using the Pima Indians Diabetes dataset.

## Contents
- `scripts/train.py` — CLI training script with preprocessing, model building, callbacks, and evaluation.
- `requirements.txt` — Python dependencies.
- `.gitignore` — files/folders to ignore in Git.
- `README.md` — this file.

## Quickstart (local)
1. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate    # Windows PowerShell
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run a smoke test (fast, single epoch, small subset):
```bash
python scripts/train.py --smoke-test
```

4. Train normally (default 20 epochs):
```bash
python scripts/train.py --epochs 50 --batch-size 32
```

## Notes & improvements included
- Zero-value handling for physiologically impossible columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
- Median imputation for those columns.
- Train/val/test split with stratification.
- StandardScaler for features (fit on train only).
- Reproducibility via `numpy` and `tensorflow` seeds.
- Callbacks: `EarlyStopping` and `ModelCheckpoint`.
- Evaluation: classification report, confusion matrix, ROC AUC.
- CLI flags: `--epochs`, `--batch-size`, `--smoke-test`, `--model-out`.

## Dataset
The script downloads the dataset from a public raw GitHub URL. If you prefer to use a local CSV, pass `--data-path path/to/file.csv` to the script.

## License
Choose a license (e.g., MIT) and add a `LICENSE` file if needed.
