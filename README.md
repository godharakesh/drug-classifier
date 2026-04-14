# Drug Classifier — CI/CD for Machine Learning

An end-to-end MLOps project that automatically trains, evaluates, versions, and deploys a drug classification model using **GitHub Actions**, **CML**, and **Hugging Face Spaces**.

**Live Demo:** [starnek/Drug-Classification](https://huggingface.co/spaces/starnek/Drug-Classification)

---

## Overview

Every push to `main` triggers a full CI/CD pipeline:

1. **Train** — fits a scikit-learn Random Forest on patient vitals
2. **Evaluate** — generates metrics and a confusion matrix, posted as a commit comment via CML
3. **Version** — saves updated model and results to the `update` branch
4. **Deploy** — uploads the app, model, and results to Hugging Face Spaces

---

## Project Structure

```
├── .github/workflows/
│   ├── ci.yml          # Train → Evaluate → Version
│   └── cd.yml          # Deploy to Hugging Face
├── App/
│   ├── drug_app.py     # Gradio web app
│   ├── README.md       # Hugging Face Space metadata
│   └── requirements.txt
├── Data/
│   └── drug.csv        # Drug200 dataset
├── Model/              # Saved scikit-learn pipeline (.skops)
├── Results/            # Metrics and confusion matrix
├── train.py            # Training script
├── Makefile            # Automation commands
└── requirements.txt
```

---

## Dataset

[Drug Classification](https://www.kaggle.com/datasets/prathamtripathi/drug-classification) — 200 patient records with features:

| Feature | Description |
|---------|-------------|
| Age | Patient age (15–74) |
| Sex | M / F |
| BP | Blood pressure (HIGH / LOW / NORMAL) |
| Cholesterol | HIGH / NORMAL |
| Na_to_K | Sodium-to-potassium ratio in blood |
| **Drug** | Target: DrugY, drugA, drugB, drugC, drugX |

---

## Model

A **scikit-learn Pipeline** combining:
- `ColumnTransformer` — ordinal encoding + median imputation + standard scaling
- `RandomForestClassifier` — 100 estimators

---

## CI Pipeline

```yaml
push to main → Install → Format → Train → Evaluate (CML report) → Save to update branch
```

The CML report (metrics + confusion matrix) is posted automatically as a commit comment.

---

## CD Pipeline

```yaml
CI completes → Pull update branch → Login to HF → Upload App / Model / Results
```

Triggered automatically after every successful CI run.

---

## Setup

### 1. Fork & clone this repo

```bash
git clone https://github.com/godharakesh/drug-classifier.git
cd drug-classifier
```

### 2. Add GitHub repository secrets

| Secret | Value |
|--------|-------|
| `HF` | Hugging Face write token |

### 3. Create a Hugging Face Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space), choose **Gradio** SDK, then update the Space name in `Makefile`:

```makefile
push-hub:
    hf upload YOUR_USERNAME/YOUR_SPACE_NAME ./App ...
```

### 4. Create the `update` branch on GitHub

```bash
git checkout -b update
git push origin update
git checkout main
```

### 5. Push to trigger the pipeline

```bash
git push origin main
```

---

## Run Locally

```bash
pip install -r requirements.txt
python train.py
python App/drug_app.py
```

App runs at `http://127.0.0.1:7860`.

---

## Tech Stack

- **scikit-learn** — model training
- **skops** — model serialization
- **Gradio** — web app
- **CML** — CI evaluation reports
- **GitHub Actions** — CI/CD automation
- **Hugging Face Spaces** — deployment
