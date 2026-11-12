# Diabetes Risk Prediction

A machine learning web service that estimates the probability that a patient has diabetes from routine clinical measurements.

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://www.docker.com/)

---

## 1. Problem description

Diabetes is a chronic disease that affects hundreds of millions of people, and many cases are diagnosed late because the early symptoms are easy to ignore. A model that flags high-risk patients using inexpensive clinical measurements can help a clinic decide who should take a confirmatory blood test first.

**Goal:** predict the binary target `Outcome` (1 = diabetes, 0 = no diabetes) from 8 clinical features.

**How the prediction is used:** the doctor fills a small web form with the patient's values and immediately gets a risk probability and a `high risk` / `low risk` label. It is a **screening aid**, not a medical diagnosis.

**Why a machine learning model?** The relationship between glucose, BMI, age, insulin and diabetes is not a simple threshold. A model can combine all the measurements at the same time and give a calibrated probability that is easier to act on than a set of separate cut-offs.

---

## 2. Dataset

**Pima Indians Diabetes Database** — originally from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), distributed as a public dataset.

* File: [`data/diabetes.csv`](data/diabetes.csv)
* Source: <https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv>
* Rows: **768** patients (women of Pima Indian heritage, 21+ years old)
* Target: `Outcome` (1 = diabetes, 0 = no diabetes) — **34.9% positive rate**
* Licence: public domain / freely redistributable for educational use

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test, mg/dL) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Family-history score of diabetes |
| `Age` | Age in years |

> **Important data detail:** in this dataset a `0` in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` or `BMI` does **not** mean a real measurement — it means *"not measured"*. Those zeros are replaced by `NaN` and imputed with the median of the training set inside the model pipeline. The API follows the same convention: send `0` for a value that was not measured.

---

## 3. Results

Three models were compared with 5-fold stratified cross-validation (metric: **ROC AUC**) and tuned with `GridSearchCV`. The test set (20% of the data, 154 patients) was never used during training or tuning.

| Model | Best hyperparameters | CV ROC AUC | Test ROC AUC | Test accuracy | Test F1 |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | `C=0.1`, `class_weight="balanced"` | **0.8450** | **0.8104** | 0.7078 | 0.6154 |
| Random Forest | `n_estimators=400`, `max_depth=4`, `min_samples_leaf=3` | 0.8389 | 0.7996 | 0.7273 | 0.5435 |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.05`, `max_depth=2` | 0.8297 | 0.8148 | 0.7208 | 0.5657 |

The logistic regression was selected because it had the best cross-validation score. With `class_weight="balanced"` it also has the best **recall** on the positive class (0.67), which is the metric that matters most in a screening tool: we prefer a false alarm over missing a sick patient.

> Full metrics are stored in [`model/metrics.json`](model/metrics.json) and reproduced in [`notebook.ipynb`](notebook.ipynb).

### Screenshots

| Input form (`/`) | Prediction result (`POST /predict`) |
|---|---|
| ![Input form](docs/form.png) | ![Prediction result](docs/prediction.png) |

---

## 4. Project structure

```
diabetes-risk-prediction/
├── README.md                  # This file
├── notebook.ipynb             # EDA + model comparison + tuning + evaluation (executed)
├── train.py                   # Trains and tunes the models, saves the pipeline
├── predict.py                 # Flask web service (HTML form + JSON API)
├── download_data.py           # Re-downloads the dataset (optional)
├── requirements.txt           # Pinned dependencies
├── Dockerfile                 # Container image for the web service
├── render.yaml                # One-click deploy configuration for Render
├── data/
│   └── diabetes.csv           # Dataset (committed)
└── model/
    ├── diabetes_pipeline.joblib  # Imputer + scaler + classifier (committed)
    └── metrics.json              # Metrics of every tuned model
```

---

## 5. How to run

### 5.1 Create and activate a virtual environment

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 5.2 Install the dependencies

```bash
pip install -r requirements.txt
```

### 5.3 (Optional) download the dataset

The dataset is already committed in `data/diabetes.csv`. To refresh it from the original source:

```bash
python download_data.py
```

### 5.4 Train the model

```bash
python train.py
```

Expected output (it takes ~1 minute):

```
Loading data...
Rows: 768 | Features: 8 | Positive rate: 34.90%
Tuning logistic_regression...
Tuning random_forest...
Tuning gradient_boosting...
Best model: logistic_regression (CV ROC AUC = 0.8450)
Model saved to:   .../model/diabetes_pipeline.joblib
```

This rewrites `model/diabetes_pipeline.joblib` and `model/metrics.json`.

### 5.5 Explore the notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook is already executed, so the plots and outputs are visible on GitHub/nbviewer. Re-run all the cells if you want to reproduce them (`Kernel → Restart & Run All`).

### 5.6 Run the web service locally

```bash
python predict.py
```

Open **<http://localhost:9696>** in your browser, or call the JSON API:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
       "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}'
```

```json
{
  "prediction": 1,
  "risk_label": "high risk",
  "risk_probability": 0.8005,
  "threshold": 0.5,
  "model": "logistic_regression"
}
```

PowerShell equivalent:

```powershell
$body = @{Pregnancies=6; Glucose=148; BloodPressure=72; SkinThickness=35; Insulin=0;
          BMI=33.6; DiabetesPedigreeFunction=0.627; Age=50} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:9696/predict -Method Post -Body $body -ContentType "application/json"
```

Health check:

```bash
curl http://localhost:9696/health
```

---

## 6. Docker

Build the image:

```bash
docker build -t diabetes-risk-prediction .
```

Run the container (the service listens on port 9696):

```bash
docker run --rm -p 9696:9696 diabetes-risk-prediction
```

Then open <http://localhost:9696> or repeat the `curl` call above.

The image only copies `predict.py` and `model/`, so it does not include the notebook, the training data or the dev dependencies.

---

## 7. Deploy to the cloud (Render)

The service reads the standard `$PORT` environment variable, so it works on Render, Railway, Fly.io or any container host. [`render.yaml`](render.yaml) contains the configuration.

1. Push this repository to GitHub.
2. Create a free account at <https://render.com> and choose **New → Blueprint** (or **New → Web Service** and select the repo).
3. Render detects the `Dockerfile` and the `render.yaml` automatically.
4. Wait for the build and copy the public URL (`https://<name>.onrender.com`).
5. Test it with `curl https://<name>.onrender.com/health`.

Add the resulting URL to the top of this README once it is live.

---

## 8. How the model was built (summary)

1. **EDA** — class imbalance, impossible zeros, distributions, boxplots by class, correlation matrix.
2. **Cleaning** — zeros → `NaN` in the five affected columns.
3. **Pipeline** — median imputation + standard scaling + classifier, fitted only on the training split to avoid leakage.
4. **Model selection** — logistic regression, random forest and gradient boosting compared with 5-fold stratified cross-validation.
5. **Tuning** — `GridSearchCV` over `C`, `class_weight`, tree depth, number of estimators and learning rate.
6. **Evaluation** — accuracy, precision, recall, F1, ROC AUC and confusion matrix on the held-out test set.
7. **Export** — `train.py` reproduces the training; the pipeline is saved with `joblib`.
8. **Serving** — `predict.py` (Flask) exposes an HTML form, a JSON API and a health endpoint; `Dockerfile` containerizes it.

---

## 9. Limitations

* Only **768 rows**, all from **women of Pima Indian heritage aged 21+**. The model should not be applied to other populations without retraining.
* The dataset is old and some values are self-reported; the `Insulin` column is missing in almost half of the rows.
* The model is a **screening aid**. It must not replace a doctor or a laboratory test.
* No calibration study was performed, so the probability should be read as a relative risk score rather than an exact clinical probability.

---

## 10. Author

**Alvaro Andres Salazar Valdez** — Machine Learning Zoomcamp 2026, midterm project.

Dataset: Pima Indians Diabetes Database (NIDDK), distributed via the Plotly datasets repository.
