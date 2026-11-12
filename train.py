"""
Train a model that predicts diabetes risk (binary classification).

The script:
  1. Loads the Pima Indians Diabetes dataset.
  2. Cleans it (biologically impossible zeros are treated as missing values).
  3. Compares three models with 5-fold cross-validation.
  4. Tunes the hyperparameters of every model with GridSearchCV.
  5. Evaluates the best model on a held-out test set.
  6. Saves the full pipeline and the metrics to disk.

Usage:
    python train.py

Outputs:
    model/diabetes_pipeline.joblib
    model/metrics.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "diabetes_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# In this dataset a zero in these columns means "not measured", not a real value:
# a living person cannot have 0 glucose, 0 blood pressure, 0 BMI, etc.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it with: python download_data.py"
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN so the imputer can handle them."""
    df = df.copy()
    for column in ZERO_AS_MISSING:
        if column in df.columns:
            df[column] = df[column].replace(0, np.nan)
    return df


def build_pipeline(estimator) -> Pipeline:
    """Impute missing values, scale the features and attach the classifier."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def get_model_grids() -> dict[str, dict]:
    """Return the candidate models and the hyperparameters to tune."""
    return {
        "logistic_regression": {
            "pipeline": build_pipeline(
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
            ),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "pipeline": build_pipeline(
                RandomForestClassifier(random_state=RANDOM_STATE)
            ),
            "params": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 4, 8],
                "classifier__min_samples_leaf": [1, 3],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "gradient_boosting": {
            "pipeline": build_pipeline(
                GradientBoostingClassifier(random_state=RANDOM_STATE)
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        },
    }


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute the metrics used to compare models on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = clean_data(load_data())
    X = df[FEATURES]
    y = df[TARGET]

    print(f"Rows: {len(df)} | Features: {len(FEATURES)} | Positive rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}
    best_name, best_score, best_model = None, -np.inf, None

    for name, config in get_model_grids().items():
        print(f"\nTuning {name}...")
        search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        results[name] = {
            "best_params": {
                key.replace("classifier__", ""): value
                for key, value in search.best_params_.items()
            },
            "cv_roc_auc": round(float(search.best_score_), 4),
            "test": test_metrics,
        }
        print(f"  CV ROC AUC : {search.best_score_:.4f}")
        print(f"  Test ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  Best params: {results[name]['best_params']}")

        if search.best_score_ > best_score:
            best_name, best_score, best_model = name, search.best_score_, search.best_estimator_

    print(f"\nBest model: {best_name} (CV ROC AUC = {best_score:.4f})")
    print("\nClassification report on the test set:")
    print(classification_report(y_test, best_model.predict(X_test), digits=4))

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "threshold": 0.5,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_metrics": results[best_name]["test"],
    }
    joblib.dump(bundle, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {"best_model": best_name, "models": results},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nModel saved to:   {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
