"""
Entrena un modelo que predice el riesgo de diabetes (clasificación binaria).

El script:
  1. Carga el dataset de diabetes de los indios Pima.
  2. Lo limpia (los ceros biológicamente imposibles se tratan como valores faltantes).
  3. Compara tres modelos con validación cruzada de 5 pliegues.
  4. Ajusta los hiperparámetros de cada modelo con GridSearchCV.
  5. Evalúa el mejor modelo sobre un conjunto de prueba reservado.
  6. Guarda el pipeline completo y las métricas en disco.

Uso:
    python train.py

Salidas:
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

# En este dataset un cero en estas columnas significa "no medido", no un valor real:
# una persona viva no puede tener 0 de glucosa, 0 de presión arterial, 0 de IMC, etc.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lee el dataset desde el archivo CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}. "
            "Descárgalo con: python download_data.py"
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza los ceros imposibles por NaN para que el imputador los maneje."""
    df = df.copy()
    for column in ZERO_AS_MISSING:
        if column in df.columns:
            df[column] = df[column].replace(0, np.nan)
    return df


def build_pipeline(estimator) -> Pipeline:
    """Imputa los valores faltantes, escala las características y agrega el clasificador."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def get_model_grids() -> dict[str, dict]:
    """Devuelve los modelos candidatos y los hiperparámetros a ajustar."""
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
    """Calcula las métricas que se usan para comparar los modelos en el conjunto de prueba."""
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

    print("Cargando datos...")
    df = clean_data(load_data())
    X = df[FEATURES]
    y = df[TARGET]

    print(f"Filas: {len(df)} | Características: {len(FEATURES)} | Tasa de positivos: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}
    best_name, best_score, best_model = None, -np.inf, None

    for name, config in get_model_grids().items():
        print(f"\nAjustando {name}...")
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
        print(f"  ROC AUC en CV   : {search.best_score_:.4f}")
        print(f"  ROC AUC en prueba: {test_metrics['roc_auc']:.4f}")
        print(f"  Mejores parámetros: {results[name]['best_params']}")

        if search.best_score_ > best_score:
            best_name, best_score, best_model = name, search.best_score_, search.best_estimator_

    print(f"\nMejor modelo: {best_name} (ROC AUC en CV = {best_score:.4f})")
    print("\nReporte de clasificación sobre el conjunto de prueba:")
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

    print(f"\nModelo guardado en:  {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
