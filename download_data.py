"""
Descarga el dataset de diabetes de los indios Pima en data/diabetes.csv.

El dataset ya está incluido en este repositorio, así que solo necesitas este
script si lo borraste o quieres actualizarlo desde la fuente original.

Uso:
    python download_data.py
"""

from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "diabetes.csv"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL} ...")
    df = pd.read_csv(URL)
    df.to_csv(DATA_PATH, index=False)
    print(f"Se guardaron {len(df)} filas en {DATA_PATH}")


if __name__ == "__main__":
    main()
