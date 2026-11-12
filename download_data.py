"""
Download the Pima Indians Diabetes dataset into data/diabetes.csv.

The dataset is already committed to this repository, so you only need this
script if you deleted it or want to refresh it from the original source.

Usage:
    python download_data.py
"""

from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "diabetes.csv"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    df = pd.read_csv(URL)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
