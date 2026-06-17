from pathlib import Path
import pandas as pd
from src.cleaning import clean_train, clean_test


# PATH DEFINITIONS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "data_raw"

MODEL_DATA_DIR = DATA_DIR / "data_model"
MODEL_DATA_DIR.mkdir(exist_ok=True)


def main():
    df_train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    df_test = pd.read_csv(RAW_DATA_DIR / "test.csv")

    df_train_clean, state = clean_train(df_train)
    df_test_clean = clean_test(df_test, state)

    df_train_clean.to_csv(MODEL_DATA_DIR / "train_model.csv", index=False)
    df_test_clean.to_csv(MODEL_DATA_DIR / "test_model.csv", index=False)

    print(f"Train: {df_train.shape} → {df_train_clean.shape}")
    print(f"Test:  {df_test.shape} → {df_test_clean.shape}")


if __name__ == "__main__":
    main()
