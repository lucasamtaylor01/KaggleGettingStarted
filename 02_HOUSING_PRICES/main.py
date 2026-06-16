from pathlib import Path
import pandas as pd
from utils.cleaning import clean_train, clean_test

DATA_DIR = Path('data')


def main():
    df_train = pd.read_csv(DATA_DIR / 'train.csv')
    df_test = pd.read_csv(DATA_DIR / 'test.csv')

    df_train_clean, state = clean_train(df_train)
    df_test_clean = clean_test(df_test, state)

    df_train_clean.to_csv(DATA_DIR / 'train_clean.csv', index=False)
    df_test_clean.to_csv(DATA_DIR / 'test_clean.csv', index=False)

    print(f"Train: {df_train.shape} → {df_train_clean.shape}")
    print(f"Test:  {df_test.shape} → {df_test_clean.shape}")


if __name__ == '__main__':
    main()
