# data_import.py

import pandas as pd

def load_data(file_path: str = "data/fin_data.csv") -> pd.DataFrame:
    """
    Завантажує CSV з фінансовими даними та повертає DataFrame.
    """
    df = pd.read_csv(file_path, sep=';', decimal=',')
    print("\n" + "="*70)
    print("=== [1] ЗАВАНТАЖЕННЯ ДАНИХ ===")
    print(f"Файл: {file_path}")
    print(f"Розмірність: {df.shape[0]} рядків, {df.shape[1]} колонок")
    print("Перші 5 рядків:")
    print(df.head(5))
    print("\nІнформація про колонки та типи даних:")
    print(df.info())
    print("="*70 + "\n")
    return df