# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def remove_outliers_iqr(df: pd.DataFrame, cols=None, iqr_factor=1.5) -> pd.DataFrame:
    """
    Вирізає викиди за допомогою IQR-фільтра (1.5 * IQR за замовчанням).
    """
    if cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(['default'])
        cols = list(numeric_cols)

    df_filtered = df.copy()
    for c in cols:
        Q1 = df_filtered[c].quantile(0.25)
        Q3 = df_filtered[c].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        df_filtered = df_filtered[(df_filtered[c] >= lower) & (df_filtered[c] <= upper)]
    return df_filtered

def prepare_data(df: pd.DataFrame,
                 id_col='edrpou',
                 drop_cols=('edrpou', 'year', 'default'),
                 target_col='default',
                 test_size=0.3,
                 random_state=42):
    """
    1. Вилучає drop_cols із X.
    2. Розділяє на X, y.
    3. Виконує train_test_split + зберігає ідентифікатори компаній.
    4. Стандартизує числові ознаки.
    """
    print("="*70)
    print("=== [2] ПІДГОТОВКА ДАНИХ ===")

    features = df.columns.difference(drop_cols)
    X = df[features].copy()
    y = df[target_col].copy()

    company_ids = df[id_col].copy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)),
        test_size=test_size, random_state=random_state, stratify=y
    )
    train_id = company_ids.iloc[idx_train]
    test_id = company_ids.iloc[idx_test]

    # Масштабування
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Всього після IQR-фільтра: {df.shape[0]} рядків.")
    print(f"train: {len(y_train)} рядків ({100*(1-test_size):.0f}%), test: {len(y_test)} рядків ({100*test_size:.0f}%).")
    print("Кількість дефолтів у train:", y_train.sum(), ", у test:", y_test.sum())
    print("Ознаки (features):", list(features))
    print("="*70 + "\n")
    return X_train_scaled, X_test_scaled, y_train, y_test, train_id, test_id, list(features)