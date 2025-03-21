import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('TkAgg')  # або 'Qt5Agg', 'MacOSX' (якщо працює коректно)
import matplotlib.pyplot as plt

def run_default_prediction():
    # 1. Завантаження даних
    file_path = 'data/fin_data.csv'
    data = pd.read_csv(file_path, sep=';', decimal=',')

    # 2. Перевірка даних
    print("Назви колонок:")
    print(data.columns)
    print("\nПерші рядки даних:")
    print(data.head())

    # 3. Підготовка даних
    # Припустимо, що 'default' – цільова змінна
    # Видаляємо неінформативні колонки (наприклад, 'edrpou', 'year' якщо вони вам не потрібні)
    features = data.columns.difference(['edrpou', 'year', 'default'])
    X = data[features]
    y = data['default']

    # 4. Розподіл на тренувальну та тестову вибірки
    # test_size=0.3 означає, що 30% даних підуть у тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5. Стандартизація ознак
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Навчання логістичної регресії
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)

    # 7. Оцінка моделі
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]  # ймовірність приналежності до класу 1

    # 7.1. Classification report та матриця змішування
    print("\nClassification report (Test set):")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (Test set):")
    print(confusion_matrix(y_test, y_pred))

    # 7.2. Коефіцієнти моделі
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": clf.coef_[0]
    })
    print("\nКоефіцієнти моделі:")
    print(coef_df)
    print("\nIntercept:", clf.intercept_[0])

    # 7.3. ROC-крива та AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"\nROC AUC: {auc_score:.4f}")

    plt.figure()  # Окремий графік
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-крива для логістичної регресії')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_default_prediction()