import os
import numpy as np
import pandas as pd

import matplotlib
# матимемо право не задавати matplotlib.use('TkAgg'), або спробувати інший
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)

# Імпорт модулів з нашого проекту
from data.data_import import load_data
from data.data_preprocessing import remove_outliers_iqr, prepare_data

from regression_models.logistic_model import (
    grid_search_logistic,
    find_best_threshold
)
from regression_models.decision_tree_model import (
    grid_search_decision_tree
)

# -----------------------------------------------
# Функції для побудови графіків (щоб не засмічувати код)
# -----------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Візуалізація матриці змішування через matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    # Додаємо числові значення
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z}", ha='center', va='center', fontweight='bold')
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.show()

def plot_prob_distribution(probas, y_true, title="Predicted probabilities"):
    """
    Гістограма розподілу імовірностей дефолту на тесті.
    Окремо відображаємо 2 гістограми: де actual=0 і де actual=1.
    """
    plt.figure()
    # Поділимо за реальним класом
    prob_0 = probas[y_true==0]
    prob_1 = probas[y_true==1]

    plt.hist(prob_0, bins=20, alpha=0.5, label='Actual=0')
    plt.hist(prob_1, bins=20, alpha=0.5, label='Actual=1')
    plt.xlabel("Probability of Default")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.show()

def run_default_prediction():
    # 1. Завантажити
    df = load_data("data/fin_data.csv")

    # 2. Прибрати викиди
    df = remove_outliers_iqr(df, iqr_factor=1.5)

    # 3. Підготувати дані
    (X_train_scaled, X_test_scaled,
     y_train, y_test,
     train_id, test_id,
     features) = prepare_data(
        df,
        id_col='edrpou',
        drop_cols=('edrpou','year','default'),
        target_col='default',
        test_size=0.3,
        random_state=42
    )

    # 4. Підбір гіперпараметрів
    best_model_logit = grid_search_logistic(X_train_scaled, y_train)
    best_model_tree  = grid_search_decision_tree(X_train_scaled, y_train)

    # 5. Пошук порогу (логістична регресія)
    best_threshold, best_f1 = find_best_threshold(
        best_model_logit, X_train_scaled, y_train, metric='f1'
    )

    # 6. Оцінка обох моделей на тесті
    print("="*70)
    print("=== [5] ОЦІНКА МОДЕЛЕЙ НА ТЕСТІ ===")

    # 6.1 Logistic (з custom threshold)
    logit_proba_test = best_model_logit.predict_proba(X_test_scaled)[:, 1]
    logit_pred_test = (logit_proba_test >= best_threshold).astype(int)

    from sklearn.metrics import classification_report
    auc_logit = roc_auc_score(y_test, logit_proba_test)
    print("\n--- LogisticRegression (Test) ---")
    print(f"Обраний поріг={best_threshold:.2f}")
    print("Classification report:")
    print(classification_report(y_test, logit_pred_test, zero_division=0))
    print(f"AUC={auc_logit:.4f}")

    # Збережемо індивідуальні результати
    df_logit = pd.DataFrame({
        'company_id': test_id.values,
        'prob_default': logit_proba_test,
        'predicted_default': logit_pred_test,
        'actual_default': y_test.values
    })
    os.makedirs("results", exist_ok=True)
    df_logit.to_csv("results/logistic_results.csv", index=False)
    print("[INFO] logistic_results.csv збережено.")

    # Візуалізації
    plot_confusion_matrix(y_test, logit_pred_test, title="Confusion Matrix - Logistic")
    plot_prob_distribution(logit_proba_test, y_test, title="Probability distribution (Logistic)")

    # 6.2 Decision Tree (поріг=0.5)
    tree_proba_test = best_model_tree.predict_proba(X_test_scaled)[:, 1]
    tree_pred_test  = (tree_proba_test >= 0.5).astype(int)

    auc_tree = roc_auc_score(y_test, tree_proba_test)
    print("\n--- DecisionTree (Test) ---")
    print("Поріг=0.50 (стандарт)")
    print("Classification report:")
    print(classification_report(y_test, tree_pred_test, zero_division=0))
    print(f"AUC={auc_tree:.4f}")

    df_tree = pd.DataFrame({
        'company_id': test_id.values,
        'prob_default': tree_proba_test,
        'predicted_default': tree_pred_test,
        'actual_default': y_test.values
    })
    df_tree.to_csv("results/tree_results.csv", index=False)
    print("[INFO] tree_results.csv збережено.")

    plot_confusion_matrix(y_test, tree_pred_test, title="Confusion Matrix - Decision Tree")
    plot_prob_distribution(tree_proba_test, y_test, title="Probability distribution (Tree)")

    # 7. Побудова ROC-кривих
    fpr_l, tpr_l, _ = roc_curve(y_test, logit_proba_test)
    fpr_t, tpr_t, _ = roc_curve(y_test, tree_proba_test)
    plt.figure()
    plt.plot(fpr_l, tpr_l, label=f'Logistic (AUC={auc_logit:.2f}), thr={best_threshold:.2f}')
    plt.plot(fpr_t, tpr_t, label=f'Tree (AUC={auc_tree:.2f}), thr=0.5')
    plt.plot([0,1],[0,1],'--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: Logistic vs. DecisionTree")
    plt.legend()
    plt.show()

    print("\n=== [6] Фінальне порівняння (AUC) ===")
    print(f"Logistic AUC={auc_logit:.4f} (threshold={best_threshold:.2f})")
    print(f"DecisionTree AUC={auc_tree:.4f} (threshold=0.50)")
    print("="*70 + "\n")

    # 8. Пояснення, чому у вихідному файлi ~500 компаній замість початкових ~5000
    print("ПОЯСНЕННЯ:")
    print("1) Частина рядків могла бути видалена через IQR-фільтр (викиди).")
    print("2) Приблизно 70% даних (TRAIN) використовується для навчання, а 30% (TEST) - для перевірки.")
    print("   У результуючому файлі зберігаються лише ті компанії, що пішли у test.")
    print("   Тому якщо первісно було ~5000, то у тесті може лишитись ~1500, а іноді ще менше через фільтрацію.")
    print("="*70)

if __name__ == "__main__":
    run_default_prediction()