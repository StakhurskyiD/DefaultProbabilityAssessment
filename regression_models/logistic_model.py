# regression_models/logistic_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def grid_search_logistic(X, y):
    """
    Pipeline: SMOTE -> LogisticRegression (підбір C, solver, class_weight) через GridSearchCV.
    """
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('logit', LogisticRegression(max_iter=1000))
    ])
    param_grid = {
        'logit__C': [0.01, 0.1, 1, 10],
        'logit__solver': ['lbfgs', 'liblinear'],
        'logit__class_weight': [None, 'balanced']
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=skf, n_jobs=-1)
    grid.fit(X, y)
    print("="*70)
    print("=== [3] GRID SEARCH LogisticRegression (with SMOTE) ===")
    print("Best params:", grid.best_params_)
    print(f"Best AUC(CV) = {grid.best_score_:.4f}")
    print("="*70 + "\n")
    return grid.best_estimator_

def find_best_threshold(model, X, y, metric='f1'):
    """
    Перебір порогу від 0 до 1 кроком 0.01, шукаємо кращий за метрикою (наприклад, F1).
    """
    from sklearn.metrics import f1_score
    probas = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    best_t = 0.5
    best_score = -1

    for t in thresholds:
        preds = (probas >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y, preds, zero_division=0)
        else:
            score = f1_score(y, preds, zero_division=0)  # Для прикладу

        if score > best_score:
            best_score = score
            best_t = t

    print("="*70)
    print(f"[Threshold search] Best {metric.upper()}={best_score:.4f} at threshold={best_t:.2f}")
    print("="*70 + "\n")
    return best_t, best_score