# regression_models/decision_tree_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def grid_search_decision_tree(X, y):
    """
    Pipeline: SMOTE -> DecisionTree (підбір max_depth, min_samples_leaf, class_weight).
    """
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('tree', DecisionTreeClassifier(random_state=42))
    ])
    param_grid = {
        'tree__max_depth': [3, 5, 7, 10, None],
        'tree__min_samples_leaf': [1, 5, 10],
        'tree__class_weight': [None, 'balanced']
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=skf, n_jobs=-1)
    grid.fit(X, y)
    print("="*70)
    print("=== [4] GRID SEARCH DecisionTree (with SMOTE) ===")
    print("Best params:", grid.best_params_)
    print(f"Best AUC(CV) = {grid.best_score_:.4f}")
    print("="*70 + "\n")
    return grid.best_estimator_