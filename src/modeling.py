"""Train, evaluate, and compare classification models."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import SEED

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def get_models() -> dict[str, Any]:
    """Return the required classification algorithms."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=SEED,
            use_label_encoder=False,
        ),
        "SVM": SVC(probability=True, random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(max_iter=500, random_state=SEED),
    }


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def build_smote_pipeline(model: Any) -> ImbPipeline:
    """Build pipeline with SMOTE applied only inside CV folds."""
    return ImbPipeline(
        [
            ("smote", SMOTE(random_state=SEED)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_model_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool = True,
) -> dict[str, float]:
    """Evaluate a model with stratified 5-fold CV; return mean metrics."""
    estimator = build_smote_pipeline(model) if use_smote else model
    start = time.time()

    auc_scores = cross_val_score(estimator, X, y, cv=CV, scoring="roc_auc", n_jobs=-1)
    fit_time = time.time() - start

    # Per-fold detailed metrics require manual CV loop
    fold_metrics: dict[str, list[float]] = {
        "auc_roc": [],
        "sensitivity": [],
        "specificity": [],
        "f1": [],
        "mcc": [],
    }

    for train_idx, test_idx in CV.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if use_smote:
            pipe = build_smote_pipeline(model)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            pipe = ImbPipeline([("scaler", StandardScaler()), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

        fold_metrics["auc_roc"].append(roc_auc_score(y_test, y_proba))
        fold_metrics["sensitivity"].append(recall_score(y_test, y_pred))
        fold_metrics["specificity"].append(specificity_score(y_test.values, y_pred))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, average="weighted"))
        fold_metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))

    return {
        "auc_roc_mean": np.mean(fold_metrics["auc_roc"]),
        "auc_roc_std": np.std(fold_metrics["auc_roc"]),
        "sensitivity_mean": np.mean(fold_metrics["sensitivity"]),
        "sensitivity_std": np.std(fold_metrics["sensitivity"]),
        "specificity_mean": np.mean(fold_metrics["specificity"]),
        "specificity_std": np.std(fold_metrics["specificity"]),
        "f1_mean": np.mean(fold_metrics["f1"]),
        "f1_std": np.std(fold_metrics["f1"]),
        "mcc_mean": np.mean(fold_metrics["mcc"]),
        "mcc_std": np.std(fold_metrics["mcc"]),
        "fit_time_s": fit_time,
        "auc_fold_scores": fold_metrics["auc_roc"],
    }


def compare_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool = True,
) -> pd.DataFrame:
    """Train and evaluate all required models; return results table."""
    results = []
    for name, model in get_models().items():
        metrics = evaluate_model_cv(model, X, y, use_smote=use_smote)
        results.append(
            {
                "Model": name,
                "AUC-ROC": f"{metrics['auc_roc_mean']:.4f} ± {metrics['auc_roc_std']:.4f}",
                "Sensitivity": f"{metrics['sensitivity_mean']:.4f} ± {metrics['sensitivity_std']:.4f}",
                "Specificity": f"{metrics['specificity_mean']:.4f} ± {metrics['specificity_std']:.4f}",
                "F1": f"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}",
                "MCC": f"{metrics['mcc_mean']:.4f} ± {metrics['mcc_std']:.4f}",
                "Fit Time (s)": round(metrics["fit_time_s"], 2),
            }
        )
    return pd.DataFrame(results)


def compare_models_wilcoxon(
    scores_a: list[float],
    scores_b: list[float],
) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired model AUC scores across CV folds."""
    stat, p_value = wilcoxon(scores_a, scores_b)
    return float(stat), float(p_value)


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )


def evaluate_on_holdout(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_smote: bool = True,
) -> dict[str, float]:
    """Fit on training data and evaluate on holdout test set."""
    pipe = build_smote_pipeline(model) if use_smote else model
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "sensitivity": recall_score(y_test, y_pred),
        "specificity": specificity_score(y_test.values, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "brier": brier_score_loss(y_test, y_proba),
    }
