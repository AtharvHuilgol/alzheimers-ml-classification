"""SHAP-based model explainability functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import ensure_dir, results_path


def select_explainer(model: Any, X_background: pd.DataFrame) -> Any:
    """Select the appropriate SHAP explainer based on model type."""
    if isinstance(model, (RandomForestClassifier,)):
        return shap.TreeExplainer(model)
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_background)
    return shap.KernelExplainer(model.predict_proba, X_background)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    X_background: pd.DataFrame | None = None,
) -> tuple[Any, np.ndarray]:
    """Compute SHAP values for a fitted model."""
    background = X_background if X_background is not None else X
    explainer = select_explainer(model, background)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class for binary classification

    return explainer, np.asarray(shap_values)


def plot_global_importance(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Path | str | None = None,
    plot_type: str = "bar",
) -> None:
    """Generate global SHAP feature importance plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
    plt.tight_layout()

    if save_path:
        save_path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """Generate SHAP beeswarm (dot) plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()

    if save_path:
        save_path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dependence(
    feature: str,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """Generate SHAP dependence plot for a single feature."""
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(feature, shap_values, X, show=False)
    plt.tight_layout()

    if save_path:
        save_path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_waterfall(
    explainer: Any,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    index: int,
    save_path: Path | str | None = None,
) -> None:
    """Generate SHAP waterfall plot for an individual prediction."""
    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=expected_value,
        data=X.iloc[index],
        feature_names=X.columns.tolist(),
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()

    if save_path:
        save_path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_top_features(shap_values: np.ndarray, feature_names: list[str], k: int = 5) -> list[str]:
    """Return top-k features by mean absolute SHAP value."""
    importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:k]
    return [feature_names[i] for i in top_idx]


def run_shap_analysis(
    model: Any,
    X_test: pd.DataFrame,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run full SHAP analysis pipeline and save figures."""
    output_dir = Path(output_dir or results_path("figures"))
    ensure_dir(output_dir)

    explainer, shap_values = compute_shap_values(model, X_test)
    top_features = get_top_features(shap_values, X_test.columns.tolist(), k=3)

    plot_global_importance(
        shap_values,
        X_test,
        save_path=output_dir / "shap_global_bar.png",
        plot_type="bar",
    )
    plot_beeswarm(shap_values, X_test, save_path=output_dir / "shap_beeswarm.png")

    for feature in top_features:
        plot_dependence(
            feature,
            shap_values,
            X_test,
            save_path=output_dir / f"shap_dependence_{feature}.png",
        )

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "top_features": top_features,
    }
