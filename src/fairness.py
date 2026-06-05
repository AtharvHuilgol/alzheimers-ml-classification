"""Subgroup evaluation and fairness metrics."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from src.modeling import specificity_score
from src.utils import ensure_dir, results_path


def assign_ses_group(ses: float) -> str:
    """Map SES to Low / Medium / High groups."""
    if pd.isna(ses):
        return "Unknown"
    if ses <= 2:
        return "Low (1-2)"
    if ses == 3:
        return "Medium (3)"
    return "High (4-5)"


def assign_education_group(educ: float) -> str:
    """Map years of education to Low / High groups."""
    if pd.isna(educ):
        return "Unknown"
    return "Low EDUC (<12)" if educ < 12 else "High EDUC (>=12)"


def add_fairness_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add subgroup columns used for fairness analysis."""
    df = df.copy()
    df["GenderGroup"] = df["Gender"].map({1: "Male", 0: "Female"})
    df["SESGroup"] = df["SES"].apply(assign_ses_group)
    df["EducationGroup"] = df["EDUC"].apply(assign_education_group)
    return df


def evaluate_subgroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute performance metrics for a single subgroup."""
    metrics: dict[str, float] = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["auc_roc"] = np.nan
    return metrics


def evaluate_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "target",
    y_pred_col: str = "y_pred",
    y_proba_col: str | None = "y_proba",
) -> pd.DataFrame:
    """Evaluate metrics per subgroup."""
    rows = []
    for group_name, group_df in df.groupby(group_col):
        y_true = group_df[y_true_col].values
        y_pred = group_df[y_pred_col].values
        y_proba = group_df[y_proba_col].values if y_proba_col else None
        metrics = evaluate_subgroup(y_true, y_pred, y_proba)
        metrics[group_col] = group_name
        rows.append(metrics)
    return pd.DataFrame(rows)


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
) -> dict[str, float]:
    """Compute demographic parity and equalized odds differences."""
    return {
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
    }


def metric_frame_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
) -> MetricFrame:
    """Build a Fairlearn MetricFrame for subgroup comparison."""
    return MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "sensitivity": recall_score,
            "f1": lambda yt, yp: f1_score(yt, yp, average="weighted"),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )


def plot_subgroup_bars(
    results_df: pd.DataFrame,
    group_col: str,
    metrics: list[str] | None = None,
    save_path: Path | str | None = None,
) -> None:
    """Grouped bar chart of metrics by subgroup."""
    metrics = metrics or ["f1", "sensitivity"]
    plot_df = results_df.melt(
        id_vars=[group_col],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x=group_col, y="Score", hue="Metric", palette="colorblind")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_path:
        save_path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fairness_heatmap(
    pivot_df: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """Heatmap of AUC-ROC across a Gender × Age Group grid."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="colorblind", vmin=0, vmax=1)
    plt.tight_layout()

    if save_path:
        save_path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_fairness_analysis(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Run full fairness analysis across defined subgroups."""
    output_dir = Path(output_dir or results_path("figures"))
    ensure_dir(output_dir)

    df = add_fairness_groups(df.copy())
    df["y_pred"] = y_pred
    if y_proba is not None:
        df["y_proba"] = y_proba

    subgroup_results = {}
    for group_col in ["GenderGroup", "AgeGroup", "SESGroup", "EducationGroup"]:
        if group_col not in df.columns:
            continue
        subgroup_results[group_col] = evaluate_by_group(df, group_col)
        plot_subgroup_bars(
            subgroup_results[group_col],
            group_col,
            save_path=output_dir / f"fairness_{group_col}.png",
        )

    fairness_metrics = compute_fairness_metrics(
        df["target"].values,
        y_pred,
        df["GenderGroup"],
    )

    if "AgeGroup" in df.columns and "GenderGroup" in df.columns:
        auc_pivot = df.groupby(["GenderGroup", "AgeGroup"]).apply(
            lambda g: roc_auc_score(g["target"], g["y_proba"])
            if len(g["target"].unique()) > 1 and y_proba is not None
            else np.nan,
            include_groups=False,
        ).unstack()
        plot_fairness_heatmap(auc_pivot, save_path=output_dir / "fairness_auc_heatmap.png")
    else:
        auc_pivot = pd.DataFrame()

    return {
        "subgroup_results": subgroup_results,
        "fairness_metrics": fairness_metrics,
        "auc_pivot": auc_pivot,
    }
