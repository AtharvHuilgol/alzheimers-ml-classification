"""Clinically motivated feature creation."""

from __future__ import annotations

import pandas as pd


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin age into clinically standard groups."""
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 65, 75, 85, 120],
        labels=["<65", "65-75", "75-85", "85+"],
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features per PRD/TRD specifications."""
    df = df.copy()

    # Brain Atrophy Ratio
    df["BrainAtrophyRatio"] = df["eTIV"] / df["nWBV"]

    # Cognitive Reserve Index
    df["CognitiveReserveIndex"] = df["EDUC"] / (df["Age"] + 1e-5)

    # Age Groups
    df = add_age_group(df)

    # MMSE Severity Bins
    df["MMSESeverity"] = pd.cut(
        df["MMSE"],
        bins=[-1, 17, 23, 30],
        labels=["Moderate", "Mild", "Normal"],
    )

    # Normalized MMSE
    df["MMSE_norm"] = df["MMSE"] / 30.0

    # SES-Education Interaction
    df["SES_EDUC_interaction"] = df["SES"] * df["EDUC"]

    return df


def engineer_longitudinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create OASIS-2 longitudinal features per subject.

    Expects longitudinal data with columns: ID, Visit, CDR, MMSE, MR Delay.
    """
    if "Visit" not in df.columns:
        return df

    records = []
    for subject_id, group in df.groupby("ID"):
        group = group.sort_values("Visit")
        baseline = group.iloc[0]
        record = {"ID": subject_id}

        record["CDR_change"] = group["CDR"].iloc[-1] - baseline["CDR"]
        if len(group) > 1 and "MR Delay" in group.columns:
            record["MMSE_slope"] = (
                group["MMSE"].diff().sum() / group["MR Delay"].diff().sum()
                if group["MR Delay"].diff().sum() != 0
                else 0.0
            )
        else:
            record["MMSE_slope"] = 0.0

        converted = (baseline["CDR"] == 0) & (group["CDR"].max() > 0)
        record["ConversionFlag"] = int(converted)

        if converted and "MR Delay" in group.columns:
            conversion_visits = group[(group["CDR"] > 0) & (group["Visit"] > 1)]
            record["TimeToConversion"] = (
                conversion_visits["MR Delay"].iloc[0] if len(conversion_visits) else None
            )
        else:
            record["TimeToConversion"] = None

        records.append(record)

    longitudinal = pd.DataFrame(records)
    return df.merge(longitudinal, on="ID", how="left")


def get_feature_columns(include_engineered: bool = True) -> list[str]:
    """Return default model feature column names."""
    base = ["Age", "Gender", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
    engineered = [
        "BrainAtrophyRatio",
        "CognitiveReserveIndex",
        "MMSE_norm",
        "SES_EDUC_interaction",
    ]
    return base + (engineered if include_engineered else [])
