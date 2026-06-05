"""Load, validate, and merge OASIS-1 and OASIS-2 datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import data_path


def load_oasis1(path: str | Path) -> pd.DataFrame:
    """Load and do initial validation of OASIS-1 (cross-sectional)."""
    df = pd.read_csv(path)
    df.rename(
        columns={
            "M/F": "Gender",
            "Educ": "EDUC",
            "Group": "Diagnosis",
        },
        inplace=True,
    )
    df["Source"] = "OASIS1"
    return df


def load_oasis2(path: str | Path) -> pd.DataFrame:
    """Load and do initial validation of OASIS-2 (longitudinal)."""
    df = pd.read_csv(path)
    df.rename(columns={"M/F": "Gender", "Educ": "EDUC"}, inplace=True)
    df["Source"] = "OASIS2"
    return df


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OASIS-1 and OASIS-2.

    Uses OASIS-2 baseline visit (Visit == 1) to avoid leakage from future visits.
    Deduplicates subjects appearing in both datasets by ID (keeps OASIS-2 entry).
    """
    df2_baseline = df2[df2["Visit"] == 1].copy()
    merged = pd.concat([df1, df2_baseline], ignore_index=True)
    merged.drop_duplicates(subset=["ID"], keep="last", inplace=True)
    return merged


def load_and_merge(
    oasis1_path: str | Path | None = None,
    oasis2_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load both datasets, merge them, and return (oasis1, oasis2, merged)."""
    oasis1_path = Path(oasis1_path or data_path("raw", "oasis_cross-sectional.csv"))
    oasis2_path = Path(oasis2_path or data_path("raw", "oasis_longitudinal.csv"))

    df1 = load_oasis1(oasis1_path)
    df2 = load_oasis2(oasis2_path)
    merged = merge_datasets(df1, df2)
    return df1, df2, merged
