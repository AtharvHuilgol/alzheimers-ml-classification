"""Cleaning, imputation, encoding, and scaling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import add_age_group

logger = logging.getLogger(__name__)

CONTINUOUS_FEATURES = ["Age", "EDUC", "MMSE", "eTIV", "nWBV", "ASF"]
MMSE_MIN, MMSE_MAX = 0, 30


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map CDR to binary target: 0 = non-demented, 1 = demented."""
    df = df.copy()
    df["target"] = (df["CDR"] > 0).astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply core cleaning rules from the TRD."""
    df = df.copy()

    # Drop rows with missing CDR
    df = df.dropna(subset=["CDR"])

    # Exclude ambiguous "Converted" label from OASIS-2
    if "Diagnosis" in df.columns:
        df = df[df["Diagnosis"] != "Converted"]
    elif "Group" in df.columns:
        df = df[df["Group"] != "Converted"]

    df = create_target(df)
    return df


def validate_mmse(df: pd.DataFrame) -> pd.DataFrame:
    """Flag MMSE values outside the valid clinical range [0, 30]."""
    if "MMSE" not in df.columns:
        return df

    invalid = (df["MMSE"] < MMSE_MIN) | (df["MMSE"] > MMSE_MAX)
    if invalid.any():
        logger.warning(
            "Found %d rows with MMSE outside [%d, %d].",
            invalid.sum(),
            MMSE_MIN,
            MMSE_MAX,
        )
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender as binary: M -> 1, F -> 0."""
    df = df.copy()
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
    return df


def impute_ses(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing SES using median per gender and age group."""
    df = df.copy()
    if "SES" not in df.columns or df["SES"].notna().all():
        return df

    df = add_age_group(df)
    for (gender, age_group), group_idx in df.groupby(["Gender", "AgeGroup"]).groups.items():
        mask = df.index.isin(group_idx) & df["SES"].isna()
        if not mask.any():
            continue
        median_ses = df.loc[group_idx, "SES"].median()
        df.loc[mask, "SES"] = median_ses

    # Fallback: global median for any remaining missing values
    if df["SES"].isna().any():
        imputer = SimpleImputer(strategy="median")
        df["SES"] = imputer.fit_transform(df[["SES"]]).ravel()

    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Cap outliers at 1.5×IQR for specified continuous columns."""
    df = df.copy()
    columns = columns or ["ASF", "eTIV"]

    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a dataframe."""
    df = clean_data(df)
    df = validate_mmse(df)
    df = encode_gender(df)
    df = impute_ses(df)
    df = cap_outliers_iqr(df)
    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standard-scale continuous features for linear models."""
    columns = columns or [c for c in CONTINUOUS_FEATURES if c in X_train.columns]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if columns:
        X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
        X_test_scaled[columns] = scaler.transform(X_test[columns])

    return X_train_scaled, X_test_scaled, scaler
