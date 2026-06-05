# OASIS Data Dictionary

Variable documentation for OASIS-1 (cross-sectional) and OASIS-2 (longitudinal) datasets used in this project.

## Raw Features

| Variable | Type | Range / Values | Description | Source |
|---|---|---|---|---|
| ID | String | — | Subject identifier | Both |
| Age | Continuous | 18–96 | Subject age in years | Both |
| Gender | Categorical | M / F (encoded 1/0) | Biological sex | Both |
| EDUC | Ordinal | 0–23+ | Years of formal education | Both |
| SES | Ordinal | 1–5 | Socioeconomic status (Hollingshead scale) | Both |
| MMSE | Continuous | 0–30 | Mini-Mental State Examination score | Both |
| CDR | Ordinal | 0, 0.5, 1, 2, 3 | Clinical Dementia Rating | Both |
| eTIV | Continuous | > 0 | Estimated Total Intracranial Volume (mm³) | Both |
| nWBV | Continuous | 0–1 | Normalized Whole Brain Volume | Both |
| ASF | Continuous | > 0 | Atlas Scaling Factor (FreeSurfer) | Both |
| Visit | Integer | 1–4 | Visit number (OASIS-2 only) | OASIS-2 |
| MR Delay | Integer | ≥ 0 | Days since baseline MRI (OASIS-2 only) | OASIS-2 |
| Diagnosis | Categorical | Nondemented, Demented, Converted | Clinical group label | OASIS-1/2 |

## Target Variable

| Variable | Type | Definition |
|---|---|---|
| target | Binary | 0 = Non-demented (CDR = 0); 1 = Demented (CDR > 0) |

CDR clinical meaning:
- 0: No dementia
- 0.5: Very mild dementia
- 1: Mild dementia
- 2: Moderate dementia

## Engineered Features

| Variable | Type | Formula / Logic | Clinical Rationale |
|---|---|---|---|
| BrainAtrophyRatio | Continuous | eTIV / nWBV | Relative brain shrinkage proxy |
| CognitiveReserveIndex | Continuous | EDUC / (Age + ε) | Education-adjusted cognitive reserve |
| AgeGroup | Categorical | <65, 65–75, 75–85, 85+ | Non-linear age effects |
| MMSESeverity | Categorical | Normal (24–30), Mild (18–23), Moderate (<18) | Clinical MMSE cutoffs |
| MMSE_norm | Continuous | MMSE / 30 | Normalized cognitive score |
| SES_EDUC_interaction | Continuous | SES × EDUC | Socioeconomic + education interaction |

## Longitudinal Features (OASIS-2)

| Variable | Type | Logic |
|---|---|---|
| CDR_change | Continuous | CDR at last visit − CDR at baseline |
| MMSE_slope | Continuous | Linear slope of MMSE over visits |
| ConversionFlag | Binary | 1 if subject converted from CDR=0 to CDR>0 |
| TimeToConversion | Continuous | Months from baseline to first CDR>0 visit |

## Preprocessing Decisions

| Step | Rule | Rationale |
|---|---|---|
| CDR null removal | Drop rows with missing CDR | Cannot assign target label |
| Converted exclusion | Remove "Converted" subjects | Ambiguous binary label |
| Gender encoding | M→1, F→0 | Model-ready binary feature |
| SES imputation | Median per gender + age group | ~20% missing in OASIS-1 |
| Outlier capping | 1.5×IQR on ASF, eTIV | Preserve sample size |
| Merge strategy | OASIS-2 baseline visit only | Avoid longitudinal leakage |
| Deduplication | Keep OASIS-2 entry on ID conflict | Longitudinal cohort priority |

## Missing Data

| Variable | Approx. Missing % | Handling |
|---|---|---|
| SES | ~20% (OASIS-1) | Median imputation by gender + age group |
| MMSE | < 5% | Median imputation if needed |
| Other features | < 2% | Document and impute as encountered |
