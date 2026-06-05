# Product Requirements Document (PRD)
## Interpretable Machine Learning for Alzheimer's Disease Classification with Fairness Analysis
**Version:** 1.0  
**Status:** Draft  
**Target Venue:** IEEE Access / Diagnostics (MDPI)

---

## 1. Project Overview

### 1.1 Vision Statement
Develop a rigorous, interpretable, and fairness-aware machine learning pipeline that classifies Alzheimer's Disease (AD) dementia status from tabular clinical data, combining OASIS-1 and OASIS-2 datasets, and produces insights suitable for peer-reviewed publication.

### 1.2 Problem Statement
Alzheimer's Disease affects over 55 million people globally. Early and accurate diagnosis from accessible clinical measurements (cognitive scores, brain volume proxies, demographics) is critical for timely intervention. Existing ML studies on OASIS data suffer from three gaps:

- They rarely combine OASIS-1 (cross-sectional) and OASIS-2 (longitudinal) cohorts
- Most lack clinical explainability beyond raw accuracy metrics
- Almost none analyze whether model performance is equitable across demographic subgroups

This project addresses all three gaps.

### 1.3 Research Questions
1. Can a machine learning model accurately classify dementia status (demented vs. non-demented) from OASIS clinical features?
2. Which clinical features are the most influential predictors, and do they align with established medical literature?
3. Does model performance differ significantly across demographic subgroups (age group, gender, SES)?
4. Which ML algorithm achieves the best trade-off between performance, interpretability, and fairness?

---

## 2. Objectives

### 2.1 Primary Objectives
- Build a binary classification model (demented vs. non-demented) achieving AUC-ROC ≥ 0.90
- Apply SHAP analysis to generate clinically interpretable feature importance explanations
- Conduct fairness analysis across gender, age group, and SES subgroups
- Compare ≥ 8 ML algorithms with statistical significance testing

### 2.2 Secondary Objectives
- Engineer novel clinically meaningful features beyond raw dataset columns
- Handle class imbalance rigorously (SMOTE + class weighting comparison)
- Produce reproducible code and results for open-source release
- Write and submit a peer-reviewed journal paper

---

## 3. Stakeholders

| Stakeholder | Role | Interest |
|---|---|---|
| Researcher (you) | Lead author | Build, analyze, and publish |
| Co-author / Supervisor | Optional | Guidance, review |
| Medical community | End user of findings | Clinical decision support insights |
| IEEE Access / Diagnostics | Publisher | Novelty, rigor, reproducibility |
| Future researchers | Readers | Reproducible baseline |

---

## 4. Dataset Requirements

### 4.1 Data Sources

| Dataset | Source | URL | Access |
|---|---|---|---|
| OASIS-1 (Cross-sectional) | oasis-brains.org / Kaggle | kaggle.com/datasets/jboysen/mri-and-alzheimers | Free |
| OASIS-2 (Longitudinal) | oasis-brains.org / Kaggle | kaggle.com/datasets/jboysen/mri-and-alzheimers | Free |

### 4.2 Dataset Characteristics

**OASIS-1:**
- 416 subjects, single visit each
- Age range: 18–96
- Features: Age, Gender, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF
- ~20% missing SES values

**OASIS-2:**
- 150 subjects, 3–4 visits over time (373 total sessions)
- Longitudinal tracking of CDR changes
- Same core features as OASIS-1
- Includes "Visit" and "MR Delay" columns

### 4.3 Target Variable Definition

| CDR Value | Clinical Meaning | Binary Label |
|---|---|---|
| 0 | No dementia | 0 (Non-demented) |
| 0.5 | Very mild dementia | 1 (Demented) |
| 1 | Mild dementia | 1 (Demented) |
| 2 | Moderate dementia | 1 (Demented) |

### 4.4 Data Quality Requirements
- Document all missing values and their handling strategy
- Remove or flag duplicate subjects appearing in both OASIS-1 and OASIS-2
- Validate value ranges for all features (e.g., MMSE must be 0–30)
- Record all preprocessing decisions in a data dictionary

---

## 5. Feature Requirements

### 5.1 Raw Features (Input)

| Feature | Type | Description |
|---|---|---|
| Age | Continuous | Subject age in years |
| Gender | Categorical | M / F |
| EDUC | Ordinal | Years of formal education |
| SES | Ordinal | Socioeconomic status (1–5) |
| MMSE | Continuous | Mini-Mental State Exam score (0–30) |
| eTIV | Continuous | Estimated Total Intracranial Volume |
| nWBV | Continuous | Normalized Whole Brain Volume |
| ASF | Continuous | Atlas Scaling Factor |

### 5.2 Engineered Features (Required)

| Feature Name | Formula / Logic | Clinical Rationale |
|---|---|---|
| Brain Atrophy Ratio | `eTIV / nWBV` | Captures relative brain shrinkage |
| Cognitive Reserve Index | `EDUC × (1/Age)` | Proxies education-adjusted brain resilience |
| Age Group | Bins: <65, 65–75, 75–85, 85+ | Captures non-linear age effects |
| MMSE Severity | Bins: Normal(24–30), Mild(18–23), Moderate(<18) | Clinically standard cutoffs |
| Normalized MMSE | `MMSE / 30` | 0–1 scale for model stability |
| SES-EDUC Interaction | `SES × EDUC` | Socioeconomic + educational compounding |

### 5.3 OASIS-2 Specific Features (Longitudinal)

| Feature | Logic |
|---|---|
| CDR Change | CDR at visit N minus CDR at baseline |
| MMSE Slope | Linear slope of MMSE over all visits per subject |
| Conversion Flag | Binary: did patient convert from CDR=0 to CDR>0? |
| Time to Conversion | Months from baseline to first CDR>0 visit |

---

## 6. Functional Requirements

### 6.1 Exploratory Data Analysis (EDA)
- FR-01: Visualize distribution of all features stratified by CDR/dementia status
- FR-02: Generate correlation heatmap for numerical features
- FR-03: Plot MMSE vs nWBV scatter colored by dementia status
- FR-04: Analyze and visualize missing data patterns
- FR-05: Show class imbalance distribution

### 6.2 Preprocessing
- FR-06: Impute missing SES values using median imputation per gender/age group
- FR-07: Encode Gender as binary (0/1)
- FR-08: Apply StandardScaler to continuous features for linear models
- FR-09: Document and justify all preprocessing steps

### 6.3 Modeling
- FR-10: Train and evaluate ≥ 8 classification algorithms
- FR-11: Use stratified 5-fold cross-validation for all models
- FR-12: Apply SMOTE on training folds only (never on test data)
- FR-13: Tune hyperparameters using GridSearchCV or Optuna for top 3 models
- FR-14: Report mean ± std for all metrics across folds

### 6.4 Explainability
- FR-15: Generate global SHAP feature importance (bar plot + beeswarm)
- FR-16: Generate SHAP summary for top 3 features
- FR-17: Generate individual SHAP waterfall plots for: one true positive, one false negative
- FR-18: Interpret top 5 SHAP features against published medical literature

### 6.5 Fairness Analysis
- FR-19: Evaluate Accuracy, F1, AUC-ROC per subgroup: Gender (M/F), Age Group, SES quintile
- FR-20: Calculate Equalized Odds and Demographic Parity metrics
- FR-21: Visualize performance disparities across subgroups
- FR-22: Discuss clinical implications of identified disparities

---

## 7. Non-Functional Requirements

| Requirement | Specification |
|---|---|
| Reproducibility | Set random seed = 42 globally; provide requirements.txt |
| Code Quality | Modular functions, docstrings, PEP8 compliant |
| Documentation | Inline comments + README explaining how to run |
| Transparency | All preprocessing, splits, and decisions documented |
| Open Source | Code released on GitHub; dataset links provided |

---

## 8. Success Metrics

### 8.1 Model Performance Targets

| Metric | Minimum Target | Stretch Target |
|---|---|---|
| AUC-ROC | ≥ 0.88 | ≥ 0.93 |
| F1-Score (weighted) | ≥ 0.82 | ≥ 0.88 |
| Sensitivity (Recall) | ≥ 0.80 | ≥ 0.88 |
| Specificity | ≥ 0.80 | ≥ 0.88 |

> Note: In medical classification, **Sensitivity (detecting true positives / sick patients) is prioritized** over Specificity. Missing a dementia diagnosis (false negative) is clinically costlier than a false alarm.

### 8.2 Publication Targets
- Paper submitted to IEEE Access or Diagnostics (MDPI)
- Code published on GitHub with DOI via Zenodo
- Preprint uploaded to arXiv (cs.LG or q-bio.NC)

---

## 9. Deliverables

| # | Deliverable | Format |
|---|---|---|
| 1 | Cleaned, merged OASIS-1 + OASIS-2 dataset | .csv |
| 2 | EDA notebook | Jupyter .ipynb |
| 3 | Preprocessing + feature engineering pipeline | Python module |
| 4 | Model training + evaluation notebook | Jupyter .ipynb |
| 5 | SHAP explainability notebook | Jupyter .ipynb |
| 6 | Fairness analysis notebook | Jupyter .ipynb |
| 7 | Final results summary | .csv / .xlsx |
| 8 | Research paper (draft) | .docx / LaTeX |
| 9 | GitHub repository | Public repo |

---

## 10. Project Timeline

| Phase | Tasks | Duration |
|---|---|---|
| **Phase 1: Setup** | Download data, environment setup, literature review | Week 1 |
| **Phase 2: EDA** | Exploratory analysis, visualizations, data report | Week 2 |
| **Phase 3: Preprocessing** | Cleaning, imputation, feature engineering, merging | Week 3 |
| **Phase 4: Modeling** | Train 8+ models, CV, hyperparameter tuning | Weeks 4–5 |
| **Phase 5: Explainability** | SHAP analysis, clinical interpretation | Week 6 |
| **Phase 6: Fairness** | Subgroup evaluation, disparity analysis | Week 7 |
| **Phase 7: Writing** | Full paper draft | Weeks 8–10 |
| **Phase 8: Review** | Revision, proofreading, formatting | Week 11 |
| **Phase 9: Submission** | Submit to journal + arXiv + GitHub | Week 12 |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Small dataset size limits model generalization | High | High | Use cross-validation; acknowledge limitation in paper |
| Class imbalance skews results | High | High | SMOTE + class weighting; report both accuracy and F1 |
| Duplicate subjects in OASIS-1 and OASIS-2 | Medium | High | Cross-reference subject IDs before merging |
| Reviewer rejects due to "not novel enough" | Medium | High | Ensure explainability + fairness sections are thorough |
| Missing SES data biases fairness analysis | Medium | Medium | Document imputation method; discuss as limitation |
| Overfitting due to small N | Medium | High | Strict train/test split; use regularization |

---

## 12. Literature Review Requirements

At minimum, review and cite papers covering:
- Prior ML studies on OASIS dataset (search Google Scholar: "OASIS Alzheimer machine learning")
- SHAP in medical AI (Lundberg & Lee 2017; Lundberg et al. 2020)
- Fairness in medical AI (Obermeyer et al. 2019)
- MMSE and CDR clinical validity (Folstein et al. 1975; Morris 1993)
- Alzheimer's neurobiological markers (nWBV, brain atrophy)

Aim for **40–60 citations** in the final paper.
