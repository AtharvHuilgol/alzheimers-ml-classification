# Technical Requirements Document (TRD)
## Interpretable ML for Alzheimer's Disease Classification
**Version:** 1.0  
**Status:** Draft

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│  OASIS-1 (CSV)  +  OASIS-2 (CSV)  →  Merged Master Dataset │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  PREPROCESSING LAYER                        │
│  Cleaning → Imputation → Encoding → Feature Engineering     │
│  → Scaling → SMOTE → Train/Test Split                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   MODELING LAYER                            │
│  8+ Algorithms → Cross-Validation → Hyperparameter Tuning  │
│  → Best Model Selection                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
┌────────────────────┐         ┌───────────────────────┐
│ EXPLAINABILITY     │         │  FAIRNESS LAYER        │
│ SHAP Analysis      │         │  Subgroup Evaluation   │
│ Feature Importance │         │  Disparity Metrics     │
└────────────────────┘         └───────────────────────┘
         │                                 │
         └────────────────┬────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│  Results Tables + Figures + Statistical Tests + Paper       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Environment & Dependencies

### 2.1 Python Version
```
Python >= 3.9
```

### 2.2 Required Libraries

```txt
# requirements.txt

# Core
numpy==1.26.4
pandas==2.2.1
scipy==1.13.0

# Visualization
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.20.0

# Machine Learning
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
catboost==1.2.5

# Imbalanced Learning
imbalanced-learn==0.12.2

# Hyperparameter Tuning
optuna==3.6.1

# Explainability
shap==0.45.0
lime==0.2.0.1

# Fairness
fairlearn==0.10.0
aif360==0.6.1

# Statistics
pingouin==0.5.4
statsmodels==0.14.2

# Utilities
joblib==1.4.0
tqdm==4.66.2
jupyter==1.0.0
```

### 2.3 Environment Setup
```bash
# Create virtual environment
python -m venv alzheimers_env
source alzheimers_env/bin/activate  # Mac/Linux
alzheimers_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set global random seed in all notebooks
import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

---

## 3. Project Directory Structure

```
alzheimers-ml/
│
├── data/
│   ├── raw/
│   │   ├── oasis_cross-sectional.csv       # OASIS-1
│   │   └── oasis_longitudinal.csv          # OASIS-2
│   ├── processed/
│   │   ├── oasis1_cleaned.csv
│   │   ├── oasis2_cleaned.csv
│   │   └── oasis_merged_final.csv
│   └── data_dictionary.md                  # Document all variables
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Modeling.ipynb
│   ├── 05_SHAP_Explainability.ipynb
│   └── 06_Fairness_Analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      # Load & merge datasets
│   ├── preprocessing.py                    # Cleaning, imputation, encoding
│   ├── feature_engineering.py             # New feature creation
│   ├── modeling.py                         # Train/evaluate all models
│   ├── explainability.py                   # SHAP functions
│   ├── fairness.py                         # Subgroup evaluation
│   └── utils.py                            # Shared utilities
│
├── results/
│   ├── figures/                            # All saved plots
│   ├── tables/                             # CSV result tables
│   └── models/                             # Saved model objects (.pkl)
│
├── paper/
│   ├── draft.docx
│   └── references.bib
│
├── requirements.txt
└── README.md
```

---

## 4. Data Pipeline Specifications

### 4.1 Data Loading

```python
# src/data_loader.py

def load_oasis1(path: str) -> pd.DataFrame:
    """Load and do initial validation of OASIS-1."""
    df = pd.read_csv(path)
    # Rename columns to standardized names
    df.rename(columns={
        'M/F': 'Gender',
        'Educ': 'EDUC',
        'Group': 'Diagnosis'
    }, inplace=True)
    df['Source'] = 'OASIS1'
    return df

def load_oasis2(path: str) -> pd.DataFrame:
    """Load and do initial validation of OASIS-2."""
    df = pd.read_csv(path)
    df.rename(columns={'M/F': 'Gender', 'Educ': 'EDUC'}, inplace=True)
    df['Source'] = 'OASIS2'
    return df

def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OASIS-1 and OASIS-2.
    Strategy: Use OASIS-2 baseline visit for longitudinal subjects
    to avoid data leakage from future visits during classification.
    Deduplicate subjects appearing in both datasets by Subject ID.
    """
    # From OASIS-2, take only the first visit (Visit == 1)
    df2_baseline = df2[df2['Visit'] == 1].copy()
    merged = pd.concat([df1, df2_baseline], ignore_index=True)
    # Drop duplicate subjects (keep OASIS-2 entry)
    merged.drop_duplicates(subset=['ID'], keep='last', inplace=True)
    return merged
```

### 4.2 Data Cleaning Specifications

| Step | Rule | Implementation |
|---|---|---|
| CDR null removal | Drop rows where CDR is null | `df.dropna(subset=['CDR'])` |
| Converted label | Map CDR > 0 → 1; CDR = 0 → 0 | `df['target'] = (df['CDR'] > 0).astype(int)` |
| Gender encoding | M → 1, F → 0 | `df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})` |
| MMSE range check | Flag MMSE outside 0–30 | Assert + log warnings |
| SES imputation | Median per gender + age_group | `SimpleImputer` per subgroup |
| ASF / eTIV outliers | IQR-based capping (1.5×IQR) | Cap, do not remove |
| Diagnosis label drop | Remove "Converted" class from OASIS-2 | `df = df[df['Group'] != 'Converted']` |

> ⚠️ **Note on "Converted" label in OASIS-2:** Some subjects are labeled "Converted" (changed from non-demented to demented mid-study). These should be excluded from binary classification to avoid ambiguity, and discussed as a limitation.

### 4.3 Feature Engineering Specifications

```python
# src/feature_engineering.py

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Brain Atrophy Ratio
    df['BrainAtrophyRatio'] = df['eTIV'] / df['nWBV']

    # 2. Cognitive Reserve Index
    df['CognitiveReserveIndex'] = df['EDUC'] / (df['Age'] + 1e-5)

    # 3. Age Groups (clinically standard bins)
    df['AgeGroup'] = pd.cut(df['Age'],
                            bins=[0, 65, 75, 85, 120],
                            labels=['<65', '65-75', '75-85', '85+'])

    # 4. MMSE Severity Bins
    df['MMSESeverity'] = pd.cut(df['MMSE'],
                                bins=[-1, 17, 23, 30],
                                labels=['Moderate', 'Mild', 'Normal'])

    # 5. Normalized MMSE
    df['MMSE_norm'] = df['MMSE'] / 30.0

    # 6. SES-Education Interaction
    df['SES_EDUC_interaction'] = df['SES'] * df['EDUC']

    return df
```

### 4.4 Train/Test Split Strategy

```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# 80/20 stratified split (maintain class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5-fold stratified cross-validation for all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 4.5 SMOTE Specification

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE must be applied ONLY inside CV folds (not on full train set before CV)
# Use ImbPipeline to enforce this:

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', classifier)
])
```

---

## 5. Modeling Specifications

### 5.1 Required Algorithms

| # | Model | Library | Key Hyperparameters to Tune |
|---|---|---|---|
| 1 | Logistic Regression | sklearn | C, penalty, solver |
| 2 | Decision Tree | sklearn | max_depth, min_samples_split |
| 3 | Random Forest | sklearn | n_estimators, max_depth, min_samples_leaf |
| 4 | Gradient Boosting (sklearn) | sklearn | n_estimators, learning_rate, max_depth |
| 5 | XGBoost | xgboost | n_estimators, learning_rate, subsample, colsample_bytree |
| 6 | LightGBM | lightgbm | num_leaves, learning_rate, min_child_samples |
| 7 | SVM | sklearn | C, kernel, gamma |
| 8 | K-Nearest Neighbors | sklearn | n_neighbors, weights, metric |
| 9 | Naive Bayes | sklearn | var_smoothing |
| 10 | Neural Network (MLP) | sklearn | hidden_layer_sizes, activation, alpha |

### 5.2 Hyperparameter Tuning Specification

```python
import optuna

# Tune top 3 models by CV AUC-ROC using Optuna
# Example for XGBoost:

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'random_state': 42
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv,
                            scoring='roc_auc', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective_xgb, n_trials=100)
```

### 5.3 Evaluation Metrics Specification

All metrics computed per CV fold; reported as `mean ± std`:

| Metric | Function | Priority |
|---|---|---|
| AUC-ROC | `roc_auc_score` | **Primary** |
| Sensitivity (Recall) | `recall_score` | **Primary** (medical) |
| F1-Score (weighted) | `f1_score(average='weighted')` | **Primary** |
| Specificity | `TN / (TN + FP)` | Secondary |
| Precision | `precision_score` | Secondary |
| Accuracy | `accuracy_score` | Tertiary (misleading with imbalance) |
| MCC | `matthews_corrcoef` | Secondary (robust to imbalance) |
| Brier Score | `brier_score_loss` | Secondary (calibration) |

### 5.4 Statistical Significance Testing

```python
from scipy.stats import wilcoxon

# Compare top 2 models with Wilcoxon signed-rank test
# Input: per-fold AUC-ROC scores of two models (5 values each)
stat, p_value = wilcoxon(model_a_scores, model_b_scores)
# Report: W statistic + p-value in results table
# Significance threshold: p < 0.05
```

### 5.5 Results Table Format

Produce a table with the following columns for the paper:

| Model | AUC-ROC | Sensitivity | Specificity | F1 | MCC | Fit Time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | x.xx ± x.xx | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... | ... |
| ... | | | | | | |

---

## 6. Explainability (SHAP) Specifications

### 6.1 SHAP Explainer Selection

| Model Type | SHAP Explainer | Notes |
|---|---|---|
| Tree-based (RF, XGB, LGBM) | `shap.TreeExplainer` | Exact SHAP values, fast |
| Linear (LR) | `shap.LinearExplainer` | Exact for linear models |
| SVM, KNN, MLP | `shap.KernelExplainer` | Approximate, slower |

### 6.2 Required SHAP Outputs

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# 1. Global Feature Importance Bar Plot
shap.summary_plot(shap_values, X_test, plot_type='bar')

# 2. Beeswarm / Dot Plot (shows direction + magnitude)
shap.summary_plot(shap_values, X_test)

# 3. SHAP Dependence Plot for top 3 features
for feature in top_3_features:
    shap.dependence_plot(feature, shap_values, X_test)

# 4. Waterfall plot for one individual (paper Figure)
shap.waterfall_plot(explainer.expected_value,
                    shap_values[true_positive_idx],
                    X_test.iloc[true_positive_idx])
```

### 6.3 SHAP Clinical Interpretation Requirements

For each of the top 5 SHAP features, write a 2–3 sentence clinical interpretation:

- What does the feature represent biologically?
- In which direction does it influence dementia prediction?
- Does this align with published medical evidence?

---

## 7. Fairness Analysis Specifications

### 7.1 Subgroup Definitions

| Attribute | Groups |
|---|---|
| Gender | Male, Female |
| Age Group | <65, 65–75, 75–85, 85+ |
| SES | Low (1–2), Medium (3), High (4–5) |
| Education | Low EDUC (<12 yrs), High EDUC (≥12 yrs) |

### 7.2 Per-Subgroup Metrics

For each subgroup, report:
- Sample size and class distribution
- AUC-ROC
- Sensitivity
- Specificity
- F1-Score

### 7.3 Fairness Metrics (Formal)

```python
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)

# Demographic Parity Difference
# Measures: |P(ŷ=1 | A=0) - P(ŷ=1 | A=1)|
# Target: < 0.1 for acceptable fairness

# Equalized Odds Difference
# Measures: max difference in TPR and FPR across groups
# Target: < 0.1

metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'sensitivity': recall_score,
        'f1': f1_score
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test['Gender']
)

metric_frame.by_group  # Shows metrics per subgroup
```

### 7.4 Fairness Visualization Requirements

- Grouped bar chart: F1 and Sensitivity by Gender and Age Group
- Heatmap: AUC-ROC across Gender × Age Group grid
- Discussion: If disparities found, propose mitigation strategies (reweighting, threshold adjustment)

---

## 8. Statistical Reporting Standards

All reported numbers must follow these standards:

| Metric | Format | Example |
|---|---|---|
| Mean ± Std | 4 decimal places | 0.9123 ± 0.0214 |
| p-values | 3 decimal places | p = 0.043 |
| Confidence intervals | 95% CI | [0.89, 0.94] |
| Sample sizes | Integer | n = 416 |
| Percentages | 1 decimal | 62.3% |

> Use **Wilcoxon signed-rank test** for paired model comparison (non-parametric, appropriate for small CV fold counts).

---

## 9. Figures Required for Paper

| Figure # | Content | Plot Type |
|---|---|---|
| Fig 1 | Dataset overview + class distribution | Bar chart |
| Fig 2 | Correlation heatmap (all features) | Heatmap |
| Fig 3 | MMSE vs nWBV by dementia status | Scatter plot |
| Fig 4 | Model comparison (AUC-ROC ± Std) | Bar chart with error bars |
| Fig 5 | ROC curves for top 3 models | Line plot |
| Fig 6 | SHAP global importance (beeswarm) | SHAP plot |
| Fig 7 | SHAP dependence plot (top feature) | SHAP plot |
| Fig 8 | SHAP waterfall for one patient | SHAP plot |
| Fig 9 | Fairness: F1 by subgroup | Grouped bar chart |
| Fig 10 | Confusion matrix (best model) | Heatmap |

**Figure requirements for publication:**
- Resolution: ≥ 300 DPI
- Format: `.tiff` or `.pdf` (IEEE) / `.png` (Diagnostics)
- Font size: ≥ 10pt for all labels
- Color scheme: colorblind-friendly (use `seaborn` colorblind palette)

---

## 10. Reproducibility Requirements

```python
# At the top of EVERY notebook:
import numpy as np
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# For XGBoost:
# Pass seed parameter directly: XGBClassifier(random_state=SEED)

# For CatBoost:
# CatBoostClassifier(random_seed=SEED)
```

**GitHub Repository Requirements:**
- `README.md` with setup and run instructions
- `requirements.txt` with pinned versions
- All notebooks runnable top-to-bottom without error
- Data download instructions (since OASIS requires registration)
- `results/` folder with all pre-generated outputs committed

---

## 11. Paper Structure (Technical Mapping)

| Paper Section | Source Notebook | Key Content |
|---|---|---|
| Abstract | All | 250 words max: problem, method, results, conclusion |
| Introduction | Literature | Gap analysis, research questions, contributions |
| Related Work | Literature | ≥ 15 prior OASIS ML papers + fairness/XAI papers |
| Materials & Methods | 01, 02, 03 | Dataset, preprocessing, feature engineering |
| Experimental Setup | 04 | Models, CV, tuning, evaluation metrics |
| Results | 04, 05, 06 | Tables + figures |
| Discussion | All | Clinical interpretation, limitations, future work |
| Conclusion | All | Summary of contributions |
| References | Zotero | APA or IEEE format depending on venue |

---

## 12. Compute Requirements

| Task | Estimated Time | RAM Needed |
|---|---|---|
| EDA | < 5 min | 2 GB |
| Preprocessing | < 2 min | 2 GB |
| Training all 10 models (CV) | 5–20 min | 4 GB |
| Optuna tuning (100 trials) | 30–90 min | 4 GB |
| SHAP (TreeExplainer) | 2–10 min | 4 GB |
| SHAP (KernelExplainer - SVM) | 30–60 min | 8 GB |

> 💡 A standard laptop (8GB RAM, no GPU) is sufficient for this project. Google Colab (free tier) works fine if local resources are limited.
