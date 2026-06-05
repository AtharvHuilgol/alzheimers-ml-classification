# Interpretable Machine Learning for Alzheimer's Disease Classification with Fairness Analysis

Binary classification pipeline for dementia status (demented vs. non-demented) using OASIS-1 and OASIS-2 clinical data, with SHAP explainability and fairness analysis across demographic subgroups.

## Project Structure

```
├── data/
│   ├── raw/                          # Place raw OASIS CSVs here
│   ├── processed/                    # Cleaned and merged outputs
│   └── data_dictionary.md            # Variable documentation
├── docs/                             # Planning documents (PRD, TRD, study guide)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Modeling.ipynb
│   ├── 05_SHAP_Explainability.ipynb
│   └── 06_Fairness_Analysis.ipynb
├── src/
│   ├── data_loader.py                # Load & merge datasets
│   ├── preprocessing.py              # Cleaning, imputation, encoding
│   ├── feature_engineering.py        # New feature creation
│   ├── modeling.py                   # Train/evaluate all models
│   ├── explainability.py             # SHAP functions
│   ├── fairness.py                   # Subgroup evaluation
│   └── utils.py                      # Shared utilities
├── results/
│   ├── figures/                      # Saved plots
│   ├── tables/                       # CSV result tables
│   └── models/                       # Saved model objects (.pkl)
├── paper/
│   ├── draft.docx
│   └── references.bib
├── requirements.txt
└── README.md
```

## Documentation

Planning and domain context live in `docs/`:

- `PRD_Alzheimers_Classification.md` — product requirements, features, success metrics
- `TRD_Alzheimers_Classification.md` — technical architecture, pipeline specs
- `Biology_Topics_Study_Guide.md` — clinical background for interpreting results

## Setup

```bash
# Create virtual environment
python -m venv alzheimers_env
alzheimers_env\Scripts\activate     # Windows
# source alzheimers_env/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Data

Download OASIS datasets and place them in `data/raw/`:

| File | Dataset |
|---|---|
| `oasis_cross-sectional.csv` | OASIS-1 |
| `oasis_longitudinal.csv` | OASIS-2 |

Sources: [oasis-brains.org](https://www.oasis-brains.org/) or [Kaggle](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)

## Workflow

Run notebooks in order from the `notebooks/` directory:

1. **01_EDA** — exploratory analysis and visualizations
2. **02_Preprocessing** — cleaning, imputation, encoding
3. **03_Feature_Engineering** — clinically motivated features
4. **04_Modeling** — train 8+ models with 5-fold CV
5. **05_SHAP_Explainability** — global and local SHAP plots
6. **06_Fairness_Analysis** — subgroup performance and disparity metrics

Or use the `src/` modules directly:

```python
from src.data_loader import load_and_merge
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features
from src.modeling import compare_all_models

df1, df2, merged = load_and_merge()
df = engineer_features(preprocess_pipeline(merged))
results = compare_all_models(df[feature_cols], df['target'])
```

## Success Metrics

| Metric | Target |
|---|---|
| AUC-ROC | ≥ 0.88 |
| F1-Score (weighted) | ≥ 0.82 |
| Sensitivity | ≥ 0.80 |
| Specificity | ≥ 0.80 |

## License

See repository license. OASIS data usage is subject to the OASIS data use agreement.
