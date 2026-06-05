# Interpretable Machine Learning for Alzheimer's Disease Classification with Fairness Analysis

## Overview

This repository contains the foundational planning and technical architecture documents for building an interpretable, fairness-aware machine learning pipeline designed to classify Alzheimer's Disease dementia status. The project leverages tabular clinical data from the OASIS-1 (cross-sectional) and OASIS-2 (longitudinal) datasets.

The primary goal is to build a binary classification model (demented vs. non-demented) that not only achieves high predictive performance but also provides clinically interpretable feature importance via SHAP analysis and evaluates model equity across demographic subgroups.

## Repository Contents

This repository currently holds the core documentation driving the project:

* **`Biology_Topics_Study_Guide.md`**: A structured study roadmap covering essential brain anatomy, the neuroscience of aging, Alzheimer's pathology, and clinical assessment tools like the MMSE and CDR. This ensures all feature engineering and model outputs are grounded in clinical reality.


* **`PRD_Alzheimers_Classification.md`**: The Product Requirements Document detailing the project's vision, problem statement, required datasets, feature engineering logic, and success metrics targeting peer-reviewed publication.


* **`TRD_Alzheimers_Classification.md`**: The Technical Requirements Document outlining the system architecture, required Python environment, data pipeline specifications, model evaluation standards, and fairness metric implementations.



## Technical Stack & Methodologies

As outlined in the TRD, the upcoming implementation will utilize:

* **Language:** Python >= 3.9


* **Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost


* **Imbalanced Learning:** SMOTE via `imbalanced-learn`

* **Explainability (XAI):** SHAP and LIME


* **Fairness Analysis:** Fairlearn and AIF360



## Getting Started

To view the project plans, simply open the respective `.md` files. If you are setting up the development environment, please refer to the environment and dependencies section within `TRD_Alzheimers_Classification.md`.
