# Biology Topics to Study Before Starting Data Science Work

## Alzheimer's Disease ML Classification Project

> Study these topics in order — each section builds on the previous.
> Goal: understand your data deeply enough to interpret model outputs clinically.

---

## 🧠 Section 1: Brain Anatomy Basics

*Why: The dataset features (nWBV, eTIV, ASF) are volumetric brain measurements. You need to know what they physically represent.*

- [] Major lobes of the brain (frontal, temporal, parietal, occipital) and their functions
- [ ] What the cerebral cortex is and why cortical thickness matters in aging
- [ ] What grey matter vs white matter is and how they differ functionally
- [ ] What the hippocampus is, where it sits, and why it's the first structure affected by Alzheimer's
- [ ] What the entorhinal cortex is and its role in memory
- [ ] What intracranial volume (ICV / eTIV) is — the total volume inside the skull
- [ ] What brain atrophy means: definition, causes, and how it's measured
- [ ] What cerebrospinal fluid (CSF) is and how it relates to brain volume measurements
- [ ] What the ventricles are and why ventricular enlargement is a sign of atrophy
- [ ] Difference between grey matter atrophy vs whole brain volume loss

---

## 🔬 Section 2: Neuroscience of Aging

*Why: Age is a top feature in your dataset. Understanding why aging affects the brain helps interpret model predictions.*

- [ ] Normal cognitive aging vs pathological cognitive decline
- [ ] How brain volume naturally decreases with age (rate, affected regions)
- [ ] What cognitive reserve is — the protective role of education and mental activity
- [ ] How synaptic density changes with age
- [ ] The role of neuroplasticity in aging
- [ ] What mild cognitive impairment (MCI) is and how it differs from normal aging and Alzheimer's
- [ ] Why women have higher Alzheimer's prevalence than men (hormonal and genetic factors)
- [ ] How lifestyle factors (exercise, diet, sleep, stress) affect brain aging

---

## 🔴 Section 3: Alzheimer's Disease (Core)

*Why: This is the disease your model is diagnosing. Deep understanding is essential for clinical interpretation of results.*

- [ ] What Alzheimer's Disease (AD) is — definition and classification as a neurodegenerative disease
- [ ] Difference between early-onset (<65 yrs) and late-onset (>65 yrs) Alzheimer's
- [ ] The two pathological hallmarks of AD:
  - [ ] Amyloid plaques (Aβ) — what they are, how they form, where they accumulate
  - [ ] Neurofibrillary tangles (tau proteins) — what tau is, hyperphosphorylation, and how tangles damage neurons
- [ ] The amyloid cascade hypothesis — the dominant (but debated) theory of AD causation
- [ ] How AD progresses through the brain: entorhinal cortex → hippocampus → neocortex (Braak stages)
- [ ] The 7 stages of Alzheimer's progression (GDS / Reisberg Scale)
- [ ] How neuronal death leads to cortical thinning and measurable brain volume loss
- [ ] The role of ApoE4 gene variant as the strongest genetic risk factor for late-onset AD
- [ ] Why early diagnosis is clinically critical (intervention window, caregiver planning)
- [ ] Current treatment landscape: no cure; only symptomatic treatments (cholinesterase inhibitors, memantine)

---

## 📏 Section 4: Clinical Assessment Tools in the Dataset

*Why: CDR and MMSE are your target variable and a key feature. You must understand exactly what they measure.*

### MMSE (Mini-Mental State Examination)

- [ ] What the MMSE tests: orientation, registration, attention, recall, language, visuospatial (30 questions)
- [ ] Score interpretation: 24–30 = Normal, 18–23 = Mild impairment, 0–17 = Severe impairment
- [ ] Limitations of MMSE: education bias, ceiling effect in early dementia, language/cultural sensitivity
- [ ] What it means clinically when MMSE declines over time
- [ ] MMSE vs MoCA (Montreal Cognitive Assessment) — why MMSE is older but still standard

### CDR (Clinical Dementia Rating)

- [ ] What the CDR measures: 6 domains — memory, orientation, judgment, community affairs, home/hobbies, personal care
- [ ] CDR scoring: 0 = No dementia, 0.5 = Very mild, 1 = Mild, 2 = Moderate, 3 = Severe
- [ ] Why CDR = 0.5 is classified as demented in this project (it indicates functional impairment)
- [ ] How CDR is administered (clinician interview + informant report)
- [ ] CDR Sum of Boxes (CDR-SB) as a continuous severity measure
- [ ] Why CDR is considered the gold standard for staging dementia severity

### SES (Socioeconomic Status)

- [ ] How SES is measured (income, occupation, education composite)
- [ ] Link between low SES and higher dementia risk (healthcare access, chronic stress, lower education)
- [ ] Why ~20% missing SES values is a concern and how to address it

---

## 🧪 Section 5: Neuroimaging Concepts (Tabular Proxies)

*Why: eTIV, nWBV, and ASF come from MRI measurements. Understanding them ensures correct feature engineering.*

- [ ] What structural MRI (sMRI / T1-weighted MRI) is and what it measures
- [ ] What eTIV (Estimated Total Intracranial Volume) represents — a proxy for head/brain size
- [ ] Why eTIV is used as a baseline correction (larger heads have more brain tissue by default)
- [ ] What nWBV (Normalized Whole Brain Volume) is: brain tissue volume as a proportion of eTIV
- [ ] Why nWBV decreases in Alzheimer's — direct consequence of neuronal death
- [ ] What ASF (Atlas Scaling Factor) is: a correction factor from FreeSurfer for head size normalization
- [ ] Why nWBV and eTIV are correlated and why you should normalize (use ratio, not raw values)
- [ ] What FreeSurfer is (the software pipeline that generates these measurements)
- [ ] Concept of brain parcellation — dividing the brain into regions for volumetric measurement

---

## 📊 Section 6: Epidemiology & Risk Factors

*Why: Required context for framing your paper's Introduction and interpreting fairness analysis results.*

- [ ] Global prevalence of Alzheimer's disease (current numbers, projected increases)
- [ ] Age as the single greatest risk factor — exponential increase after age 65
- [ ] Modifiable vs non-modifiable risk factors for AD:
  - Non-modifiable: age, genetics (APOE4), family history, Down syndrome
  - Modifiable: hypertension, diabetes, obesity, smoking, depression, physical inactivity, hearing loss
- [ ] The Lancet Commission's 12 modifiable risk factors for dementia prevention
- [ ] Why education and cognitive reserve are protective (cognitive reserve hypothesis)
- [ ] Sex/gender differences in AD prevalence and presentation
- [ ] Racial and socioeconomic disparities in AD diagnosis and access to care
- [ ] What the "dementia care gap" is in low/middle income countries

---

## ⚕️ Section 7: Clinical Context for Model Outputs

*Why: Medical AI papers must connect model results to clinical implications. This section prepares you to do that.*

- [ ] What sensitivity (recall) means clinically: ability to correctly identify sick patients (true positive rate)
- [ ] What specificity means clinically: ability to correctly rule out disease in healthy patients
- [ ] Why in disease screening, **high sensitivity is prioritized** over specificity
- [ ] What false negatives mean in Alzheimer's: missed diagnoses, delayed treatment
- [ ] What false positives mean: unnecessary anxiety, follow-up costs, stigma
- [ ] The concept of clinical decision support (CDS) — how ML assists (not replaces) physicians
- [ ] Why model explainability is ethically critical in medical AI (EU AI Act, FDA guidance)
- [ ] What algorithmic bias is in healthcare and why it causes real harm
- [ ] The difference between a screening tool, a diagnostic tool, and a prognostic tool
- [ ] TRIPOD guidelines: reporting standards for clinical prediction models (skim the paper)

---

## 📚 Recommended Resources

### Free Online Resources

- **Khan Academy** — Brain structure and function (free, beginner-friendly)
- **Coursera** — "Understanding the Brain" (University of Chicago, free to audit)
- **Alzheimer's Association** (alz.org) — Stages and symptoms, risk factors
- **OASIS Brain website** (oasis-brains.org) — Methods paper for the dataset
- **NIA (nia.nih.gov)** — Alzheimer's disease fact sheets

### Key Papers to Read

- Folstein et al. (1975) — Original MMSE paper
- Morris (1993) — Original CDR paper
- Buckner et al. (2004) — Aging, Alzheimer's, and cortical thinning
- Livingston et al. (2020) — Lancet Commission on dementia prevention (12 risk factors)
- Marcus et al. (2007) — OASIS-1 original dataset paper ← **Required reading**
- Marcus et al. (2010) — OASIS-2 longitudinal dataset paper ← **Required reading**

### YouTube Channels

- **Nucleus Medical Media** — Animated Alzheimer's pathology overview
- **3Blue1Brown** / neuroscience channels — Brain anatomy basics
- **BrainFacts.org** — Short neuroscience explainers

---

## ✅ Study Priority Order

| Priority | Topic | Time Estimate |
|---|---|---|
| 🔴 Must know before coding | CDR and MMSE scoring | 2–3 hrs |
| 🔴 Must know before coding | What nWBV and eTIV represent | 1–2 hrs |
| 🔴 Must know before coding | Alzheimer's hallmarks and progression | 3–4 hrs |
| 🟡 Know before modeling | Brain anatomy basics | 2–3 hrs |
| 🟡 Know before modeling | Cognitive reserve + education effect | 1–2 hrs |
| 🟡 Know before modeling | Risk factors and epidemiology | 2–3 hrs |
| 🟢 Know before writing | Sensitivity vs specificity clinical framing | 1 hr |
| 🟢 Know before writing | TRIPOD guidelines overview | 1–2 hrs |
| 🟢 Know before writing | Algorithmic bias in healthcare | 1–2 hrs |

**Total estimated study time: ~16–24 hours**
