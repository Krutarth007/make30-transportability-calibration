# Dynamic MAKE30 Prediction Across U.S. and European ICUs

**Cross-continental transportability and calibration drift of a landmark machine-learning model for Major Adverse Kidney Events at 30 days (MAKE30).**

This repository contains the analysis code and **aggregate** result artifacts for a dynamic prediction study of MAKE30 in critically ill adults, developed on two U.S. ICU databases and externally validated on a European one. The work is reported per TRIPOD+AI and is associated with a manuscript under review.

> ⚠️ **No patient-level data is included in this repository, and none ever can be.** The underlying databases are restricted-access and require individual PhysioNet credentialing. See [Data availability & PhysioNet compliance](#data-availability--physionet-compliance) before cloning, forking, or contributing.

---

## Study at a glance

- **Outcome:** MAKE30 — composite of all-cause death, new renal replacement therapy (RRT) not present at baseline, or persistent renal dysfunction (final creatinine ≥ 2× baseline) within 30 days.
- **Design:** Dynamic landmarking at L = 6, 12, 24, 48 h; a single LightGBM estimator with landmark time as a covariate (van Houwelingen–style dynamic prediction).
- **Development data:** MIMIC-IV (n = 48,207 stays) and eICU-CRD (n = 131,061 stays), both U.S.
- **External validation:** SICdb (Salzburg, Austria; n = 21,091 stays) — a zero-shot cross-continental transportability test.
- **Central finding:** Strong discrimination transports across continents, but **calibration drifts** under external deployment; a two-parameter (intercept + slope) recalibration estimated out-of-sample at the new site largely corrects it.
- **Interpretability:** SHAP (tree explainer) with gain-importance fallback; a prespecified nephrotoxin-block ablation via a harmonized, time-varying Nephrotoxin Exposure Index (NTEI).

Headline performance (common-support, transportable feature set):

| Setting | AUROC (95% CI) | AUPRC | Brier | Calib. slope | ICI |
|---|---|---|---|---|---|
| Internal test | 0.852 (0.847–0.857) | 0.485 | 0.076 | 1.008 | 0.006 |
| External (SICdb), as-is | 0.786 (0.776–0.795) | 0.390 | 0.118 | 1.038 | 0.156 |
| External (SICdb), recalibrated | — | — | — | — | 0.004 |

External recalibration: intercept = −1.174, slope = 1.054. Full numbers in [`results_bundle.json`](results_bundle.json).

---

## Data availability & PhysioNet compliance

This study uses three **restricted-access** clinical databases. **None of them — nor any row-level data, cache, or trained model derived from them — is or may be redistributed here.**

| Database | Access license | How to obtain |
|---|---|---|
| MIMIC-IV v2.2 | PhysioNet Credentialed Health Data License 1.5.0 | Credentialing + CITI training + signed DUA |
| eICU-CRD v2.0 | PhysioNet Credentialed Health Data License 1.5.0 | Credentialing + CITI training + signed DUA |
| SICdb v1.0.8 | PhysioNet **Contributor Review** Health Data License 1.5.0 | Credentialing + signed DUA **+ per-study approval by the SICdb contributors** |

**What this repository deliberately excludes (and why):**

- **No raw or intermediate patient data.** All `*.parquet` cache files produced by the notebook are local-only and `.gitignore`d. Distributing them would redistribute restricted data.
- **No trained model weights.** `make30_dynamic_model.joblib` is **not** included. Per PhysioNet's guidance, *models derived from MIMIC are treated as containing sensitive information and should be shared on PhysioNet under the same agreement as the source data* — not on open repositories. (The SICdb contributor-review terms reinforce this.) Credentialed researchers can regenerate the model bit-for-bit from the notebook; see [Reproducing the study](#reproducing-the-study).
- **Notebook outputs are stripped.** Cell outputs are cleared before commit (see `nbstripout` below) so that no identifiers or row-level previews are ever embedded in version control.

**What *is* included is aggregate-only:** summary tables (counts, percentages, medians/IQRs, performance metrics), publication figures (group-level curves and de-identified SHAP summaries), and JSON metric/schema bundles. These are the results and the code that produced them — exactly what the DUA encourages contributors to disseminate openly.

If you intend to run this code, you must independently obtain credentialed access to all three databases through [PhysioNet](https://physionet.org/) and comply with each governing DUA. **Do not** commit any data, cache, or model artifact to this or any public repository.

---

## Repository structure

```
make30-transportability-calibration/
├── README.md
├── LICENSE                              # code license (see License section)
├── .gitignore                          # excludes data, cache, *.joblib, *.parquet
├── requirements.txt
├── notebooks/
│   └── MAKE30_pipeline.ipynb           # full 3-part pipeline (cohort → features → modeling)
├── figures/                            # aggregate, publication-ready PNGs
│   ├── fig_qc_make30_components.png
│   ├── fig_cif_death.png
│   ├── fig_creatinine_trajectory.png   # median creatinine by outcome group
│   ├── fig_ntei_by_outcome.png
│   ├── fig_per_landmark_auroc.png
│   ├── fig_calibration.png
│   ├── fig_decision_curve.png
│   ├── fig_lodo_transportability.png
│   ├── fig_shap_beeswarm.png           # de-identified SHAP summary
│   └── fig_shap_ntei_dependence.png
├── tables/                             # aggregate CSVs (summary statistics only)
│   ├── table1_baseline.csv
│   ├── consort_flow.csv
│   ├── table_make30_incidence.csv
│   ├── table2_performance.csv
│   ├── table3_per_landmark_auroc.csv
│   ├── table_calibration_drift.csv
│   ├── table4_ablations.csv
│   ├── table5_subgroups_external.csv
│   ├── table_component_resolved.csv
│   ├── table_lodo_transportability.csv
│   ├── table_temporal_validation.csv
│   ├── table_feature_importance.csv
│   └── ... (remaining sensitivity / robustness tables)
├── results_bundle.json                 # aggregate headline metrics + counts
└── feature_manifest.json               # feature/channel schema + NTEI definition (no data)
```

> Note: `make30_dynamic_model.joblib` and all `*.parquet` cache files are intentionally **absent** and are listed in `.gitignore`.

---

## Reproducing the study

> Reproduction requires credentialed PhysioNet access to MIMIC-IV, eICU-CRD, and SICdb. Without it, the cohort-building cells cannot run, but the methods, configuration, and aggregate outputs remain fully inspectable.

1. **Obtain access** to all three databases via PhysioNet and download them locally.
2. **Set data paths.** Point the configuration cell (Part 1, Section 1) to your local database roots. Do not place data inside the repository tree.
3. **Create the environment:**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Run the notebook end to end.** It is organized as three parts in one file:
   - **Part 1 — Cohort & outcome:** multi-database I/O, baseline creatinine hierarchy, RRT ascertainment, MAKE30 labeling, Table 1 / CONSORT / QC.
   - **Part 2 — Feature engineering & NTEI:** per-channel temporal summaries across 20 vital/lab channels, the harmonized Nephrotoxin Exposure Index, and the landmark feature matrix.
   - **Part 3 — Modeling & validation:** grouped dev/external split, LightGBM training, calibration analysis and SICdb recalibration, decision-curve analysis, NTEI ablation, subgroup transportability, component-resolved analysis, and SHAP.
5. **Determinism:** a fixed seed governs all stochastic steps; with identical database versions the model and metrics regenerate exactly. Cached intermediates speed re-runs and are written outside version control.

### Keeping the repo clean (required)

Before any commit, strip notebook outputs so identifiers and previews are never embedded:

```bash
pip install nbstripout
nbstripout --install        # installs a git filter for this repo
nbstripout notebooks/MAKE30_pipeline.ipynb   # one-time clean of the existing file
```

---

## Methods summary

- **Cohort:** first ICU stay per adult (age ≥ 18) with LOS ≥ 6 h; prevalent kidney failure excluded (baseline eGFR < 15, chronic dialysis, or ESRD).
- **Baseline creatinine hierarchy:** most recent stable pre-admission value → earliest plausible in-stay value → CKD-EPI 2021 race-free back-estimation at eGFR 75.
- **Model:** LightGBM, hyperparameters fixed a priori (no site-specific tuning) to avoid optimistic bias; references include a regularized logistic baseline on the common-support set and a KDIGO-based clinical reference.
- **Validation:** grouped internal split; zero-shot external validation on SICdb; leave-one-database-out internal–external cross-validation; temporal validation in MIMIC-IV (train 2008–2016, test 2017–2019).
- **Performance:** AUROC and AUPRC for discrimination; the full calibration hierarchy (calibration-in-the-large, slope, flexible calibration via ICI and E50/E90 from a loess smoother); Brier score; decision-curve net benefit. Cluster bootstrap over whole stays for confidence intervals; DeLong's test for AUROC differences.
- **Recalibration:** two-parameter (intercept + slope) logistic update of the predicted log-odds at the external site, estimated out-of-sample by grouped k-fold cross-validation.

---

## Related preprints

- *Calibration drift under cross-institutional deployment: an external validation framework for ICU mortality prediction across MIMIC-IV and eICU.* medRxiv (2026). https://doi.org/10.64898/2026.05.03.26352335
- *Unmeasured but not unbiased: the Missingness Demographic Leakage Audit (MDLA) for calibration-aware fairness evaluation in critical care mortality prediction.* medRxiv (2026). https://doi.org/10.64898/2026.05.01.26352193

---

## Citation

If you use this code or build on this work, please cite the manuscript (under review) and the source databases (MIMIC-IV, eICU-CRD, SICdb) per their PhysioNet citation requirements. A `CITATION.cff` will be added upon publication.

---

## License

The **code** in this repository is released under the terms in [`LICENSE`](LICENSE) (recommended: MIT or Apache-2.0).

This license applies **only** to the source code. It does **not** grant any rights to the MIMIC-IV, eICU-CRD, or SICdb data, which remain governed by their respective PhysioNet licenses and data use agreements. The aggregate figures, tables, and JSON artifacts in this repository are summary statistics derived from those databases and are provided for reproducibility and review; they contain no patient-level information.

---

## Contact

Krutarth Patel — ORCID [0009-0002-8748-8098](https://orcid.org/0009-0002-8748-8098)

For data access, contact PhysioNet directly; the author cannot share restricted data, cache, or trained models.
