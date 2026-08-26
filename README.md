# Public Services and Migrants: Projecting Demand in Austria (2025–2051)

A demographic microsimulation framework that projects the demand for public services in Austria under different migration scenarios. The model tracks Austrian and foreign citizens separately across 35 NUTS-3 regions from 2025 to 2051, and translates population change into sector-specific service demand for **education**, **healthcare**, and **housing**.

---

## Overview

Austria's public services face rising and geographically uneven pressure from demographic change and migration. This project builds an integrated pipeline that:

1. **Projects total population** forward using a cohort-component (Leslie matrix) model with region-, age-, sex-, and nationality-specific mortality, fertility, and net migration.
2. **Disaggregates the foreign-citizen share** using differential fertility and income-group stratification, under three migration scenarios (Pre-Crisis, Crisis-Era, Post-COVID).
3. **Translates population projections** into demand for schools/teachers, hospital bed-days, and dwellings using sector-specific statistical models.
4. **Quantifies uncertainty** via Monte Carlo simulation and robustness tests for each sector model.
5. **Compares the two citizenship groups** on a common footing, via a first-order labour-supply projection and an age-standardised utilisation ratio.

---

## Methodology

### Population Model

- **Baseline (total):** `src/project_total_population.py` Cohort-component projection using Eurostat mortality, adjusted fertility (by nationality), sex ratios, and dynamic-residual net migration estimates. Runs 2022–2026 to calibrate and 2025–2051 to project.

- **Foreign-citizen disaggregation:** `src/project_migrant_population.py` Overlays the foreign-citizen population using differential fertility rates, income-group stratification, and naturalisation rates. Three migration scenarios are applied.

- **Native/foreign decomposition:** `src/decompose_native_foreign.py` Decomposes EUROPOP-aligned projections into Austrian and foreign-citizen components.

- **Migration scenarios:** `src/estimate_migration_scenarios.py` Estimates historical net migration distributions and constructs the three scenarios from Statcube micro-data.

- **Naturalisation:** `src/compute_naturalisation_rates.py` Estimates age-, sex- and region-specific naturalisation rates from 2022–2024 administrative data and broadcasts the NUTS-2 rates to single ages at NUTS-3. Fixed income-group shares of naturalisation *events* are converted into group-specific relative propensities, so the schedule reproduces the observed national total rather than deflating it.

### Sector Models

| Sector | Script | Method |
|--------|--------|--------|
| **Education** | `src/model_education_demand.py` | Binomial-empirical GLM for school enrollment rates by age, nationality, and school type; student-to-teacher ratios applied to project teacher demand |
| **Healthcare** | `src/model_healthcare_demand.py` | Robust Poisson GLM (HC3 standard errors) fitted to 2019 bed-day data; 1,000 Monte Carlo draws for uncertainty bands |
| **Housing** | `src/model_housing_demand.py` | Log-transformed per-capita dwelling demand model with ÖROK regional cluster fixed effects |

### Labour Supply and Comparative Indices

- **Labour supply:** `src/project_labour_supply.py` A first-order projection of the employed labour force, `L_k(t) = N_wap(t) × e × s_k`: the projected working-age (15–64) population by five-year age band, times LFS 2023 employment rates, times NACE sectoral shares. Reported for foreign and Austrian nationals separately and for the three service-relevant sectors (P education, Q health and social work, F construction). Uncertainty is a band bounded by the recent-arrival (0–5 years of residence) and settled (>10 years) employment rates.

- **Age-standardised utilisation ratio:** `src/age_standardised_ratio.py` Separates a difference in per-capita utilisation between citizenship groups from a difference in their age composition, by direct age standardisation. Both groups are weighted by a common age structure, so `crude ratio = age-composition factor × ASR` and the ratio is invariant to age structure and to group size. Healthcare is additionally refit from the patient microdata for an exact `exp(Xβ)` and a 1,000-draw Monte Carlo interval.

### Robustness & Validation

Each sector has dedicated robustness scripts (`src/robustness_education.py`, `src/robustness_healthcare.py`, `src/robustness_housing.py`) running 6–8 tests: overdispersion checks, leave-one-out cross-validation, temporal stability, and comparison to observed 2019 data.

---

## Repository Structure

```
.
├── src/                              # All Python scripts
│   ├── config.py                     # Data vintages and derived file paths
│   │
│   ├── ── Core population model ──
│   ├── project_total_population.py
│   ├── project_migrant_population.py
│   ├── decompose_native_foreign.py
│   ├── estimate_migration_scenarios.py
│   ├── compute_naturalisation_rates.py
│   │
│   ├── ── Sector demand models ──
│   ├── model_education_demand.py
│   ├── model_healthcare_demand.py
│   ├── model_housing_demand.py
│   │
│   ├── ── Labour supply & comparative indices ──
│   ├── project_labour_supply.py
│   ├── age_standardised_ratio.py
│   │
│   ├── ── Robustness ──
│   ├── robustness_education.py
│   ├── robustness_healthcare.py
│   ├── robustness_housing.py
│   │
│   └── ── Auxiliary ──
│       ├── education_demand_percapita.py
│       └── housing_demand_percapita.py
│
├── Data/                             # Source data
│   ├── export_labour_inputs/         # LFS 2023 employment rates and sectoral shares
│   ├── Statcube/                     # Austrian population, migration, naturalisations, pupils, teachers
│   ├── Eurostat/                     # Mortality, fertility, sex ratios, net migration
│   └── OEROK/Budget_forecasts/       # Housing cluster codes and household forecasts
│
├── PQT/                              # Population projection and sector-demand outputs
├── XLSX/                             # Model inputs and the summary tables the scripts write out
├── GEOJSON/                          # NUTS-3 regional boundaries
└── PNG/                              # Figures written by the scripts
```

Not all of this arrives with a clone. See [Data availability](#data-availability) for what is included, what has to be obtained from the sources listed below, and what is produced by running the pipeline.

> **All scripts use paths relative to the repository root.** Always run them from the repo root, not from inside `src/`.

> **Input filenames live in one place.** `src/config.py` sets the projection vintage (`PROJECTION_VINTAGE`) and derives every filename from it, so moving the pipeline onto a newly generated projection means changing one line rather than editing each script. The Eurostat baseline and the net-migration scenario file are inputs to the projection rather than outputs of it, so they carry their own, independent vintages.

---

## Running the Pipeline

Run all scripts from the **repository root** in the following order:

```bash
# 1. Prepare naturalisation rates
python src/compute_naturalisation_rates.py

# 2. Run baseline population projection
python src/project_total_population.py

# 3. Run foreign-citizen disaggregation
python src/project_migrant_population.py

# 4. Decompose into native/foreign components
python src/decompose_native_foreign.py

# 5. Run sector demand models
python src/model_education_demand.py
python src/model_healthcare_demand.py
python src/model_housing_demand.py

# 6. Labour supply and comparative indices
python src/project_labour_supply.py
python src/age_standardised_ratio.py
```

### Data availability

This repository holds the code and the small, microdata-free inputs; the bulk data is not redistributed here. Specifically:

- **Included:** the decomposed population projection (`PQT/negtest_projected_simEU_..._20260721_adjfert_adjasfr.parquet`, 36 MB) — the output of steps 1–4 and the input to everything downstream — together with `Data/export_labour_inputs/` (LFS 2023 aggregates), the Statcube naturalisation counts and `XLSX/redistributed_ig_pop_2024.csv`. **`compute_naturalisation_rates.py` (step 1) and `project_labour_supply.py` (step 6) therefore run from a clean clone**, reproducing the observed ~11,400 annual naturalisations and the reported labour-supply figures respectively.
- **Not included:** the sector-model outputs (`teacher_projections`, `dwelling_demand`, `healthcare_demand`; hundreds of MB each) and the patient-level hospital bed-day file, which is licensed microdata. The sector models (step 5) rebuild the first set from the shipped projection given their own inputs; `src/age_standardised_ratio.py` needs both and reports what is missing rather than failing on a missing path.

---

## Data Sources

| Source | Description |
|--------|-------------|
| [Eurostat EUROPOP](https://ec.europa.eu/eurostat) | Mortality, fertility, sex ratios, net migration projections at NUTS-3 level |
| [Statistik Austria (STATcube)](https://www.statistik.at/en/databases/statcube) | Population by age/sex/nationality, internal and international migration, naturalisations, school enrollment, teaching staff |
| [ÖROK](https://www.oerok.gv.at/) | Regional housing cluster classifications and household forecasts |
| Statistik Austria Mikrozensus-Arbeitskräfteerhebung 2023 | Employment rates by citizenship, age band and duration of residence; sectoral shares of employment (`Data/export_labour_inputs/`). The person file itself is licensed microdata and is **not** redistributed here; only the weighted aggregates are, each row naming the dataset in its `source` field. Rates are weighted by the annual person weight `gewjahr` and cover ages 15–64; `group` distinguishes citizenship (`Foreign_total`) from country of birth (`Foreign_born`). Reproducing the tables from source requires microdata access from Statistik Austria |
| [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco) | NUTS-3 boundary GeoJSON (`NUTS_RG_03M_2024_3035.geojson`) |

---

## Requirements

Python 3.10+ with the following packages:

```
pandas
numpy
matplotlib
seaborn
geopandas
scipy
statsmodels
scikit-learn
patsy
pyarrow
requests
```

`cartogram` is imported by `src/model_housing_demand.py` for an optional plotting path and is skipped if absent.