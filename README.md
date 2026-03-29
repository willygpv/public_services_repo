# Public Services and Migrants: Projecting Demand in Austria (2025–2051)

A demographic microsimulation framework that projects the demand for public services in Austria under different migration scenarios. The model tracks Austrian and foreign-born populations separately across 35 NUTS-3 regions from 2025 to 2051, and translates population change into sector-specific service demand for **education**, **healthcare**, and **housing**.

---

## Overview

Austria's public services face rising and geographically uneven pressure from demographic change and migration. This project builds an integrated pipeline that:

1. **Projects total population** forward using a cohort-component (Leslie matrix) model with region-, age-, sex-, and nationality-specific mortality, fertility, and net migration.
2. **Disaggregates the foreign-born share** using differential fertility and income-group stratification, under three migration scenarios (Pre-Crisis, Crisis-Era, Post-COVID).
3. **Translates population projections** into demand for schools/teachers, hospital bed-days, and dwellings using sector-specific statistical models.
4. **Quantifies uncertainty** via Monte Carlo simulation and robustness tests for each sector model.
5. **Visualises results** as maps, decomposition charts, regional scatter plots, and summary tables.

---

## Methodology

### Population Model

- **Baseline (total):** `src/project_total_population.py`
  Cohort-component projection using Eurostat mortality, adjusted fertility (by nationality), sex ratios, and dynamic-residual net migration estimates. Runs 2022–2026 to calibrate and 2025–2051 to project.

- **Foreign-born disaggregation:** `src/project_migrant_population.py`
  Overlays the foreign-born population using differential fertility rates, income-group stratification, and naturalisation rates. Three migration scenarios are applied.

- **Native/foreign decomposition:** `src/decompose_native_foreign.py`
  Decomposes EUROPOP-aligned projections into Austrian and foreign-born components.

- **Migration scenarios:** `src/estimate_migration_scenarios.py`
  Estimates historical net migration distributions and constructs the three scenarios from Statcube micro-data.

- **Naturalisation:** `src/compute_naturalisation_rates.py`
  Broadcasts NUTS-2 naturalisation rates to NUTS-3 using population shares.

### Sector Models

| Sector | Script | Method |
|--------|--------|--------|
| **Education** | `src/model_education_demand.py` | Binomial-empirical GLM for school enrollment rates by age, nationality, and school type; student-to-teacher ratios applied to project teacher demand |
| **Healthcare** | `src/model_healthcare_demand.py` | Robust Poisson GLM (HC3 standard errors) fitted to 2019 bed-day data; 1,000 Monte Carlo draws for uncertainty bands |
| **Housing** | `src/model_housing_demand.py` | Log-transformed per-capita dwelling demand model with ÖROK regional cluster fixed effects |

### Robustness & Validation

Each sector has dedicated robustness scripts (`src/robustness_education.py`, `src/robustness_healthcare.py`, `src/robustness_healthcare_extended.py`, `src/robustness_housing.py`) running 6–8 tests: overdispersion checks, leave-one-out cross-validation, temporal stability, and comparison to observed 2019 data.

---

## Repository Structure

```
.
├── src/                              # All Python scripts
│   ├── ── Core population model ──
│   ├── project_total_population.py
│   ├── project_migrant_population.py
│   ├── decompose_native_foreign.py
│   ├── estimate_migration_scenarios.py
│   │
│   ├── ── Sector demand models ──
│   ├── model_education_demand.py
│   ├── model_healthcare_demand.py
│   ├── model_housing_demand.py
│   │
│   ├── ── Visualisation & reporting ──
│   ├── plot_maps.py
│   ├── plot_demand_decomposition.py
│   ├── plot_regional_analysis.py
│   ├── generate_summary_tables.py
│   ├── plot_methods_figures.py
│   │
│   ├── ── Robustness & calibration ──
│   ├── robustness_education.py
│   ├── robustness_healthcare.py
│   ├── robustness_healthcare_extended.py
│   ├── robustness_housing.py
│   ├── calibrate_fertility_mixing.py
│   │
│   └── ── Auxiliary ──
│       ├── education_demand_percapita.py
│       ├── housing_demand_percapita.py
│       └── build_patient_dataset.py
│
├── Data/
│   ├── Eurostat/                     # Mortality, fertility, sex ratios, net migration (CSV)
│   ├── Statcube/                     # Austrian population, migration, pupils, teachers (CSV)
│   └── OEROK/Budget_forecasts/       # Housing cluster codes and household forecasts (XLSX)
├── XLSX/                             # Input CSVs and pre-computed summary files
├── PQT/                              # Parquet files: population projections and sector demand outputs
├── GEOJSON/                          # NUTS-3 regional boundaries
└── CSV/                              # Regional decomposition outputs
```

> **All scripts use paths relative to the repository root.** Always run them from the repo root, not from inside `src/`.

---

## Running the Pipeline

Run all scripts from the **repository root** in the following order:

```bash
# 1. Prepare naturalisation rates
python src/compute_naturalisation_rates.py

# 2. Build patient dataset (only needed if regenerating from raw hospital data)
python src/build_patient_dataset.py

# 3. Run baseline population projection
python src/project_total_population.py

# 4. Run foreign-born disaggregation
python src/project_migrant_population.py

# 5. Decompose into native/foreign components
python src/decompose_native_foreign.py

# 6. Run sector demand models
python src/model_education_demand.py
python src/model_healthcare_demand.py
python src/model_housing_demand.py

# 7. Generate visualisations and tables
python src/plot_demand_decomposition.py
python src/plot_regional_analysis.py
python src/plot_maps.py
python src/generate_summary_tables.py
```

Pre-computed parquet outputs for steps 3–6 are already included in `PQT/`, so the visualisation scripts (step 7) can be run independently without re-running the full pipeline.

---

## Data Sources

| Source | Description |
|--------|-------------|
| [Eurostat EUROPOP](https://ec.europa.eu/eurostat) | Mortality, fertility, sex ratios, net migration projections at NUTS-3 level |
| [Statistik Austria (STATcube)](https://www.statistik.at/en/databases/statcube) | Population by age/sex/nationality, internal and international migration, naturalisations, school enrollment, teaching staff |
| [ÖROK](https://www.oerok.gv.at/) | Regional housing cluster classifications and household forecasts |
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
pyreadr
requests
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn geopandas scipy statsmodels scikit-learn patsy pyarrow pyreadr requests
```

---

## Key Outputs

- **Population projections** by NUTS-3 region, single year of age, sex, and nationality (2025–2051), three migration scenarios
- **Education demand:** enrollment rates and teacher demand by school type, region, and nationality
- **Healthcare demand:** hospital bed-days with Monte Carlo uncertainty bands by region and demographic group
- **Housing demand:** projected dwelling requirements by region and ÖROK cluster
- **Decomposition metrics:** share of total demand attributable to migration vs. demographic ageing
- **Regional maps and scatter plots** for all sectors
