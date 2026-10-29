# Housing, education and healthcare demand in Austria by citizenship, 2025-2050

Code and input data to project the dwellings, teachers and hospital beds needed in Austria up to 2050, for Austrian and foreign nationals, by region and for 54 migration scenarios.

## Setup

Python 3.12 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
bash run_pipeline.sh
```

This runs the scripts in `src/` in order and takes about 40 minutes. `ROBUSTNESS=1 bash run_pipeline.sh` also runs the robustness tests, which take about 40 minutes more. Each script writes a log to `logs/`. Results are written to `PQT/`, `XLSX/` and `CSV/`, and the summary tables are in `logs/16_generate_summary_tables.log`.

The hospital model uses the 2019 inpatient records of the Austrian Federal Ministry of Health. These data are licensed and not included here. With the file saved as `XLSX/patients_full_factorial_2019.csv.gz` the whole pipeline runs; without it, the run stops before the hospital model.

## Scripts

1. `prepare_population_inputs.py`: start populations
2. `estimate_migration_scenarios.py`: net migration scenarios
3. `compute_naturalisation_rates.py`: naturalisation rates
4. `project_total_population.py`: total population projection
5. `project_migrant_population.py`: foreign population projection
6. `decompose_native_foreign.py`: population by citizenship
7. `education_demand_percapita.py`, `housing_demand_percapita.py`: observed rates
8. `model_education_demand.py`, `model_housing_demand.py`: teacher and dwelling demand
9. `project_labour_supply.py`: employment by citizenship
10. `model_healthcare_demand.py`: hospital bed demand
11. `age_standardised_ratio.py`: age-standardised utilisation ratios
12. `demand_decomposition.py`, `regional_analysis.py`: decomposition of demand growth and regional regressions
13. `generate_summary_tables.py`: summary tables
14. `robustness_housing.py`, `robustness_education.py`, `robustness_healthcare.py`: robustness tests (results in their logs)

`config.py` holds the file names, and `glm_cov.py` and `oerok_projection.py` are used by the models.

## Data

`Data/` holds the input tables:

- Statistik Austria (STATcube): population, migration, naturalisations, pupils, teachers, housing census and register counts
- Eurostat: EUROPOP2019 and EUROPOP2023 projections
- ÖROK: population and household projections, housing-market regions
- `export_labour_inputs/`: employment rates and sector shares from the 2023 Labour Force Survey (Statistik Austria)

`GEOJSON/` holds the NUTS boundaries from Eurostat GISCO.
