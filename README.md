# Housing, education and healthcare demand in Austria by citizenship, 2025-2050

Code and input data to project the dwellings, teachers and hospital beds needed in Austria up to 2050, for Austrian and foreign nationals, by region and for 54 migration scenarios.

## System requirements

- Python 3.12. Python 3.13 does not work: the pinned packages have no wheels for it, and the install stops when SciPy tries to build from source.
- bash (on Windows, Git Bash).
- At least 16 GB of RAM (peak use 13 GB), or 24 GB with the hospital file (peak use 19 GB). 3 GB of free disk space. No special hardware.
- Tested from a clean clone on a Windows 11 laptop with Python 3.12, 16 cores and 31 GB of RAM. Not yet run on Linux or macOS, although the pinned packages have wheels for both.

## Installation

```bash
git clone https://github.com/willygpv/public_services_repo.git
cd public_services_repo
py -3.12 -m venv .venv          # Linux or macOS: python3.12 -m venv .venv
source .venv/Scripts/activate   # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The install takes about 3 minutes. On Windows, keep the folder path short (for example `C:\tmp`): paths over 260 characters stop some packages from loading.

## Run

From the repository root:

```bash
bash run_pipeline.sh
```

This runs the scripts in `src/` in order on the data in `Data/`, which also serve as the demo, and takes about 35 minutes. `ROBUSTNESS=1 bash run_pipeline.sh` also runs the robustness tests, which take about 50 minutes more. Each script writes a log to `logs/`. Results are written to `PQT/`, `XLSX/` and `CSV/`.

## With and without the hospital file

The hospital model uses the 2019 inpatient records of the Austrian Federal Ministry of Health. These records are protected and are not included here. Researchers need an individual arrangement with the Ministry to use them. With the file saved as `XLSX/patients_full_factorial_2019.csv.gz`, the whole pipeline runs. Without it, the hospital steps are skipped with a message and everything else still runs.

| Results | Without the file | With the file |
|---|---|---|
| Population, housing, education, labour supply | yes | yes |
| Decomposition, regional regressions, summary tables | housing and education | all three sectors |
| Age-standardised ratios | housing and education | all three sectors |
| Hospital bed projection, `Healthcare` rows and columns, hospital robustness tests | no | yes |

## Results of the full run

`results/` holds results of a run with the hospital file, so that the hospital results can be read without the records. They cannot be regenerated without the records. Nothing here is broken down by age or sex, and finer hospital results are not included.

- `summary_tables.txt`: Tables 1 to 6 for all three sectors, as in `logs/16_generate_summary_tables.log`
- `age_standardised_ratio_summary.csv`: age-standardised ratios, rounded to three decimals
- `regression_statistics_MIXED.csv`, `growth_vs_growth_stats_MIXED.csv`, `temporal_regression_stats_MIXED.csv`, `temporal_growth_growth_stats_MIXED.csv`: regional regressions, rounded to four decimals

## Expected results

Values are medians with the range across migration scenarios. `Table 1` and `Table 5` refer to the tables in `logs/16_generate_summary_tables.log`, and `Table 5` is Table 1 of the paper. Numbers can differ in the last digits on other machines.

| Result | Expected value | Where |
|---|---|---|
| Population 2050, millions | 9.5 [9.1, 9.8] | `Table 1`, Population, TOTAL |
| Foreign population 2050, millions | 2.8 [2.5, 3.1] | `Table 1`, Population, Foreign |
| Teachers 2050, thousands | 122.7 [116.7, 129.0] | `Table 1`, Education, TOTAL |
| Dwellings 2050, millions | 4.5 [4.4, 4.7] | `Table 1`, Housing, TOTAL |
| Vienna, foreign share 2050, % | 48.9 [45.8, 51.8] | `Table 5`, Vienna |
| Vienna, housing growth, % | +16.9 [+12.1, +21.8] | `Table 5`, Vienna |
| Vienna, education growth, % | -1.5 [-7.3, +4.4] | `Table 5`, Vienna |
| Age-standardised ratio, housing | 0.96 | `logs/13_age_standardised_ratio.log` |
| Age-standardised ratio, education | 0.95 | `logs/13_age_standardised_ratio.log` |
| With the file: hospital beds 2050, thousands | 34.3 [33.9, 34.7] | `Table 1`, Healthcare, TOTAL |
| With the file: Vienna, healthcare growth, % | +29.5 [+26.4, +32.6] | `Table 5`, Vienna |
| With the file: age-standardised ratio, healthcare | 0.62 [0.60, 0.64] | `logs/13_age_standardised_ratio.log`, exact GLM refit |

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
10. `model_healthcare_demand.py`: hospital bed demand (needs the hospital file)
11. `age_standardised_ratio.py`: age-standardised utilisation ratios
12. `demand_decomposition.py`, `regional_analysis.py`: decomposition of demand growth and regional regressions
13. `generate_summary_tables.py`: summary tables
14. `robustness_housing.py`, `robustness_education.py`, `robustness_healthcare.py`: robustness tests, results in their logs (`robustness_healthcare.py` needs the hospital file)

`config.py` holds the file names, and `glm_cov.py` and `oerok_projection.py` are used by the models.

## Data

`Data/` holds the input tables:

- Statistik Austria (STATcube): population, migration, naturalisations, pupils, teachers, housing census and register counts
- Eurostat: EUROPOP2019 and EUROPOP2023 projections
- ÖROK: population and household projections, housing-market regions
- `export_labour_inputs/`: employment rates and sector shares, aggregated from the 2023 Labour Force Survey microdata (Statistik Austria, STATcube)

`GEOJSON/` holds the NUTS boundaries from Eurostat GISCO.
