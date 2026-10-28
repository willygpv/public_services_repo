"""File names shared by the pipeline scripts. Run every script from the repository root."""

PROJECTION_VINTAGE = '20261023'
TOTAL_POPULATION_VINTAGE = PROJECTION_VINTAGE
MIGRATION_SCENARIOS_STAMP = '20251203_123931'

MIGRANT_STEM = f'negtest_projected_mig_EU_2025_2051_h61_{PROJECTION_VINTAGE}_adjfert'
PROJECTION_STEM = f'negtest_projected_simEU_{MIGRANT_STEM}_adjasfr'

PATH_PROJECTION = f'PQT/{PROJECTION_STEM}.parquet'
PATH_MIGRANT_PROJECTION = f'PQT/{MIGRANT_STEM}.parquet'

PATH_TOTAL_POPULATION = f'PQT/projected_AT_EU_2025_2051_{TOTAL_POPULATION_VINTAGE}.parquet'
PATH_TOTAL_POPULATION_ADJFERT = (
    f'PQT/projected_AT_EU_2025_2051_{TOTAL_POPULATION_VINTAGE}_adjfert.parquet'
)
PATH_MIGRATION_SCENARIOS = (
    f'PQT/dynres_net_migration_SensPreCrisis_SensCrisisEra_SensPostCovid'
    f'_scenarios_{MIGRATION_SCENARIOS_STAMP}.parquet'
)

PATH_NATURALISATION = 'XLSX/naturalisation_rates_2024.csv'
PATH_FOREIGN_POPULATION_2024 = 'XLSX/foreign_population_2024.csv'
PATH_FOREIGN_POPULATION_2025 = 'XLSX/foreign_population_2025.csv'
PATH_PROJECTION_LOG = f'XLSX/log_{MIGRANT_STEM}.csv'
PATH_POPULATION_2025 = 'XLSX/AT_population_totals_2025.csv.gz'

PATH_OEROK_POPULATION = 'Data/OEROK/Population_forecasts/BevProgn2021-Ergebnisse 1jaehrig.csv'
PATH_OEROK_REGION_CODES = 'Data/OEROK/Population_forecasts/Codes.xlsx'
PATH_OEROK_HOUSEHOLDS = (
    'Data/OEROK/Budget_forecasts/'
    'OEROK_SR_215_Haushaltsprognose_2022-2051_Tabellenband_Hauptvariante.xlsx'
)

PATH_EDUCATION_DEMAND = f'PQT/teacher_projections_{PROJECTION_STEM}_NEWTEST-type.parquet'
PATH_HOUSING_DEMAND = f'PQT/dwelling_demand_{PROJECTION_STEM}.parquet'
PATH_HEALTHCARE_DEMAND = f'PQT/healthcare_demand_{PROJECTION_STEM}_clean.parquet'

PATH_PATIENTS = 'XLSX/patients_full_factorial_2019.csv.gz'
PATH_PATIENT_COHORTS = 'XLSX/patient_cohorts_2019.csv'

HEALTHCARE_RHS = (
    'C(age_group) + C(sex) + C(Nationality) + C(NUTS2)'
    ' + C(age_group):C(Nationality) + C(age_group):C(sex)'
    ' + C(NUTS2):C(age_group) + C(NUTS2):C(Nationality)'
)

PATH_REGIONAL_DECOMPOSITION = 'CSV/regional_nationality_decomposition_MIXED.csv'
PATH_REGIONAL_DECOMPOSITION_SCENARIOS = 'CSV/regional_nationality_decomposition_MIXED_per_scenario.csv'
PATH_NATIONAL_METRICS = 'XLSX/final_demographics_data_processed.csv'
