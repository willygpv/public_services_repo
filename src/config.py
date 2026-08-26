#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared data paths for the projection pipeline.

Every script downstream of the population projection reads the same vintage of
the same files. Those filenames used to be spelled out in each script, so
regenerating the projection meant editing a dozen scripts by hand, and any
script that was missed went on silently reading the previous vintage. The
vintage is set once, here.

To move the pipeline onto a newly generated projection, change
PROJECTION_VINTAGE and re-run. The Eurostat baseline and the net-migration
scenario file are inputs to the projection rather than outputs of it, so they
carry their own vintages and do not move with it.

Scripts are run from the repository root (``python src/<script>.py``), which
puts ``src/`` on the import path, so ``import config`` resolves to this module.
"""

# ==========================================
# VINTAGES
# ==========================================

# Population projection, as produced by project_migrant_population.py and
# decomposed by decompose_native_foreign.py.
PROJECTION_VINTAGE = '20260721'

# Upstream inputs. These do not move with the projection.
EUROSTAT_VINTAGE = '20251204'
MIGRATION_SCENARIOS_STAMP = '20251203_123931'

# ==========================================
# DERIVED FILENAMES
# ==========================================

# Foreign-born projection, before the native/foreign decomposition.
MIGRANT_STEM = f'negtest_projected_mig_EU_2025_2051_h61_{PROJECTION_VINTAGE}_adjfert'

# Decomposed projection: the file every sector model and figure script reads.
PROJECTION_STEM = f'negtest_projected_simEU_{MIGRANT_STEM}_adjasfr'

# Income-group-stratified variant of the same projection.
PROJECTION_STEM_WITH_IG = (
    f'negtest_projected_simEU_{MIGRANT_STEM}_with_ig_adjasfr'
)

PATH_PROJECTION = f'PQT/{PROJECTION_STEM}.parquet'
PATH_PROJECTION_WITH_IG = f'PQT/{PROJECTION_STEM_WITH_IG}.parquet'
PATH_MIGRANT_PROJECTION = f'PQT/{MIGRANT_STEM}.parquet'

PATH_TOTAL_POPULATION = f'PQT/projected_AT_EU_2025_2051_{EUROSTAT_VINTAGE}.parquet'
PATH_TOTAL_POPULATION_ADJFERT = (
    f'PQT/projected_AT_EU_2025_2051_{EUROSTAT_VINTAGE}_adjfert.parquet'
)
PATH_MIGRATION_SCENARIOS = (
    f'PQT/dynres_net_migration_SensPreCrisis_SensCrisisEra_SensPostCovid'
    f'_scenarios_{MIGRATION_SCENARIOS_STAMP}.parquet'
)

# Naturalisation schedule: written by compute_naturalisation_rates.py, read by
# project_migrant_population.py. Naming the file in one place keeps the two
# halves of that hand-off from drifting apart.
PATH_NATURALISATION = 'XLSX/redistributed_nat_2024.csv'

# Sector-model outputs, named after the projection they were built from. Both
# the housing and the healthcare model write a "_clean" file, so that is the
# name their consumers read.
PATH_EDUCATION_DEMAND = f'PQT/teacher_projections_{PROJECTION_STEM}_NEWTEST-type.parquet'
PATH_HOUSING_DEMAND = f'PQT/dwelling_demand_{PROJECTION_STEM}_clean.parquet'
PATH_HEALTHCARE_DEMAND = f'PQT/healthcare_demand_{PROJECTION_STEM}_clean.parquet'
