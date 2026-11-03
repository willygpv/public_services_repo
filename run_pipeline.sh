#!/usr/bin/env bash
# Runs all steps in order from the repository root. Logs go to logs/.
# Usage: bash run_pipeline.sh   (or PY=/path/to/python bash run_pipeline.sh)
# ROBUSTNESS=1 bash run_pipeline.sh also runs the robustness tests.
# Without the hospital file the hospital steps are skipped and the rest still runs (see README).

set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
PATIENTS=XLSX/patients_full_factorial_2019.csv.gz
export PYTHONIOENCODING=utf-8
mkdir -p logs PQT XLSX CSV

STEPS=(
  prepare_population_inputs
  estimate_migration_scenarios
  compute_naturalisation_rates
  project_total_population
  project_migrant_population
  decompose_native_foreign
  education_demand_percapita
  housing_demand_percapita
  model_education_demand
  model_housing_demand
  project_labour_supply
  model_healthcare_demand
  age_standardised_ratio
  demand_decomposition
  regional_analysis
  generate_summary_tables
)
if [ "${ROBUSTNESS:-0}" = "1" ]; then
  STEPS+=(robustness_housing robustness_education robustness_healthcare)
fi

n=0
for s in "${STEPS[@]}"; do
  n=$((n + 1))
  if { [ "$s" = "model_healthcare_demand" ] || [ "$s" = "robustness_healthcare" ]; } && [ ! -f "$PATIENTS" ]; then
    echo "[$(date +%H:%M:%S)] $s skipped: $PATIENTS not found (see README)"
    continue
  fi
  log=$(printf 'logs/%02d_%s.log' "$n" "$s")
  start=$(date +%s)
  echo "[$(date +%H:%M:%S)] $s"
  if ! "$PY" "src/$s.py" > "$log" 2>&1; then
    echo "$s failed, see $log"
    exit 1
  fi
  echo "    done in $(( $(date +%s) - start ))s"
done

echo "Finished. Summary tables: logs/16_generate_summary_tables.log"
if [ ! -f "$PATIENTS" ]; then
  echo "The hospital steps were skipped, so the tables have no healthcare results (see README)."
fi
