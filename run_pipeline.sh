#!/usr/bin/env bash
# Runs all steps in order from the repository root. Logs go to logs/.
# Usage: bash run_pipeline.sh   (or PY=/path/to/python bash run_pipeline.sh)
# ROBUSTNESS=1 bash run_pipeline.sh also runs the robustness tests.

set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
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
  if [ "$s" = "model_healthcare_demand" ] && [ ! -f XLSX/patients_full_factorial_2019.csv.gz ]; then
    echo "XLSX/patients_full_factorial_2019.csv.gz not found. The hospital model and the steps after it need this file (see README)."
    exit 1
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
