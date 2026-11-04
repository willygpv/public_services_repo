#!/usr/bin/env bash
# The doubled naturalisation scenario of Supplementary Table 9. Run it after run_pipeline.sh.
# Several steps write files with fixed names, so the scenario runs on a copy of the inputs and
# outputs in sensitivity_naturalisation/, and the main results stay as they are.
# Usage: bash run_sensitivity_naturalisation.sh   (or PY=/path/to/python bash run_sensitivity_naturalisation.sh)
# Without the hospital file the hospital step is skipped and the rest still runs (see README).

set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
COPY=sensitivity_naturalisation
PATIENTS=XLSX/patients_full_factorial_2019.csv.gz
RESULTS=logs/sensitivity_naturalisation.log
export PYTHONIOENCODING=utf-8

if [ ! -f CSV/regional_nationality_decomposition_MIXED.csv ]; then
  echo "Run run_pipeline.sh first: the main results are not there yet."
  exit 1
fi
if [ -e "$COPY" ]; then
  echo "$COPY/ already exists. Delete it to run the scenario again."
  exit 1
fi

echo "[$(date +%H:%M:%S)] copying the inputs and outputs to $COPY/"
mkdir -p "$COPY/logs"
cp -r src Data GEOJSON PQT XLSX CSV "$COPY/" || exit 1

STEPS=(
  project_migrant_population
  decompose_native_foreign
  model_education_demand
  model_housing_demand
  model_healthcare_demand
  demand_decomposition
  regional_analysis
)
SCENARIO=(NAT_RATE_MULT=2 PROJECTION_VINTAGE=nat2 TOTAL_POPULATION_VINTAGE=20261023)

n=0
for s in "${STEPS[@]}"; do
  n=$((n + 1))
  if [ "$s" = "model_healthcare_demand" ] && [ ! -f "$PATIENTS" ]; then
    echo "[$(date +%H:%M:%S)] $s skipped: $PATIENTS not found (see README)"
    continue
  fi
  log=$(printf 'logs/%02d_%s.log' "$n" "$s")
  start=$(date +%s)
  echo "[$(date +%H:%M:%S)] $s"
  if ! (cd "$COPY" && env "${SCENARIO[@]}" "$PY" "src/$s.py" > "$log" 2>&1); then
    echo "$s failed, see $COPY/$log"
    exit 1
  fi
  echo "    done in $(( $(date +%s) - start ))s"
done

mkdir -p logs
{
  echo "--- Main projection"
  "$PY" src/sensitivity_naturalisation_summary.py || exit 1
  echo
  echo "--- Doubled naturalisation"
  (cd "$COPY" && env "${SCENARIO[@]}" "$PY" src/sensitivity_naturalisation_summary.py) || exit 1
  echo
  echo "--- Drivers of demand in 2050, percentage points of 2025 demand"
  "$PY" src/sensitivity_naturalisation_decomposition.py . "$COPY" || exit 1
} > "$RESULTS" 2>&1 || { echo "The summary failed, see $RESULTS"; exit 1; }

cat "$RESULTS"
echo "Finished. Results: $RESULTS"
