#!/usr/bin/env python
# coding: utf-8
"""
Healthcare Demand — EU Scenarios (Data Pipeline)
=================================================
Loads patient bed-day data and population projections, fits a robust Poisson
model, projects healthcare demand with Monte Carlo uncertainty, and saves
the result as a parquet file.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix


# ── Helpers ───────────────────────────────────────────────────────────

def print_column_summary(df: pd.DataFrame, max_unique: int | None = None) -> None:
    """Print dtype, nunique, and sample unique values for every column."""
    for col in df.columns:
        uniques = df[col].unique()[:max_unique] if max_unique else df[col].unique()
        print(
            f"Column '{col}' "
            f"(Type: {df[col].dtype}, Unique: {df[col].nunique()}): "
            f"Unique values -> {uniques}"
        )


def convert_age(age_str: str) -> int:
    """Parse Eurostat age codes ('Y5', 'Y_LT1', 'Y_GE100') to integers."""
    if age_str == "Y_GE100":
        return 100
    if age_str == "Y_LT1":
        return 0
    return int(age_str.lstrip("Y"))


def map_age_to_ag_id(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    """Map single-year ages to 19 standard 5-year age groups (1–19, where 19 = 90+)."""
    df = df.copy()
    df["age_group"] = ((df[age_col] // 5) + 1).clip(upper=19).astype(int)
    return df


# ── Stage 1: Load & prepare patient bed-day data ─────────────────────

def load_bed_days(path: str) -> pd.DataFrame:
    """Load the full-factorial patient file and standardise column names."""
    df = pd.read_csv(path)
    df = df.rename(columns={"geo_2": "NUTS2", "ag_id": "age_group"})
    return df


def build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual records into demographic cohorts."""
    cohort_features = ["year", "sex", "age_group", "Nationality", "NUTS2"]
    return (
        df
        .groupby(cohort_features, dropna=False)
        .agg(
            Population=("patient_no", "count"),
            total_stays=("num_stays", "sum"),
            avg_stays=("num_stays", "mean"),
            total_days=("num_days", "sum"),
            avg_days=("num_days", "mean"),
        )
        .reset_index()
    )


# ── Stage 2: Load & prepare population projections ───────────────────

def load_projections(path: str, demand_year: int) -> pd.DataFrame:
    """Load population projections, convert ages, aggregate to NUTS2."""
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "OBS_VALUE": "Population",
        "geo": "NUTS3",
        "TIME_PERIOD": "year",
    })

    df["age"] = df["age"].apply(convert_age)
    df = map_age_to_ag_id(df, age_col="age")
    df["Population"] = df["Population"].clip(lower=0)

    df = (
        df
        .groupby(["year", "NUTS3", "age_group", "Nationality", "sex", "scenario"],
                 as_index=False)
        ["Population"].sum()
    )

    # Filter to demand horizon and aggregate NUTS3 → NUTS2
    df = df[df["year"] <= demand_year].copy()
    df["NUTS2"] = df["NUTS3"].str[:4]

    return (
        df
        .groupby(["year", "NUTS2", "age_group", "Nationality", "sex", "scenario"],
                 as_index=False)
        ["Population"].sum()
    )


# ── Stage 3: Robust Poisson model ────────────────────────────────────

def fit_robust_poisson(df: pd.DataFrame):
    """
    Fit a Poisson GLM with HC3 robust standard errors on aggregated
    patient data.  Returns (df_agg, model_results).
    """
    print("--- STARTING ROBUST POISSON WORKFLOW ---")
    print(f"1. Aggregating {len(df):,} rows...")

    group_cols = ["age_group", "sex", "Nationality", "NUTS2"]
    for col in group_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df_agg = (
        df
        .groupby(group_cols, observed=True)
        .agg(num_days=("num_days", "sum"), Population=("num_days", "count"))
        .reset_index()
    )
    print(f"   -> Compressed to {len(df_agg):,} groups.")

    print("2. Fitting Poisson with Robust Standard Errors (HC3)...")
    formula = "num_days ~ C(age_group) + C(sex) + C(Nationality) + C(NUTS2) + C(age_group):C(Nationality) + C(age_group):C(sex)"
    model = smf.glm(
        formula=formula,
        data=df_agg,
        offset=np.log(df_agg["Population"]),
        family=sm.families.Poisson(),
    ).fit(cov_type="HC3")

    print("   -> Robust Poisson Model Converged.")
    print(model.summary().tables[1])

    print("\n3. Calculating Predicted Rates...")
    df_agg["predicted_bed_days_per_capita"] = model.predict(
        df_agg, offset=np.zeros(len(df_agg))
    )

    return df_agg, model


# ── Stage 4: Monte Carlo projection ──────────────────────────────────

def project_with_uncertainty(
    df_proj: pd.DataFrame,
    df_agg: pd.DataFrame,
    model_results,
    n_simulations: int = 1000,
) -> pd.DataFrame:
    """
    Draw *n_simulations* coefficient vectors from the model's (robust)
    sampling distribution and project bed-day demand for every row in
    *df_proj*.  Adds Bed_Days_{Lower,Mean,Upper,Std} columns in-place.
    """
    print(f"--- RUNNING {n_simulations} SIMULATIONS FOR ROBUSTNESS ---")
    np.random.seed(42)
    
    params = model_results.params
    cov_matrix = model_results.cov_params()
    simulated_params = np.random.multivariate_normal(params, cov_matrix, n_simulations)

    formula = "C(age_group) + C(sex) + C(Nationality) + C(NUTS2) + C(age_group):C(Nationality) + C(age_group):C(sex)"
    X_proj = dmatrix(formula, df_proj, return_type="dataframe")

    predicted_rates = np.exp(X_proj.values @ simulated_params.T)
    bed_days_matrix = predicted_rates * df_proj["Population"].values[:, None] / 365

    df_proj["Bed_Days_Lower"] = np.percentile(bed_days_matrix, 2.5, axis=1)
    df_proj["Bed_Days_Mean"] = np.mean(bed_days_matrix, axis=1)
    df_proj["Bed_Days_Upper"] = np.percentile(bed_days_matrix, 97.5, axis=1)
    df_proj["Bed_Days_Std"] = np.std(bed_days_matrix, axis=1)

    return df_proj


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    base_year = 2019
    demand_year = 2050

    bed_days_path = r"XLSX/patients_full_factorial_2019.csv.gz"
    projected_path = (
        r"PQT/negtest_projected_simEU_negtest_projected_mig_EU"
        r"_2025_2051_h61_20251204_adjfert_adjasfr.parquet"
    )
    cohorts_out = r"XLSX/patient_cohorts_2019.csv"

    # ── 1. Bed-day data ───────────────────────────────────────────
    df_bd = load_bed_days(bed_days_path)

    # Cohort summary (side output)
    df_cohorts = build_cohorts(df_bd)
    print(df_cohorts.head())
    df_cohorts.to_csv(cohorts_out, index=False)
    print_column_summary(df_cohorts, max_unique=15)

    # Keep only base year
    df_bd = df_bd[df_bd["year"] == base_year]
    print_column_summary(df_bd)

    # ── 2. Population projections ─────────────────────────────────
    df_proj = load_projections(projected_path, demand_year)
    print(df_proj.head())
    print_column_summary(df_proj)

    # ── 3. Fit model ──────────────────────────────────────────────
    df_agg, model = fit_robust_poisson(df_bd)

    # ── 4. Project demand ─────────────────────────────────────────
    df_bd_demand = project_with_uncertainty(df_proj, df_agg, model)

    # ── 5. Save ───────────────────────────────────────────────────
    filename_no_ext = os.path.splitext(os.path.basename(projected_path))[0]
    output_path = f"PQT/healthcare_demand_{filename_no_ext}_clean.parquet"

    df_bd_demand.to_parquet(output_path, index=False, compression="gzip")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()