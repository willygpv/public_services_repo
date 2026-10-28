"""Poisson model of hospital bed-days on the 2019 patient data and projected bed-day demand. Needs the licensed patient file (see README)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats

import config
import glm_cov


def print_column_summary(df: pd.DataFrame, max_unique: int | None = None) -> None:
    for col in df.columns:
        uniques = df[col].unique()[:max_unique] if max_unique else df[col].unique()
        print(
            f"Column '{col}' "
            f"(Type: {df[col].dtype}, Unique: {df[col].nunique()}): "
            f"Unique values -> {uniques}"
        )


def convert_age(age_str: str) -> int:
    if age_str == "Y_GE100":
        return 100
    if age_str == "Y_LT1":
        return 0
    return int(age_str.lstrip("Y"))


def map_age_to_ag_id(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = ((df[age_col] // 5) + 1).clip(upper=19).astype(int)
    return df


def load_bed_days(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"geo_2": "NUTS2", "ag_id": "age_group"})
    return df


def build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
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


def load_projections(path: str, demand_year: int) -> pd.DataFrame:
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

    df = df[df["year"] <= demand_year].copy()
    df["NUTS2"] = df["NUTS3"].str[:4]

    return (
        df
        .groupby(["year", "NUTS2", "age_group", "Nationality", "sex", "scenario"],
                 as_index=False)
        ["Population"].sum()
    )


def fit_robust_poisson(df: pd.DataFrame):
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
    formula = f"num_days ~ {config.HEALTHCARE_RHS}"
    model = smf.glm(
        formula=formula,
        data=df_agg,
        offset=np.log(df_agg["Population"]),
        family=sm.families.Poisson(),
    ).fit()

    print("   -> Robust Poisson Model Converged.")
    se = np.sqrt(np.diag(glm_cov.cov_hc3(model)))
    z = model.params / se
    print(pd.DataFrame({"coef": model.params, "se_HC3": se, "z": z,
                        "P>|z|": 2 * stats.norm.sf(np.abs(z))}).to_string(float_format="%.4f"))

    print("\n3. Calculating Predicted Rates...")
    df_agg["predicted_bed_days_per_capita"] = model.predict(
        df_agg, offset=np.zeros(len(df_agg))
    )

    return df_agg, model


def project_with_uncertainty(
    df_proj: pd.DataFrame,
    df_agg: pd.DataFrame,
    model_results,
    n_simulations: int = 1000,
) -> pd.DataFrame:
    print(f"--- RUNNING {n_simulations} SIMULATIONS FOR ROBUSTNESS ---")
    np.random.seed(42)

    params = model_results.params
    cov_matrix = glm_cov.cov_hc3(model_results)
    simulated_params = np.random.multivariate_normal(params, cov_matrix, n_simulations)

    X_proj = dmatrix(config.HEALTHCARE_RHS, df_proj, return_type="dataframe")
    assert list(X_proj.columns) == list(params.index), "projection design does not match the model"

    predicted_rates = np.exp(X_proj.values @ simulated_params.T)
    bed_days_matrix = predicted_rates * df_proj["Population"].values[:, None] / 365

    df_proj["Bed_Days_Lower"] = np.percentile(bed_days_matrix, 2.5, axis=1)
    df_proj["Bed_Days_Mean"] = (np.exp(X_proj.values @ params.values)
                                * df_proj["Population"].values / 365)
    df_proj["Bed_Days_Upper"] = np.percentile(bed_days_matrix, 97.5, axis=1)
    df_proj["Bed_Days_Std"] = np.std(bed_days_matrix, axis=1)

    return df_proj


def main() -> None:
    base_year = 2019
    demand_year = 2050

    bed_days_path = config.PATH_PATIENTS
    projected_path = config.PATH_PROJECTION
    cohorts_out = config.PATH_PATIENT_COHORTS

    df_bd = load_bed_days(bed_days_path)

    df_cohorts = build_cohorts(df_bd)
    print(df_cohorts.head())
    df_cohorts.to_csv(cohorts_out, index=False)
    print_column_summary(df_cohorts, max_unique=15)

    df_bd = df_bd[df_bd["year"] == base_year]

    df_proj = load_projections(projected_path, demand_year)
    print(df_proj.head())
    print_column_summary(df_proj)

    df_agg, model = fit_robust_poisson(df_bd)

    df_bd_demand = project_with_uncertainty(df_proj, df_agg, model)

    output_path = config.PATH_HEALTHCARE_DEMAND

    df_bd_demand.to_parquet(output_path, index=False, compression="gzip")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
