#!/usr/bin/env python
# coding: utf-8

"""
Robustness Tests for Healthcare Demand Projection Model
========================================================
Runs 8 robustness/sensitivity tests on the Poisson healthcare model
and prints structured results for analysis.

Requirements: same environment as the main healthcare script.
Place this file in the same directory as the main script so paths align.
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION (must match main script)
# ==========================================
BASE_YEAR = 2019
DEMAND_YEAR = 2050
N_BOOTSTRAP = 200
N_MC_SIMS = 500

BED_DAYS_PATH = r'XLSX/patients_full_factorial_2019.csv.gz'
PROJECTED_PATH = (
    r'XLSX/negtest_projected_simEU_negtest_projected_mig_EU'
    r'_2025_2051_h61_20251204_adjfert_adjasfr.csv'
)

FORMULA_ADDITIVE = "num_days ~ C(age_group) + C(sex) + C(Nationality) + C(NUTS2)"

# ==========================================
# DATA LOADING (reused from main script)
# ==========================================

def convert_age(age_str):
    if age_str == "Y_GE100": return 100
    if age_str == "Y_LT1": return 0
    return int(str(age_str).lstrip("Y"))


def map_age_to_ag_id(df, age_col="age"):
    df = df.copy()
    df["age_group"] = ((df[age_col] // 5) + 1).clip(upper=19).astype(int)
    return df


def load_bed_days():
    print("--- Loading patient bed-day data ---")
    df = pd.read_csv(BED_DAYS_PATH)
    df = df.rename(columns={"geo_2": "NUTS2", "ag_id": "age_group"})
    df = df[df["year"] == BASE_YEAR].copy()
    print(f"  Rows: {len(df):,}")
    print(f"  NUTS2 regions: {sorted(df['NUTS2'].unique())}")
    print(f"  Age groups: {sorted(df['age_group'].unique())}")
    print(f"  Nationalities: {df['Nationality'].unique()}")
    return df


def aggregate_cohorts(df):
    """Aggregate individual records into demographic cohorts for modelling."""
    group_cols = ["age_group", "sex", "Nationality", "NUTS2"]
    for col in group_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df_agg = (
        df.groupby(group_cols, observed=True)
        .agg(num_days=("num_days", "sum"), Population=("num_days", "count"))
        .reset_index()
    )
    return df_agg


def load_projections():
    print("--- Loading population projections ---")
    df = pd.read_csv(PROJECTED_PATH)
    df = df.rename(columns={
        "OBS_VALUE": "Population", "geo": "NUTS3",
        "TIME_PERIOD": "year",
    })
    df["age"] = df["age"].apply(convert_age)
    df = map_age_to_ag_id(df, age_col="age")
    df["Population"] = df["Population"].clip(lower=0)
    df = df.groupby(
        ["year", "NUTS3", "age_group", "Nationality", "sex", "scenario"],
        as_index=False
    )["Population"].sum()
    df = df[df["year"] <= DEMAND_YEAR].copy()
    df["NUTS2"] = df["NUTS3"].str[:4]
    df = df.groupby(
        ["year", "NUTS2", "age_group", "Nationality", "sex", "scenario"],
        as_index=False
    )["Population"].sum()
    print(f"  Scenarios: {df['scenario'].nunique()}")
    print(f"  Years: {df['year'].min()}--{df['year'].max()}")
    return df


# ==========================================
# CORE MODEL FUNCTIONS
# ==========================================

def fit_poisson(df_agg, formula=FORMULA_ADDITIVE, cov_type="HC3"):
    """Fit Poisson GLM with offset. Returns (result, design_info)."""
    model = smf.glm(
        formula=formula,
        data=df_agg,
        offset=np.log(df_agg["Population"].clip(lower=1)),
        family=sm.families.Poisson(),
    ).fit(cov_type=cov_type)
    return model


def predict_rates(model, df_agg):
    """Predict per-capita bed-day rates (no offset)."""
    return model.predict(df_agg, offset=np.zeros(len(df_agg)))


def project_demand_2050(model, df_proj_2050):
    """Project total annual bed-days at 2050 for a given projection slice."""
    formula_rhs = "C(age_group) + C(sex) + C(Nationality) + C(NUTS2)"
    X = dmatrix(formula_rhs, df_proj_2050, return_type="dataframe")
    rates = np.exp(X.values @ model.params.values)
    return (rates * df_proj_2050["Population"].values).sum()


# ==========================================
# TEST 1: OVERDISPERSION DIAGNOSTIC
# ==========================================

def test1_overdispersion(df_bd, df_agg):
    print("\n" + "=" * 80)
    print("TEST 1: OVERDISPERSION DIAGNOSTIC")
    print("=" * 80)

    # Fit Poisson
    poisson_model = fit_poisson(df_agg)
    poisson_aic = poisson_model.aic
    poisson_dev = poisson_model.deviance
    poisson_pearson = poisson_model.pearson_chi2
    df_resid = poisson_model.df_resid

    # Dispersion estimates
    pearson_disp = poisson_pearson / df_resid
    deviance_disp = poisson_dev / df_resid

    print(f"\n--- Poisson Model Diagnostics ---")
    print(f"  AIC:                     {poisson_aic:,.1f}")
    print(f"  Deviance:                {poisson_dev:,.1f}")
    print(f"  Pearson chi-sq:          {poisson_pearson:,.1f}")
    print(f"  df_resid:                {df_resid}")
    print(f"  Pearson dispersion:      {pearson_disp:.4f}")
    print(f"  Deviance dispersion:     {deviance_disp:.4f}")
    print(f"  (Dispersion = 1.0 under Poisson; >1 indicates overdispersion)")

    # Fit Negative Binomial
    print(f"\n--- Negative Binomial Comparison ---")
    try:
        nb_model = smf.glm(
            formula=FORMULA_ADDITIVE,
            data=df_agg,
            offset=np.log(df_agg["Population"].clip(lower=1)),
            family=sm.families.NegativeBinomial(alpha=1.0),
        ).fit()

        nb_aic = nb_model.aic
        nb_dev = nb_model.deviance

        print(f"  NB AIC:                  {nb_aic:,.1f}")
        print(f"  NB Deviance:             {nb_dev:,.1f}")
        print(f"  AIC difference (P - NB): {poisson_aic - nb_aic:+,.1f}")

        # Compare key coefficients
        print(f"\n--- Coefficient Comparison: Poisson vs Negative Binomial ---")
        print(f"{'Parameter':>35s} {'Poisson':>10s} {'NegBin':>10s} {'Diff':>10s}")
        for param in poisson_model.params.index:
            p_val = poisson_model.params[param]
            if param in nb_model.params.index:
                nb_val = nb_model.params[param]
                print(f"  {param:>33s} {p_val:10.4f} {nb_val:10.4f} {p_val - nb_val:+10.4f}")

        # Specifically report nationality coefficient
        nat_params = [p for p in poisson_model.params.index if 'Nationality' in str(p)]
        if nat_params:
            p_nat = poisson_model.params[nat_params[0]]
            nb_nat = nb_model.params[nat_params[0]]
            print(f"\n  Nationality coefficient (Poisson):  {p_nat:.4f} -> rate ratio = {np.exp(p_nat):.4f}")
            print(f"  Nationality coefficient (NegBin):   {nb_nat:.4f} -> rate ratio = {np.exp(nb_nat):.4f}")

    except Exception as e:
        print(f"  NegBin fitting failed: {e}")
        # Try with different alpha values
        for alpha in [0.1, 0.5, 2.0, 5.0]:
            try:
                nb_m = smf.glm(
                    formula=FORMULA_ADDITIVE, data=df_agg,
                    offset=np.log(df_agg["Population"].clip(lower=1)),
                    family=sm.families.NegativeBinomial(alpha=alpha),
                ).fit()
                print(f"  NB(alpha={alpha}): AIC={nb_m.aic:,.1f}, Deviance={nb_m.deviance:,.1f}")
            except:
                print(f"  NB(alpha={alpha}): FAILED")

    # Residual analysis
    print(f"\n--- Residual Summary (Poisson) ---")
    resid_pearson = poisson_model.resid_pearson
    resid_deviance = poisson_model.resid_deviance
    print(f"  Pearson residuals:  mean={np.mean(resid_pearson):.4f}, "
          f"std={np.std(resid_pearson):.4f}, "
          f"min={np.min(resid_pearson):.2f}, max={np.max(resid_pearson):.2f}")
    print(f"  Deviance residuals: mean={np.mean(resid_deviance):.4f}, "
          f"std={np.std(resid_deviance):.4f}, "
          f"min={np.min(resid_deviance):.2f}, max={np.max(resid_deviance):.2f}")

    return poisson_model


# ==========================================
# TEST 2: ALTERNATIVE SPECIFICATIONS
# ==========================================

def test2_alternative_specs(df_agg):
    print("\n" + "=" * 80)
    print("TEST 2: ALTERNATIVE MODEL SPECIFICATIONS")
    print("=" * 80)

    specs = {
        'Baseline (additive)':
            "num_days ~ C(age_group) + C(sex) + C(Nationality) + C(NUTS2)",
        'Nat x Age':
            "num_days ~ C(age_group) * C(Nationality) + C(sex) + C(NUTS2)",
        'Sex x Age':
            "num_days ~ C(age_group) * C(sex) + C(Nationality) + C(NUTS2)",
        'Nat x Region':
            "num_days ~ C(age_group) + C(sex) + C(Nationality) * C(NUTS2)",
        'Nat x Age + Sex x Age':
            "num_days ~ C(age_group) * C(Nationality) + C(age_group) * C(sex) + C(NUTS2)",
        'Full interactions':
            "num_days ~ C(age_group) * C(Nationality) + C(age_group) * C(sex) + C(Nationality) * C(NUTS2)",
    }

    print(f"\n--- Model Comparison ---")
    print(f"{'Specification':>30s} {'AIC':>14s} {'Deviance':>14s} {'Pearson_disp':>14s} {'n_params':>10s}")

    results = {}
    for name, formula in specs.items():
        try:
            model = smf.glm(
                formula=formula, data=df_agg,
                offset=np.log(df_agg["Population"].clip(lower=1)),
                family=sm.families.Poisson(),
            ).fit(cov_type="HC3")
            disp = model.pearson_chi2 / model.df_resid
            print(f"  {name:>28s} {model.aic:14,.1f} {model.deviance:14,.1f} "
                  f"{disp:14.4f} {len(model.params):10d}")
            results[name] = model
        except Exception as e:
            print(f"  {name:>28s} FAILED: {e}")

    # Deep dive: Nationality x Age interaction
    if 'Nat x Age' in results:
        print(f"\n--- Nationality x Age Interaction Effects ---")
        model_ia = results['Nat x Age']
        interaction_params = {k: v for k, v in model_ia.params.items()
                             if 'Nationality' in str(k) and 'age_group' in str(k)}

        if interaction_params:
            print(f"{'Interaction term':>55s} {'Coeff':>10s} {'SE':>10s} {'p-value':>10s} {'RR':>8s}")
            for param, coef in sorted(interaction_params.items()):
                se = model_ia.bse[param]
                pval = model_ia.pvalues[param]
                rr = np.exp(coef)
                sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
                print(f"  {param:>53s} {coef:10.4f} {se:10.4f} {pval:10.4f} {rr:8.4f} {sig}")

            # Predicted rates by age group and nationality under interaction model
            print(f"\n--- Predicted Per-Capita Rates: Additive vs Interaction Model ---")
            baseline = results.get('Baseline (additive)')
            if baseline:
                pred_base = predict_rates(baseline, df_agg)
                pred_ia = predict_rates(model_ia, df_agg)
                df_compare = df_agg[['age_group', 'Nationality', 'Population']].copy()
                df_compare['Rate_Additive'] = pred_base
                df_compare['Rate_Interaction'] = pred_ia

                summary = df_compare.groupby(['age_group', 'Nationality']).apply(
                    lambda g: pd.Series({
                        'Rate_Add': np.average(g['Rate_Additive'], weights=g['Population']),
                        'Rate_IA': np.average(g['Rate_Interaction'], weights=g['Population']),
                    })
                ).reset_index()

                pivot_add = summary.pivot(index='age_group', columns='Nationality', values='Rate_Add')
                pivot_ia = summary.pivot(index='age_group', columns='Nationality', values='Rate_IA')

                print(f"{'AG':>4s} | {'Additive':^30s} | {'Interaction':^30s} | {'Ratio IA/Add':^20s}")
                nats = pivot_add.columns.tolist()
                header_sub = " | ".join([f"{'Aus':>8s} {'For':>8s} {'Ratio':>8s}"] * 2) + f" | {'Aus':>8s} {'For':>8s}"
                print(f"     | {header_sub}")

                for ag in sorted(summary['age_group'].unique()):
                    aus_add = pivot_add.loc[ag, nats[0]] if ag in pivot_add.index else np.nan
                    for_add = pivot_add.loc[ag, nats[1]] if ag in pivot_add.index and len(nats) > 1 else np.nan
                    aus_ia = pivot_ia.loc[ag, nats[0]] if ag in pivot_ia.index else np.nan
                    for_ia = pivot_ia.loc[ag, nats[1]] if ag in pivot_ia.index and len(nats) > 1 else np.nan
                    ratio_add = for_add / aus_add if aus_add > 0 else np.nan
                    ratio_ia = for_ia / aus_ia if aus_ia > 0 else np.nan
                    print(f"  {ag:>3d} | {aus_add:8.3f} {for_add:8.3f} {ratio_add:8.3f} | "
                          f"{aus_ia:8.3f} {for_ia:8.3f} {ratio_ia:8.3f} | "
                          f"{aus_ia/aus_add if aus_add > 0 else np.nan:8.3f} "
                          f"{for_ia/for_add if for_add > 0 else np.nan:8.3f}")

    return results


# ==========================================
# TEST 3: LEAVE-ONE-REGION-OUT
# ==========================================

def test3_leave_region_out(df_agg):
    print("\n" + "=" * 80)
    print("TEST 3: LEAVE-ONE-REGION-OUT CROSS-VALIDATION")
    print("=" * 80)

    regions = sorted(df_agg['NUTS2'].unique())
    print(f"Regions: {regions}")

    # Use formula without region FE so we can predict held-out region
    formula_no_region = "num_days ~ C(age_group) + C(sex) + C(Nationality)"

    results = []
    for region in regions:
        df_train = df_agg[df_agg['NUTS2'] != region].copy()
        df_test = df_agg[df_agg['NUTS2'] == region].copy()

        try:
            model = smf.glm(
                formula=formula_no_region, data=df_train,
                offset=np.log(df_train["Population"].clip(lower=1)),
                family=sm.families.Poisson(),
            ).fit(cov_type="HC3")

            pred_rates = model.predict(df_test, offset=np.zeros(len(df_test)))
            pred_days = pred_rates * df_test['Population'].values
            obs_days = df_test['num_days'].values

            mask = obs_days > 0
            mape = np.mean(np.abs((obs_days[mask] - pred_days[mask]) / obs_days[mask])) * 100
            total_obs = obs_days.sum()
            total_pred = pred_days.sum()
            bias_pct = ((total_pred - total_obs) / total_obs) * 100
            rmse = np.sqrt(np.mean((obs_days - pred_days) ** 2))

            results.append({
                'Region': region, 'MAPE': mape, 'RMSE': rmse,
                'Total_Obs': total_obs, 'Total_Pred': total_pred,
                'Bias_pct': bias_pct, 'N_cohorts': len(df_test)
            })
        except Exception as e:
            print(f"  {region}: FAILED ({e})")

    df_res = pd.DataFrame(results)
    print(f"\n--- Leave-One-Region-Out Results ---")
    print(df_res.to_string(index=False, float_format='%.2f'))
    print(f"\nMean MAPE:  {df_res['MAPE'].mean():.2f}%")
    print(f"Mean Bias:  {df_res['Bias_pct'].mean():+.2f}%")
    print(f"Worst MAPE: {df_res.loc[df_res['MAPE'].idxmax(), 'Region']} ({df_res['MAPE'].max():.2f}%)")
    print(f"Best MAPE:  {df_res.loc[df_res['MAPE'].idxmin(), 'Region']} ({df_res['MAPE'].min():.2f}%)")

    # Also report with region FE (in-sample fit)
    print(f"\n--- In-Sample Fit (with region FE, for comparison) ---")
    model_full = fit_poisson(df_agg)
    pred_full = predict_rates(model_full, df_agg)
    pred_days_full = pred_full * df_agg['Population'].values
    obs_days_full = df_agg['num_days'].values
    mask_full = obs_days_full > 0
    mape_full = np.mean(np.abs(
        (obs_days_full[mask_full] - pred_days_full[mask_full]) / obs_days_full[mask_full]
    )) * 100
    print(f"  In-sample MAPE: {mape_full:.2f}%")
    print(f"  In-sample total obs: {obs_days_full.sum():,.0f}, pred: {pred_days_full.sum():,.0f}")

    return df_res


# ==========================================
# TEST 4: BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================

def test4_bootstrap(df_agg, df_proj, n_boot=N_BOOTSTRAP):
    print("\n" + "=" * 80)
    print(f"TEST 4: BOOTSTRAP CONFIDENCE INTERVALS (n={n_boot})")
    print("=" * 80)

    # Reference model
    ref_model = fit_poisson(df_agg)
    ref_params = ref_model.params

    # Get one projection scenario at 2050
    one_scen = df_proj['scenario'].unique()[0]
    df_2050 = df_proj[
        (df_proj['scenario'] == one_scen) & (df_proj['year'] == DEMAND_YEAR)
    ].copy()
    # Aggregate to match model structure
    proj_cells = df_2050.groupby(
        ['age_group', 'sex', 'Nationality', 'NUTS2'], as_index=False
    )['Population'].sum()

    # Reference prediction
    formula_rhs = "C(age_group) + C(sex) + C(Nationality) + C(NUTS2)"
    X_proj = dmatrix(formula_rhs, proj_cells, return_type="dataframe")
    ref_rates = np.exp(X_proj.values @ ref_params.values)
    ref_total_beds = (ref_rates * proj_cells['Population'].values).sum() / 365

    # Bootstrap
    n_cohorts = len(df_agg)
    boot_params = []
    boot_total_beds = []
    boot_nat_coef = []

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            print(f"  Bootstrap iteration {b + 1}/{n_boot}...")

        idx = np.random.choice(n_cohorts, size=n_cohorts, replace=True)
        df_boot = df_agg.iloc[idx].copy().reset_index(drop=True)

        try:
            model_b = smf.glm(
                formula=FORMULA_ADDITIVE, data=df_boot,
                offset=np.log(df_boot["Population"].clip(lower=1)),
                family=sm.families.Poisson(),
            ).fit()

            boot_params.append(model_b.params.values)

            # Project to 2050
            rates_b = np.exp(X_proj.values @ model_b.params.values)
            total_beds_b = (rates_b * proj_cells['Population'].values).sum() / 365
            boot_total_beds.append(total_beds_b)

            # Track nationality coefficient
            nat_params = [p for p in model_b.params.index if 'Nationality' in str(p)]
            if nat_params:
                boot_nat_coef.append(model_b.params[nat_params[0]])

        except:
            pass

    boot_params = np.array(boot_params)
    n_success = len(boot_params)
    print(f"\nSuccessful iterations: {n_success}/{n_boot}")

    if n_success < 10:
        print("Too few successful iterations.")
        return

    # Parameter CIs
    print(f"\n--- Bootstrap 95% CIs for Poisson Coefficients ---")
    print(f"{'Parameter':>40s} {'Estimate':>10s} {'Boot_2.5%':>10s} {'Boot_97.5%':>11s} {'Boot_SE':>10s}")
    for i, name in enumerate(ref_params.index):
        est = ref_params.iloc[i]
        lo = np.percentile(boot_params[:, i], 2.5)
        hi = np.percentile(boot_params[:, i], 97.5)
        se = np.std(boot_params[:, i])
        print(f"  {name:>38s} {est:10.4f} {lo:10.4f} {hi:11.4f} {se:10.4f}")

    # Total beds uncertainty
    arr = np.array(boot_total_beds)
    print(f"\n--- Bootstrap Uncertainty in Projected Daily Beds (2050, one scenario) ---")
    print(f"  Reference:  {ref_total_beds:,.0f}")
    print(f"  Median:     {np.median(arr):,.0f}")
    print(f"  2.5%:       {np.percentile(arr, 2.5):,.0f}")
    print(f"  97.5%:      {np.percentile(arr, 97.5):,.0f}")
    print(f"  CV:         {np.std(arr) / np.mean(arr) * 100:.2f}%")
    print(f"  Width (95% CI): {np.percentile(arr, 97.5) - np.percentile(arr, 2.5):,.0f} beds")

    # Nationality coefficient
    if boot_nat_coef:
        arr_nat = np.array(boot_nat_coef)
        print(f"\n--- Bootstrap: Nationality Coefficient ---")
        print(f"  Estimate:   {ref_params[[p for p in ref_params.index if 'Nationality' in str(p)][0]]:.4f}")
        print(f"  Boot 95%CI: [{np.percentile(arr_nat, 2.5):.4f}, {np.percentile(arr_nat, 97.5):.4f}]")
        print(f"  Rate ratio: {np.exp(np.median(arr_nat)):.4f} "
              f"[{np.exp(np.percentile(arr_nat, 2.5)):.4f}, {np.exp(np.percentile(arr_nat, 97.5)):.4f}]")


# ==========================================
# TEST 5: COUNTERFACTUAL — FOREIGN RATES = NATIVE RATES
# ==========================================

def test5_counterfactual(df_agg, df_proj):
    print("\n" + "=" * 80)
    print("TEST 5: COUNTERFACTUAL — WHAT IF FOREIGN = NATIVE UTILISATION RATES?")
    print("=" * 80)

    # Fit reference model
    model = fit_poisson(df_agg)

    # Identify nationality parameter
    nat_params = [p for p in model.params.index if 'Nationality' in str(p)]
    if not nat_params:
        print("  No nationality parameter found.")
        return

    nat_coef = model.params[nat_params[0]]
    print(f"\n  Nationality coefficient: {nat_coef:.4f}")
    print(f"  Rate ratio (foreign/Austrian): {np.exp(nat_coef):.4f}")
    print(f"  Foreign nationals use {(1 - np.exp(nat_coef)) * 100:.1f}% fewer bed-days per capita")

    # Project under actual vs counterfactual for multiple scenarios
    formula_rhs = "C(age_group) + C(sex) + C(Nationality) + C(NUTS2)"

    print(f"\n--- Counterfactual Demand at {DEMAND_YEAR} (all scenarios) ---")
    scenarios = df_proj[df_proj['year'] == DEMAND_YEAR]['scenario'].unique()

    cf_results = []
    for scen in scenarios:
        df_2050 = df_proj[
            (df_proj['scenario'] == scen) & (df_proj['year'] == DEMAND_YEAR)
        ].copy()
        proj_cells = df_2050.groupby(
            ['age_group', 'sex', 'Nationality', 'NUTS2'], as_index=False
        )['Population'].sum()

        X = dmatrix(formula_rhs, proj_cells, return_type="dataframe")

        # Actual rates
        rates_actual = np.exp(X.values @ model.params.values)
        beds_actual = (rates_actual * proj_cells['Population'].values).sum() / 365

        # Counterfactual: set nationality coefficient to 0
        params_cf = model.params.copy()
        params_cf[nat_params[0]] = 0.0
        rates_cf = np.exp(X.values @ params_cf.values)
        beds_cf = (rates_cf * proj_cells['Population'].values).sum() / 365

        # Decompose by nationality
        mask_foreign = proj_cells['Nationality'] == 'Foreign country'
        beds_foreign_actual = (rates_actual[mask_foreign.values] * proj_cells.loc[mask_foreign, 'Population'].values).sum() / 365
        beds_foreign_cf = (rates_cf[mask_foreign.values] * proj_cells.loc[mask_foreign, 'Population'].values).sum() / 365

        cf_results.append({
            'scenario': scen,
            'Beds_Actual': beds_actual,
            'Beds_Counterfactual': beds_cf,
            'Diff_beds': beds_cf - beds_actual,
            'Diff_pct': ((beds_cf - beds_actual) / beds_actual) * 100,
            'Foreign_Actual': beds_foreign_actual,
            'Foreign_CF': beds_foreign_cf,
        })

    df_cf = pd.DataFrame(cf_results)

    print(f"\n--- Summary Across All Scenarios ---")
    print(f"  Actual beds (median):           {df_cf['Beds_Actual'].median():,.0f}")
    print(f"  Counterfactual beds (median):   {df_cf['Beds_Counterfactual'].median():,.0f}")
    print(f"  Additional beds if equal rates: {df_cf['Diff_beds'].median():,.0f} "
          f"({df_cf['Diff_pct'].median():+.2f}%)")
    print(f"  Range of additional beds:       [{df_cf['Diff_beds'].min():,.0f}, {df_cf['Diff_beds'].max():,.0f}]")
    print(f"\n  Foreign beds (actual, median):        {df_cf['Foreign_Actual'].median():,.0f}")
    print(f"  Foreign beds (counterfactual, median): {df_cf['Foreign_CF'].median():,.0f}")
    print(f"  Increase in foreign demand:            {(df_cf['Foreign_CF'].median() - df_cf['Foreign_Actual'].median()):,.0f} "
          f"({((df_cf['Foreign_CF'].median() / df_cf['Foreign_Actual'].median()) - 1) * 100:+.1f}%)")

    # Also compute for base year (2019-equivalent population)
    print(f"\n--- Counterfactual Applied to Base Year Population ---")
    pred_actual = predict_rates(model, df_agg)
    beds_base_actual = (pred_actual * df_agg['Population'].values).sum() / 365

    params_cf = model.params.copy()
    params_cf[nat_params[0]] = 0.0
    # Need to manually predict with modified params
    X_base = dmatrix(formula_rhs, df_agg, return_type="dataframe")
    rates_cf_base = np.exp(X_base.values @ params_cf.values)
    beds_base_cf = (rates_cf_base * df_agg['Population'].values).sum() / 365

    print(f"  Base year actual:        {beds_base_actual:,.0f} daily beds")
    print(f"  Base year counterfactual:{beds_base_cf:,.0f} daily beds")
    print(f"  Difference:              {beds_base_cf - beds_base_actual:,.0f} ({((beds_base_cf / beds_base_actual) - 1) * 100:+.2f}%)")


# ==========================================
# TEST 6: VARIANCE DECOMPOSITION
# ==========================================

def test6_variance_decomposition(df_agg, df_proj, n_mc=N_MC_SIMS):
    print("\n" + "=" * 80)
    print("TEST 6: VARIANCE DECOMPOSITION (Demographic Scenario vs Parameter Uncertainty)")
    print("=" * 80)

    model = fit_poisson(df_agg)
    params = model.params
    cov = model.cov_params()
    formula_rhs = "C(age_group) + C(sex) + C(Nationality) + C(NUTS2)"

    np.random.seed(42)
    sim_params = np.random.multivariate_normal(params, cov, n_mc)

    scenarios = df_proj[df_proj['year'] == DEMAND_YEAR]['scenario'].unique()
    print(f"  Demographic scenarios: {len(scenarios)}")
    print(f"  MC simulations:        {n_mc}")

    # For each scenario, compute mean beds and MC spread
    scenario_means = []
    scenario_spreads = []
    all_results = []

    for i, scen in enumerate(scenarios):
        if (i + 1) % 10 == 0:
            print(f"  Processing scenario {i + 1}/{len(scenarios)}...", end="\r")

        df_2050 = df_proj[
            (df_proj['scenario'] == scen) & (df_proj['year'] == DEMAND_YEAR)
        ].copy()
        proj_cells = df_2050.groupby(
            ['age_group', 'sex', 'Nationality', 'NUTS2'], as_index=False
        )['Population'].sum()

        X = dmatrix(formula_rhs, proj_cells, return_type="dataframe")
        pop = proj_cells['Population'].values

        # MC projection
        rates_matrix = np.exp(X.values @ sim_params.T)  # (n_cells, n_mc)
        beds_matrix = (rates_matrix * pop[:, None]).sum(axis=0) / 365  # (n_mc,)

        mean_beds = beds_matrix.mean()
        std_beds = beds_matrix.std()
        scenario_means.append(mean_beds)
        scenario_spreads.append(std_beds)

        all_results.append({
            'scenario': scen,
            'beds_mean': mean_beds,
            'beds_std': std_beds,
            'beds_2.5': np.percentile(beds_matrix, 2.5),
            'beds_97.5': np.percentile(beds_matrix, 97.5),
        })

    print()
    df_res = pd.DataFrame(all_results)

    # Variance decomposition
    grand_mean = np.mean(scenario_means)
    var_across_scenarios = np.var(scenario_means)  # demographic uncertainty
    mean_within_var = np.mean([s**2 for s in scenario_spreads])  # parameter uncertainty

    total_var = var_across_scenarios + mean_within_var
    share_demo = var_across_scenarios / total_var * 100
    share_param = mean_within_var / total_var * 100

    print(f"\n--- Variance Decomposition at {DEMAND_YEAR} ---")
    print(f"  Grand mean (daily beds):          {grand_mean:,.0f}")
    print(f"  Var(across scenarios):             {var_across_scenarios:,.0f} ({share_demo:.1f}%)")
    print(f"  Mean Var(within scenario / MC):    {mean_within_var:,.0f} ({share_param:.1f}%)")
    print(f"  Total variance:                    {total_var:,.0f}")

    print(f"\n--- Range by Source ---")
    print(f"  Demographic range: [{min(scenario_means):,.0f}, {max(scenario_means):,.0f}] "
          f"(span: {max(scenario_means) - min(scenario_means):,.0f})")
    print(f"  MC 95% CI width (median): {df_res['beds_97.5'].median() - df_res['beds_2.5'].median():,.0f}")

    # Percentage terms
    demo_span_pct = (max(scenario_means) - min(scenario_means)) / grand_mean * 100
    mc_span_pct = (df_res['beds_97.5'].median() - df_res['beds_2.5'].median()) / grand_mean * 100
    print(f"\n  Demographic span as % of mean: {demo_span_pct:.1f}%")
    print(f"  MC 95% CI as % of mean:        {mc_span_pct:.1f}%")

    return df_res


# ==========================================
# TEST 7: AGE-GRADIENT SENSITIVITY
# ==========================================

def test7_age_gradient_sensitivity(df_agg, df_proj):
    print("\n" + "=" * 80)
    print("TEST 7: AGE-GRADIENT SENSITIVITY")
    print("=" * 80)

    model = fit_poisson(df_agg)
    formula_rhs = "C(age_group) + C(sex) + C(Nationality) + C(NUTS2)"

    # Print the age coefficients
    age_params = {k: v for k, v in model.params.items() if 'age_group' in str(k)}
    print(f"\n--- Age Group Coefficients (reference = youngest group) ---")
    print(f"{'Age Group':>25s} {'Coefficient':>12s} {'Rate Ratio':>12s}")
    print(f"  {'(reference)':>23s} {'0.0000':>12s} {'1.0000':>12s}")
    for name, coef in sorted(age_params.items(), key=lambda x: x[1]):
        print(f"  {name:>23s} {coef:12.4f} {np.exp(coef):12.4f}")

    # Get one scenario for 2050
    one_scen = df_proj['scenario'].unique()[0]
    df_2050 = df_proj[
        (df_proj['scenario'] == one_scen) & (df_proj['year'] == DEMAND_YEAR)
    ].copy()
    proj_cells = df_2050.groupby(
        ['age_group', 'sex', 'Nationality', 'NUTS2'], as_index=False
    )['Population'].sum()
    X = dmatrix(formula_rhs, proj_cells, return_type="dataframe")
    pop = proj_cells['Population'].values

    # Baseline projection
    rates_base = np.exp(X.values @ model.params.values)
    beds_base = (rates_base * pop).sum() / 365

    # Identify old-age parameters (age groups 14-19, roughly 65+)
    old_age_params = [k for k in age_params if any(
        f'T.{ag}' in str(k) for ag in [14, 15, 16, 17, 18, 19]
    )]

    perturbations = [-0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20]
    print(f"\n--- Sensitivity: Perturbing Age 65+ Coefficients ---")
    print(f"  (Affects age groups 14-19: {old_age_params})")
    print(f"{'Perturbation':>15s} {'Daily Beds':>15s} {'Change vs base':>15s}")

    for p in perturbations:
        params_mod = model.params.copy()
        for op in old_age_params:
            params_mod[op] *= (1.0 + p)
        rates_mod = np.exp(X.values @ params_mod.values)
        beds_mod = (rates_mod * pop).sum() / 365
        change = ((beds_mod / beds_base) - 1) * 100
        print(f"  {p:+.0%}            {beds_mod:15,.0f} {change:+14.1f}%")

    # Decompose demand by age group
    print(f"\n--- Demand Decomposition by Age Group (2050, one scenario) ---")
    proj_cells_copy = proj_cells.copy()
    proj_cells_copy['pred_beds'] = rates_base * pop / 365
    by_age = proj_cells_copy.groupby('age_group').agg(
        Population=('Population', 'sum'),
        Daily_Beds=('pred_beds', 'sum')
    ).reset_index()
    by_age['Share'] = by_age['Daily_Beds'] / by_age['Daily_Beds'].sum() * 100
    by_age['Per_Capita'] = by_age['Daily_Beds'] / by_age['Population'] * 365  # annual bed-days

    print(f"{'AG':>4s} {'Population':>12s} {'Daily_Beds':>12s} {'Share%':>8s} {'Ann_BedDays_pc':>15s}")
    for _, row in by_age.iterrows():
        print(f"  {int(row['age_group']):>3d} {row['Population']:12,.0f} {row['Daily_Beds']:12,.0f} "
              f"{row['Share']:8.1f} {row['Per_Capita']:15.2f}")

    # By nationality
    print(f"\n--- Demand Decomposition by Nationality (2050, one scenario) ---")
    proj_cells_copy2 = proj_cells.copy()
    proj_cells_copy2['pred_beds'] = rates_base * pop / 365
    by_nat = proj_cells_copy2.groupby('Nationality').agg(
        Population=('Population', 'sum'),
        Daily_Beds=('pred_beds', 'sum')
    ).reset_index()
    by_nat['Share'] = by_nat['Daily_Beds'] / by_nat['Daily_Beds'].sum() * 100
    by_nat['Pop_Share'] = by_nat['Population'] / by_nat['Population'].sum() * 100
    by_nat['PDI'] = by_nat['Share'] / by_nat['Pop_Share']

    print(f"{'Nationality':>20s} {'Population':>12s} {'Pop%':>8s} {'Daily_Beds':>12s} {'Demand%':>8s} {'PDI':>6s}")
    for _, row in by_nat.iterrows():
        print(f"  {row['Nationality']:>18s} {row['Population']:12,.0f} {row['Pop_Share']:8.1f} "
              f"{row['Daily_Beds']:12,.0f} {row['Share']:8.1f} {row['PDI']:6.2f}")


# ==========================================
# TEST 8: GOODNESS-OF-FIT DIAGNOSTICS
# ==========================================

def test8_goodness_of_fit(df_agg):
    print("\n" + "=" * 80)
    print("TEST 8: GOODNESS-OF-FIT DIAGNOSTICS")
    print("=" * 80)

    model = fit_poisson(df_agg)
    pred_rates = predict_rates(model, df_agg)
    pred_days = pred_rates * df_agg['Population'].values
    obs_days = df_agg['num_days'].values

    # Overall fit
    ss_res = np.sum((obs_days - pred_days) ** 2)
    ss_tot = np.sum((obs_days - np.mean(obs_days)) ** 2)
    pseudo_r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(obs_days, pred_days)[0, 1]

    print(f"\n--- Overall Fit ---")
    print(f"  Pseudo R-squared:     {pseudo_r2:.6f}")
    print(f"  Correlation:          {corr:.6f}")
    print(f"  Total observed:       {obs_days.sum():,.0f}")
    print(f"  Total predicted:      {pred_days.sum():,.0f}")
    print(f"  Ratio (pred/obs):     {pred_days.sum() / obs_days.sum():.6f}")

    # Residuals by age group
    print(f"\n--- Residuals by Age Group ---")
    df_diag = df_agg.copy()
    df_diag['pred_days'] = pred_days
    df_diag['residual'] = obs_days - pred_days
    df_diag['pct_error'] = np.where(obs_days > 0,
                                     (pred_days - obs_days) / obs_days * 100, 0)

    by_age = df_diag.groupby('age_group').agg(
        obs=('num_days', 'sum'), pred=('pred_days', 'sum'),
        mean_pct_err=('pct_error', 'mean'),
        n_cohorts=('num_days', 'count')
    ).reset_index()
    by_age['bias_pct'] = (by_age['pred'] - by_age['obs']) / by_age['obs'] * 100

    print(f"{'AG':>4s} {'Observed':>14s} {'Predicted':>14s} {'Bias%':>8s} {'Mean_cell_err%':>15s}")
    for _, row in by_age.iterrows():
        print(f"  {int(row['age_group']):>3d} {row['obs']:14,.0f} {row['pred']:14,.0f} "
              f"{row['bias_pct']:8.2f} {row['mean_pct_err']:15.2f}")

    # Residuals by nationality
    print(f"\n--- Residuals by Nationality ---")
    by_nat = df_diag.groupby('Nationality').agg(
        obs=('num_days', 'sum'), pred=('pred_days', 'sum'),
        mean_pct_err=('pct_error', 'mean')
    ).reset_index()
    by_nat['bias_pct'] = (by_nat['pred'] - by_nat['obs']) / by_nat['obs'] * 100

    for _, row in by_nat.iterrows():
        print(f"  {row['Nationality']:>20s}: obs={row['obs']:,.0f}, pred={row['pred']:,.0f}, "
              f"bias={row['bias_pct']:+.2f}%, mean_cell_err={row['mean_pct_err']:+.2f}%")

    # Residuals by region
    print(f"\n--- Residuals by Region ---")
    by_reg = df_diag.groupby('NUTS2').agg(
        obs=('num_days', 'sum'), pred=('pred_days', 'sum'),
        mean_pct_err=('pct_error', 'mean')
    ).reset_index()
    by_reg['bias_pct'] = (by_reg['pred'] - by_reg['obs']) / by_reg['obs'] * 100

    for _, row in by_reg.iterrows():
        print(f"  {row['NUTS2']}: obs={row['obs']:,.0f}, pred={row['pred']:,.0f}, "
              f"bias={row['bias_pct']:+.2f}%")

    # Residuals by nationality x age (key cross-tabulation)
    print(f"\n--- Bias by Nationality x Age Group (%) ---")
    cross = df_diag.groupby(['age_group', 'Nationality']).agg(
        obs=('num_days', 'sum'), pred=('pred_days', 'sum')
    ).reset_index()
    cross['bias_pct'] = (cross['pred'] - cross['obs']) / cross['obs'].clip(lower=1) * 100

    pivot = cross.pivot(index='age_group', columns='Nationality', values='bias_pct')
    print(pivot.to_string(float_format='%.2f'))

    # Test for systematic patterns in residuals
    print(f"\n--- Correlation of Residuals with Predictors ---")
    df_diag['age_numeric'] = df_diag['age_group'].astype(float)
    df_diag['nat_numeric'] = (df_diag['Nationality'] == 'Foreign country').astype(float)
    resid = df_diag['residual'].values

    for col_name, col_data in [('age', df_diag['age_numeric']),
                                ('nationality', df_diag['nat_numeric']),
                                ('population', df_diag['Population'])]:
        r, p = scipy_stats.pearsonr(col_data.values, resid)
        print(f"  Corr(residual, {col_name:>12s}): r={r:+.4f}, p={p:.4f}")


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 80)
    print("HEALTHCARE MODEL ROBUSTNESS TESTS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df_bd = load_bed_days()
    df_agg = aggregate_cohorts(df_bd)
    print(f"  Cohorts: {len(df_agg)}")

    df_proj = load_projections()

    # Run tests
    poisson_model = test1_overdispersion(df_bd, df_agg)
    specs_results = test2_alternative_specs(df_agg)
    test3_leave_region_out(df_agg)
    test4_bootstrap(df_agg, df_proj, n_boot=N_BOOTSTRAP)
    test5_counterfactual(df_agg, df_proj)
    test6_variance_decomposition(df_agg, df_proj, n_mc=N_MC_SIMS)
    test7_age_gradient_sensitivity(df_agg, df_proj)
    test8_goodness_of_fit(df_agg)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()