"""Robustness tests of the education model."""

import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy import stats as scipy_stats

import config

warnings.filterwarnings('ignore')


FORMULA_PARSIMONIOUS = "C(Nationality) + C(Age) + C(NUTS3)"
FORMULA_BASELINE = "C(Nationality) + C(Age) + C(NUTS3) + C(Age):C(Nationality)"
FORMULA_TREND = FORMULA_BASELINE + " + Year_std"
FORMULA_TREND_NAT = FORMULA_TREND + " + C(Nationality):Year_std"

BASE_YEAR = 2023
DEMAND_YEAR = 2050
YEARS_HISTORIC = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

PATH_INPUT_RATIO = r'XLSX/df_ratio_nuts3_cob_shift_mod-type.csv'
PATH_INPUT_TEACHERS = r'XLSX/df_tpers_st_nuts3_mod-type.csv'
PATH_PROJECTION_PQT = config.PATH_PROJECTION
PATH_EXTRAPOLATION_2025 = r'Data/Statcube/population-2025_nuts3_agegroups_nat.csv'

N_BOOTSTRAP = 200
SEED = 42


def load_historic_data():
    df_ratio = pd.read_csv(PATH_INPUT_RATIO)
    df_teachers = pd.read_csv(PATH_INPUT_TEACHERS)
    df_teachers['scenario'] = 'BSL'
    return df_ratio, df_teachers


def prepare_population_projections():
    projected_all = pd.read_parquet(PATH_PROJECTION_PQT)
    projected_all.rename(columns={
        'sex': 'Sex', 'age': 'Age', 'OBS_VALUE': 'Population',
        'geo': 'NUTS3', 'TIME_PERIOD': 'Year'
    }, inplace=True)

    def convert_age(x):
        if x == 'Y_GE100': return 100
        elif x == 'Y_LT1': return 0
        return int(str(x).lstrip('Y'))

    projected_all['Age'] = projected_all['Age'].apply(convert_age)
    projected_all['Age'] = projected_all['Age'].apply(lambda x: 5 if x <= 5 else (50 if x >= 50 else x))
    projected_all['Population'] = projected_all['Population'].clip(lower=0)
    projected_all = projected_all.groupby(
        ['Year', 'NUTS3', 'Age', 'Nationality', 'scenario'], as_index=False
    )['Population'].sum()
    projected_all = projected_all[projected_all['Year'] <= DEMAND_YEAR]

    extra_df = pd.read_csv(PATH_EXTRAPOLATION_2025, skiprows=8, encoding='latin-1')
    if 'Values' in extra_df.columns:
        symbol_idx = extra_df[extra_df['Values'] == "Symbol"].index
        if not symbol_idx.empty:
            extra_df = extra_df.iloc[:symbol_idx[0]]

    extra_df['Number'] = pd.to_numeric(
        extra_df['Number'].replace('-', '0'), errors='coerce'
    ).fillna(0)
    extra_df.columns = [
        'Values', 'Year', 'NUTS3', 'Age', 'Nationality',
        'Population', 'Annotations', 'Values_drop'
    ]
    extra_df = extra_df[extra_df['NUTS3'] != 'Not classifiable <0>']
    extra_df['Year'] = pd.to_numeric(extra_df['Year'], errors='coerce')
    extra_df['NUTS3'] = extra_df['NUTS3'].str.extract(r'<(.*?)>')
    extra_df = extra_df[extra_df['Age'] != 'Not applicable']

    def map_age_bin(age_str):
        match = re.search(r"\d+", str(age_str))
        num = int(match.group()) if match else None
        if "under" in str(age_str): return 5
        elif "plus" in str(age_str): return 50
        elif "to" in str(age_str): return 50 if num >= 50 else num
        elif num is not None: return 5 if num <= 5 else (50 if num >= 50 else num)
        return np.nan

    extra_df["Age"] = extra_df["Age"].apply(map_age_bin)
    extra_df = extra_df.groupby(
        ["Year", "NUTS3", "Nationality", "Age"], as_index=False
    )["Population"].sum()
    extra_df['scenario'] = 'extrapolated'
    extra_df['Year'] = 2024

    return pd.concat([projected_all, extra_df], ignore_index=True)


def prepare_wide_data(df_historical, school_type_cols, age_stats=None, year_stats=None):
    df_wide = df_historical.pivot_table(
        index=['Year', 'NUTS3', 'Age', 'Nationality'],
        columns='School_Type', values='Students', fill_value=0
    ).reset_index()

    df_pop = df_historical[
        ['Year', 'NUTS3', 'Age', 'Nationality', 'Population']
    ].drop_duplicates()
    df_wide = pd.merge(df_wide, df_pop, on=['Year', 'NUTS3', 'Age', 'Nationality'])

    for col in school_type_cols:
        if col not in df_wide.columns:
            df_wide[col] = 0

    df_wide['Total_Students'] = df_wide[school_type_cols].sum(axis=1)
    df_wide['Population_Int'] = df_wide['Population'].round().astype(int).clip(lower=1)
    df_wide['Enrolled_Count'] = np.minimum(
        df_wide['Total_Students'].round().astype(int),
        df_wide['Population_Int']
    )
    df_wide['Total_Enrollment_Ratio'] = df_wide['Total_Students'] / df_wide['Population'].clip(lower=1)
    df_wide['Surplus_Rate'] = np.maximum(0, df_wide['Total_Enrollment_Ratio'] - 1.0)

    if age_stats is None:
        age_stats = {'mean': df_wide['Age'].mean(), 'std': df_wide['Age'].std()}
    if year_stats is None:
        year_stats = {'mean': df_wide['Year'].mean(), 'std': df_wide['Year'].std()}

    df_wide['Age_std'] = (df_wide['Age'] - age_stats['mean']) / age_stats['std']
    df_wide['Year_std'] = (df_wide['Year'] - year_stats['mean']) / year_stats['std']
    df_wide['const'] = 1.0

    return df_wide, age_stats, year_stats


def fit_binomial_glm(df_agg, formula):
    y_successes = df_agg['Enrolled_Count'].values
    y_trials = df_agg['Population_Int'].values
    y_failures = np.maximum(0, y_trials - y_successes)
    y_binomial = np.column_stack([y_successes, y_failures])
    try:
        X_dm = dmatrix(formula, df_agg, return_type='dataframe')
        model = sm.GLM(y_binomial, X_dm, family=sm.families.Binomial())
        result = model.fit()
        return result, result.aic, X_dm.design_info
    except Exception as e:
        return None, np.inf, None


def predict_enrollment(model_result, design_info, df_proj, age_stats, year_stats,
                       scenario_type='statusquo'):
    df = df_proj.copy()
    df['Age_std'] = (df['Age'] - age_stats['mean']) / age_stats['std']
    if scenario_type == 'statusquo':
        df['Year_std'] = 0.0
    else:
        df['Year_std'] = (df['Year'] - year_stats['mean']) / year_stats['std']
    df['const'] = 1.0
    try:
        X = dmatrix(design_info, df, return_type='dataframe')
        return model_result.predict(X)
    except:
        return None


def compute_projected_students(df_proj, base_rate, surplus_lookup, emp_dist, school_cols):
    df = df_proj.copy()
    df = df.merge(surplus_lookup, on=['Age', 'NUTS3', 'Nationality'], how='left')
    df['Surplus_Rate'] = df['Surplus_Rate'].fillna(0)
    total_ratio = base_rate.values + df['Surplus_Rate'].values
    total_students = df['Population'].values * total_ratio

    df_merged = df.merge(emp_dist, on=['NUTS3', 'Age', 'Nationality'], how='left',
                         suffixes=('', '_dist'))
    result = {}
    for col in school_cols:
        dist_col = col if col in df_merged.columns else col + '_dist'
        if dist_col in df_merged.columns:
            result[col] = total_students * df_merged[dist_col].fillna(0).values
        else:
            result[col] = np.zeros(len(df))
    return result


def build_surplus_lookup(df_wide):
    return df_wide.groupby(['Age', 'NUTS3', 'Nationality'])['Surplus_Rate'].mean().reset_index()


def build_empirical_distributions(df_wide, school_cols):
    cohort_cols = ['NUTS3', 'Age', 'Nationality']
    df_sums = df_wide.groupby(cohort_cols)[school_cols + ['Total_Students']].sum().reset_index()
    total = df_sums['Total_Students'].clip(lower=1)
    for s in school_cols:
        df_sums[s] = df_sums[s] / total
    return df_sums[cohort_cols + school_cols]


def full_pipeline(df_train, df_test, school_cols, formula_key='sq'):
    formulas = {
    'sq': FORMULA_BASELINE,
    'trend': FORMULA_TREND
    }

    df_wide_train, a_stats, y_stats = prepare_wide_data(df_train, school_cols)
    surplus = build_surplus_lookup(df_wide_train)
    emp_dist = build_empirical_distributions(df_wide_train, school_cols)

    result, aic, design_info = fit_binomial_glm(df_wide_train, formulas[formula_key])
    if result is None:
        return None

    test_cells = df_test.groupby(
        ['Year', 'NUTS3', 'Age', 'Nationality'], as_index=False
    ).agg(
        Observed_Students=('Students', 'sum'),
        Population=('Population', 'first')
    )

    base_rate = predict_enrollment(
        result, design_info, test_cells, a_stats, y_stats,
        scenario_type='statusquo' if formula_key == 'sq' else 'trend'
    )
    if base_rate is None:
        return None

    test_cells['Predicted_Base_Rate'] = base_rate.values
    test_cells = test_cells.merge(surplus, on=['Age', 'NUTS3', 'Nationality'], how='left')
    test_cells['Surplus_Rate'] = test_cells['Surplus_Rate'].fillna(0)
    test_cells['Predicted_Students'] = (
        test_cells['Population'] * (test_cells['Predicted_Base_Rate'] + test_cells['Surplus_Rate'])
    )
    return test_cells


def test1_loyo_cv(df_ratio, school_cols):
    print("\n" + "=" * 80)
    print("TEST 1: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
    print("=" * 80)

    available_years = sorted(df_ratio['Year'].unique())
    print(f"Available years: {available_years}")

    results = []
    for hold_out in available_years:
        df_train = df_ratio[df_ratio['Year'] != hold_out].copy()
        df_test = df_ratio[df_ratio['Year'] == hold_out].copy()

        test_cells = full_pipeline(df_train, df_test, school_cols, formula_key='sq')
        if test_cells is None:
            print(f"  Year {hold_out}: FAILED to fit")
            continue

        obs = test_cells['Observed_Students'].values
        pred = test_cells['Predicted_Students'].values
        mask = obs > 0

        mape = np.mean(np.abs((obs[mask] - pred[mask]) / obs[mask])) * 100
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        mae = np.mean(np.abs(obs - pred))
        corr = np.corrcoef(obs, pred)[0, 1]

        results.append({
            'Hold_Out_Year': hold_out, 'MAPE': mape, 'RMSE': rmse,
            'MAE': mae, 'Correlation': corr, 'N_cells': len(test_cells)
        })

    df_res = pd.DataFrame(results)
    print("\n--- Overall LOYO-CV Results ---")
    print(df_res.to_string(index=False, float_format='%.3f'))
    print(f"\nMean MAPE across folds: {df_res['MAPE'].mean():.2f}%")
    print(f"Std  MAPE across folds: {df_res['MAPE'].std():.2f}%")
    print(f"Mean Correlation:        {df_res['Correlation'].mean():.4f}")

    print("\n--- LOYO-CV Breakdown by Nationality (hold-out = base year) ---")
    df_train_all = df_ratio[df_ratio['Year'] != BASE_YEAR].copy()
    df_test_base = df_ratio[df_ratio['Year'] == BASE_YEAR].copy()
    test_cells = full_pipeline(df_train_all, df_test_base, school_cols, 'sq')
    if test_cells is not None:
        for nat in test_cells['Nationality'].unique():
            sub = test_cells[test_cells['Nationality'] == nat]
            obs = sub['Observed_Students'].values
            pred = sub['Predicted_Students'].values
            mask = obs > 0
            mape = np.mean(np.abs((obs[mask] - pred[mask]) / obs[mask])) * 100
            print(f"  {nat:>15s}: MAPE = {mape:.2f}%, N = {len(sub)}")

    print("\n--- LOYO-CV Breakdown by NUTS3 (hold-out = base year) ---")
    if test_cells is not None:
        for nuts in sorted(test_cells['NUTS3'].unique()):
            sub = test_cells[test_cells['NUTS3'] == nuts]
            obs = sub['Observed_Students'].values
            pred = sub['Predicted_Students'].values
            mask = obs > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((obs[mask] - pred[mask]) / obs[mask])) * 100
                print(f"  {nuts}: MAPE = {mape:.2f}%, N = {len(sub)}")

    return df_res


def test2_str_sensitivity(df_ratio, df_teachers, df_projection, school_cols):
    print("\n" + "=" * 80)
    print("TEST 2: STUDENT-TO-TEACHER RATIO SENSITIVITY")
    print("=" * 80)

    base_tp = df_teachers[df_teachers['Year'] == BASE_YEAR].copy()
    base_str = base_tp.groupby(['NUTS3', 'School_Type']).agg(
        Students_total=('Student-to-teacher', 'first')
    ).reset_index()
    base_str.rename(columns={'Students_total': 'Base_STR'}, inplace=True)

    str_by_year = df_teachers.groupby(['Year', 'NUTS3', 'School_Type'])['Student-to-teacher'].first().reset_index()
    str_stats = str_by_year.groupby(['NUTS3', 'School_Type'])['Student-to-teacher'].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()

    print("\n--- Historical Student-to-Teacher Ratio Statistics ---")
    overall_str = str_by_year.groupby('Year')['Student-to-teacher'].mean()
    print(overall_str.to_string())
    print(f"\nOverall mean STR: {str_by_year['Student-to-teacher'].mean():.2f}")
    print(f"Overall std  STR: {str_by_year['Student-to-teacher'].std():.2f}")

    df_wide_full, a_stats, y_stats = prepare_wide_data(df_ratio, school_cols)
    result_sq, _, design_sq = fit_binomial_glm(
        df_wide_full,
        FORMULA_BASELINE
    )
    surplus = build_surplus_lookup(df_wide_full)
    emp_dist = build_empirical_distributions(df_wide_full, school_cols)

    proj_scenarios = df_projection['scenario'].unique()
    one_scen = proj_scenarios[0]
    df_proj_2050 = df_projection[
        (df_projection['scenario'] == one_scen) & (df_projection['Year'] == DEMAND_YEAR)
    ].copy()
    proj_cells = df_proj_2050.groupby(
        ['Year', 'NUTS3', 'Age', 'Nationality'], as_index=False
    )['Population'].first()

    base_rate = predict_enrollment(result_sq, design_sq, proj_cells, a_stats, y_stats, 'statusquo')
    proj_cells['Base_Rate'] = base_rate.values
    proj_cells = proj_cells.merge(surplus, on=['Age', 'NUTS3', 'Nationality'], how='left')
    proj_cells['Surplus_Rate'] = proj_cells['Surplus_Rate'].fillna(0)
    proj_cells['Total_Students'] = proj_cells['Population'] * (
        proj_cells['Base_Rate'] + proj_cells['Surplus_Rate']
    )

    total_students_2050 = proj_cells['Total_Students'].sum()

    perturbations = [-0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20]
    print("\n--- Teacher Demand Sensitivity to STR Perturbation (2050) ---")
    print(f"{'Perturbation':>15s} {'STR_mult':>10s} {'Teachers':>15s} {'Change_vs_base':>15s}")

    avg_str = base_tp['Students'].sum() / base_tp['Teachers'].sum()
    baseline_teachers = total_students_2050 / avg_str
    for p in perturbations:
        multiplier = 1.0 + p
        avg_str = base_tp['Students'].sum() / base_tp['Teachers'].sum()
        teachers = total_students_2050 / (avg_str * multiplier)
        change_pct = ((teachers / baseline_teachers) - 1) * 100 if baseline_teachers else 0
        print(f"  {p:+.0%}            {avg_str * multiplier:10.2f} {teachers:15,.0f} {change_pct:+14.1f}%")

    print("\n--- Base-year STR by School Type ---")
    str_by_type = str_by_year[str_by_year['Year'] == BASE_YEAR].groupby('School_Type')[
        'Student-to-teacher'
    ].agg(['mean', 'std', 'min', 'max'])
    print(str_by_type.to_string(float_format='%.2f'))


def test3_bootstrap(df_ratio, school_cols, n_boot=N_BOOTSTRAP):
    print("\n" + "=" * 80)
    print(f"TEST 3: BOOTSTRAP CONFIDENCE INTERVALS (n={n_boot})")
    print("=" * 80)

    df_wide_full, a_stats, y_stats = prepare_wide_data(df_ratio, school_cols)
    formula = FORMULA_BASELINE

    ref_result, ref_aic, _ = fit_binomial_glm(df_wide_full, formula)
    if ref_result is None:
        print("Reference model failed to fit.")
        return
    ref_params = ref_result.params

    rng = np.random.default_rng(SEED)

    region_years = df_wide_full[['NUTS3', 'Year']].drop_duplicates()
    n_ry = len(region_years)

    boot_params = []
    boot_aic = []
    boot_total_students = []

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            print(f"  Bootstrap iteration {b + 1}/{n_boot}...")

        idx = rng.choice(n_ry, size=n_ry, replace=True)
        sampled_ry = region_years.iloc[idx]
        sampled_ry = sampled_ry.reset_index(drop=True)
        sampled_ry['_merge_key'] = range(len(sampled_ry))

        pieces = []
        for _, row in sampled_ry.iterrows():
            mask = (df_wide_full['NUTS3'] == row['NUTS3']) & (df_wide_full['Year'] == row['Year'])
            pieces.append(df_wide_full[mask].copy())

        if not pieces:
            continue
        df_boot = pd.concat(pieces, ignore_index=True)

        df_boot['Age_std'] = (df_boot['Age'] - a_stats['mean']) / a_stats['std']
        df_boot['Year_std'] = (df_boot['Year'] - y_stats['mean']) / y_stats['std']

        res_b, aic_b, _ = fit_binomial_glm(df_boot, formula)
        if res_b is not None:
            boot_params.append(res_b.params.reindex(ref_params.index).values)
            boot_aic.append(aic_b)
            try:
                X_orig = dmatrix(formula, df_wide_full, return_type='dataframe')
                pred_b = res_b.predict(X_orig)
                total_stud = (pred_b * df_wide_full['Population_Int'].values).sum()
                boot_total_students.append(total_stud)
            except:
                pass

    boot_params = np.array(boot_params, dtype=float)
    n_successful = len(boot_params)
    print(f"\nSuccessful bootstrap iterations: {n_successful}/{n_boot}")

    if n_successful < 10:
        print("Too few successful iterations for reliable CIs.")
        return

    param_names = ref_result.params.index.tolist()
    n_incomplete = int(np.isnan(boot_params).any(axis=1).sum())
    print(f"Draws with a dropped category (parameters set to NaN): {n_incomplete}")
    print(f"\n--- Bootstrap 95% CIs for GLM Coefficients (n={n_successful}) ---")
    print(f"{'Parameter':>40s} {'Estimate':>10s} {'Boot_2.5%':>10s} {'Boot_97.5%':>11s} {'Boot_SE':>10s}")
    for i, name in enumerate(param_names):
        est = ref_params.iloc[i]
        lo = np.nanpercentile(boot_params[:, i], 2.5)
        hi = np.nanpercentile(boot_params[:, i], 97.5)
        se = np.nanstd(boot_params[:, i])
        print(f"  {name:>38s} {est:10.4f} {lo:10.4f} {hi:11.4f} {se:10.4f}")

    if boot_total_students:
        arr = np.array(boot_total_students)
        print(f"\n--- Bootstrap Uncertainty in Total Predicted Students (base-year data) ---")
        print(f"  Median:   {np.median(arr):,.0f}")
        print(f"  2.5%:     {np.percentile(arr, 2.5):,.0f}")
        print(f"  97.5%:    {np.percentile(arr, 97.5):,.0f}")
        print(f"  CV:       {np.std(arr) / np.mean(arr) * 100:.2f}%")

    print(f"\n--- Bootstrap AIC Distribution ---")
    print(f"  Reference AIC: {ref_aic:,.1f}")
    print(f"  Boot median:   {np.median(boot_aic):,.1f}")
    print(f"  Boot range:    [{np.min(boot_aic):,.1f}, {np.max(boot_aic):,.1f}]")


def test4_school_type_stability(df_ratio, school_cols):
    print("\n" + "=" * 80)
    print("TEST 4: SCHOOL-TYPE DISTRIBUTION STABILITY OVER TIME")
    print("=" * 80)

    yearly_shares = (
        df_ratio.groupby(['Year', 'School_Type'])['Students'].sum().reset_index()
    )
    total_by_year = yearly_shares.groupby('Year')['Students'].transform('sum')
    yearly_shares['Share'] = yearly_shares['Students'] / total_by_year

    pivot = yearly_shares.pivot(index='Year', columns='School_Type', values='Share')
    print("\n--- National School-Type Shares by Year ---")
    print(pivot.to_string(float_format='%.4f'))

    print("\n--- Temporal Variability of Shares ---")
    print(f"{'School_Type':>30s} {'Mean':>8s} {'Std':>8s} {'CV(%)':>8s} {'Min':>8s} {'Max':>8s} {'Trend_slope':>12s}")
    for col in pivot.columns:
        series = pivot[col].dropna()
        mean_val = series.mean()
        std_val = series.std()
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        x = np.arange(len(series))
        if len(x) > 2:
            slope, intercept, r, p, se = scipy_stats.linregress(x, series.values)
        else:
            slope, p = 0, 1
        print(f"  {col:>28s} {mean_val:8.4f} {std_val:8.4f} {cv:8.2f} {series.min():8.4f} "
              f"{series.max():8.4f} {slope:+11.5f} {'*' if p < 0.05 else ''}")

    print("\n--- School-Type Shares by Nationality (pooled years) ---")
    nat_shares = df_ratio.groupby(['Nationality', 'School_Type'])['Students'].sum().reset_index()
    total_by_nat = nat_shares.groupby('Nationality')['Students'].transform('sum')
    nat_shares['Share'] = nat_shares['Students'] / total_by_nat
    pivot_nat = nat_shares.pivot(index='School_Type', columns='Nationality', values='Share')
    print(pivot_nat.to_string(float_format='%.4f'))

    print("\n--- Impact: Recent-Year (2023) vs Pooled Shares ---")
    recent_shares = df_ratio[df_ratio['Year'] == BASE_YEAR].groupby('School_Type')['Students'].sum()
    recent_shares = recent_shares / recent_shares.sum()
    pooled_shares = df_ratio.groupby('School_Type')['Students'].sum()
    pooled_shares = pooled_shares / pooled_shares.sum()

    comparison = pd.DataFrame({
        'Pooled': pooled_shares,
        'Recent_2023': recent_shares,
        'Diff_pp': (recent_shares - pooled_shares) * 100
    })
    print(comparison.to_string(float_format='%.4f'))


def test5_alternative_specs(df_ratio, school_cols):
    print("\n" + "=" * 80)
    print("TEST 5: ALTERNATIVE MODEL SPECIFICATIONS")
    print("=" * 80)

    df_wide, a_stats, y_stats = prepare_wide_data(df_ratio, school_cols)

    specs = {
        'Parsimonious (additive)':        FORMULA_PARSIMONIOUS,
        'Baseline (Age x Nat)':           FORMULA_BASELINE,
        'Nat x Region interaction':       "C(Nationality) * C(NUTS3) + C(Age)",
        'Full Nat interactions':          "C(Nationality) * C(Age) + C(Nationality) * C(NUTS3)",
        'Trend (Baseline + Year)':        FORMULA_TREND,
        'Trend + Nat x Region':           FORMULA_BASELINE + " + C(Nationality):C(NUTS3) + Year_std",
    }

    print("\n--- Model Comparison (AIC, BIC, Deviance) ---")
    print(f"{'Specification':>35s} {'AIC':>12s} {'BIC':>12s} {'Deviance':>12s} {'df_resid':>10s} {'n_params':>10s}")

    for name, formula in specs.items():
        res, aic, _ = fit_binomial_glm(df_wide, formula)
        if res is not None:
            print(f"  {name:>33s} {res.aic:12.1f} {res.bic_deviance:12.1f} "
                  f"{res.deviance:12.1f} {res.df_resid:10.0f} {len(res.params):10d}")
        else:
            print(f"  {name:>33s} {'FAILED':>12s}")

    print("\n--- Trend Coefficient Stability (Restricted Training Windows) ---")
    windows = {
        'Full (2015-2023)': None,
        'Recent (2019-2023)': 2019,
        'Recent (2020-2023)': 2020,
        'Pre-COVID (2015-2019)': ('2015', '2019'),
    }

    trend_formula = FORMULA_TREND
    print(f"{'Window':>25s} {'Year_coeff':>12s} {'SE':>10s} {'p-value':>10s} {'AIC':>12s}")

    for wname, constraint in windows.items():
        if constraint is None:
            df_sub = df_ratio.copy()
        elif isinstance(constraint, tuple):
            df_sub = df_ratio[
                (df_ratio['Year'] >= int(constraint[0])) & (df_ratio['Year'] <= int(constraint[1]))
            ].copy()
        else:
            df_sub = df_ratio[df_ratio['Year'] >= constraint].copy()

        df_w, _, _ = prepare_wide_data(df_sub, school_cols)
        res, aic, _ = fit_binomial_glm(df_w, trend_formula)
        if res is not None:
            year_idx = [i for i, n in enumerate(res.params.index) if 'Year_std' in str(n)]
            if year_idx:
                idx = year_idx[0]
                coef = res.params.iloc[idx]
                se = res.bse.iloc[idx]
                pval = res.pvalues.iloc[idx]
                print(f"  {wname:>23s} {coef:12.5f} {se:10.5f} {pval:10.4f} {aic:12.1f}")
            else:
                print(f"  {wname:>23s} {'Year_std not found':>12s}")
        else:
            print(f"  {wname:>23s} {'FAILED':>12s}")


def test6_leave_region_out(df_ratio, school_cols):
    print("\n" + "=" * 80)
    print("TEST 6: LEAVE-ONE-REGION-OUT CROSS-VALIDATION")
    print("=" * 80)

    regions = sorted(df_ratio['NUTS3'].unique())
    print(f"Regions: {regions}")

    results = []
    for region in regions:
        df_train = df_ratio[df_ratio['NUTS3'] != region].copy()
        df_test = df_ratio[df_ratio['NUTS3'] == region].copy()

        df_wide_train, a_stats, y_stats = prepare_wide_data(df_train, school_cols)
        formula_no_region = "C(Nationality) + C(Age) + C(Age):C(Nationality)"
        res, aic, design = fit_binomial_glm(df_wide_train, formula_no_region)

        if res is None:
            print(f"  {region}: FAILED")
            continue

        test_cells = df_test.groupby(
            ['Year', 'NUTS3', 'Age', 'Nationality'], as_index=False
        ).agg(Observed_Students=('Students', 'sum'), Population=('Population', 'first'))

        base_rate = predict_enrollment(res, design, test_cells, a_stats, y_stats, 'statusquo')
        if base_rate is None:
            continue

        surplus_train = build_surplus_lookup(df_wide_train)
        test_cells = test_cells.merge(surplus_train, on=['Age', 'NUTS3', 'Nationality'], how='left')
        global_surplus = df_wide_train.groupby(['Age', 'Nationality'])['Surplus_Rate'].mean().reset_index()
        global_surplus.rename(columns={'Surplus_Rate': 'Global_Surplus'}, inplace=True)
        test_cells = test_cells.merge(global_surplus, on=['Age', 'Nationality'], how='left')
        test_cells['Surplus_Rate'] = test_cells['Surplus_Rate'].fillna(test_cells['Global_Surplus']).fillna(0)

        test_cells['Predicted_Students'] = test_cells['Population'] * (
            base_rate.values + test_cells['Surplus_Rate'].values
        )

        obs = test_cells['Observed_Students'].values
        pred = test_cells['Predicted_Students'].values
        mask = obs > 0

        if mask.sum() == 0:
            continue

        mape = np.mean(np.abs((obs[mask] - pred[mask]) / obs[mask])) * 100
        bias = np.mean(pred - obs)
        rmse = np.sqrt(np.mean((obs - pred) ** 2))

        results.append({
            'Region': region, 'MAPE': mape, 'RMSE': rmse,
            'Bias': bias, 'N_cells': len(test_cells)
        })

    df_res = pd.DataFrame(results)
    print("\n--- Leave-One-Region-Out Results ---")
    print(df_res.to_string(index=False, float_format='%.2f'))
    print(f"\nMean MAPE:  {df_res['MAPE'].mean():.2f}%")
    print(f"Worst MAPE: {df_res.loc[df_res['MAPE'].idxmax(), 'Region']} "
          f"({df_res['MAPE'].max():.2f}%)")
    print(f"Best MAPE:  {df_res.loc[df_res['MAPE'].idxmin(), 'Region']} "
          f"({df_res['MAPE'].min():.2f}%)")
    return df_res


def test7_surplus_sensitivity(df_ratio, school_cols):
    print("\n" + "=" * 80)
    print("TEST 7: SURPLUS COMPONENT SENSITIVITY")
    print("=" * 80)

    df_wide, a_stats, y_stats = prepare_wide_data(df_ratio, school_cols)

    surplus = df_wide['Surplus_Rate']
    n_nonzero = (surplus > 0).sum()
    total_cells = len(surplus)
    print(f"\n--- Surplus Rate Distribution ---")
    print(f"  Total cells:         {total_cells}")
    print(f"  Cells with surplus:  {n_nonzero} ({n_nonzero / total_cells * 100:.1f}%)")
    print(f"  Mean (all):          {surplus.mean():.4f}")
    print(f"  Mean (nonzero):      {surplus[surplus > 0].mean():.4f}")
    print(f"  Median (nonzero):    {surplus[surplus > 0].median():.4f}")
    print(f"  Max:                 {surplus.max():.4f}")
    print(f"  P95 (nonzero):       {surplus[surplus > 0].quantile(0.95):.4f}")

    print("\n--- Mean Surplus Rate by Region ---")
    by_region = df_wide.groupby('NUTS3')['Surplus_Rate'].agg(['mean', 'sum', 'count']).reset_index()
    by_region['nonzero_frac'] = df_wide.groupby('NUTS3')['Surplus_Rate'].apply(
        lambda x: (x > 0).mean()
    ).values
    print(by_region.to_string(index=False, float_format='%.4f'))

    formula = FORMULA_BASELINE
    res, _, design = fit_binomial_glm(df_wide, formula)
    if res is None:
        print("Model fitting failed.")
        return

    X = dmatrix(design, df_wide, return_type='dataframe')
    base_rate = res.predict(X)
    pop = df_wide['Population_Int'].values

    scenarios = {
        'No surplus (floor=0)': np.zeros(len(df_wide)),
        'Observed average surplus': build_surplus_lookup(df_wide).merge(
            df_wide[['Age', 'NUTS3', 'Nationality']],
            on=['Age', 'NUTS3', 'Nationality'], how='right'
        )['Surplus_Rate'].fillna(0).values,
        'Max observed surplus per cell': df_wide.groupby(
            ['Age', 'NUTS3', 'Nationality']
        )['Surplus_Rate'].transform('max').values,
    }

    print(f"\n--- Impact on Total Students (base-year data) ---")
    print(f"{'Scenario':>35s} {'Total_Students':>18s} {'Change_vs_mean':>18s}")
    ref_students = None
    for name, surplus_vals in scenarios.items():
        total = (pop * (base_rate.values + surplus_vals)).sum()
        if name == 'Observed average surplus':
            ref_students = total
        change = ((total / ref_students) - 1) * 100 if ref_students else 0
        print(f"  {name:>33s} {total:18,.0f} {change:+17.2f}%")


def test8_variance_decomposition(df_ratio, df_projection, school_cols):
    print("\n" + "=" * 80)
    print("TEST 8: VARIANCE DECOMPOSITION (Demographic Scenario vs Behavioral Model)")
    print("=" * 80)

    df_wide, a_stats, y_stats = prepare_wide_data(df_ratio, school_cols)
    surplus = build_surplus_lookup(df_wide)
    emp_dist = build_empirical_distributions(df_wide, school_cols)

    formula_sq = FORMULA_BASELINE
    formula_tr = FORMULA_TREND

    res_sq, _, des_sq = fit_binomial_glm(df_wide, formula_sq)
    res_tr, _, des_tr = fit_binomial_glm(df_wide, formula_tr)

    if res_sq is None or res_tr is None:
        print("Model fitting failed.")
        return

    proj_scenarios = df_projection[df_projection['Year'] == DEMAND_YEAR]['scenario'].unique()
    print(f"Demographic scenarios: {len(proj_scenarios)}")

    results = []
    for scen in proj_scenarios:
        df_sub = df_projection[
            (df_projection['scenario'] == scen) & (df_projection['Year'] == DEMAND_YEAR)
        ].copy()
        proj_cells = df_sub.groupby(
            ['Year', 'NUTS3', 'Age', 'Nationality'], as_index=False
        )['Population'].first()

        for beh_label, (model_res, model_des, scen_type) in {
            'statusquo': (res_sq, des_sq, 'statusquo'),
            'trend': (res_tr, des_tr, 'trend')
        }.items():
            rate = predict_enrollment(model_res, model_des, proj_cells, a_stats, y_stats, scen_type)
            if rate is None:
                continue
            proj_cells_m = proj_cells.merge(surplus, on=['Age', 'NUTS3', 'Nationality'], how='left')
            proj_cells_m['Surplus_Rate'] = proj_cells_m['Surplus_Rate'].fillna(0)
            total_students = (proj_cells['Population'].values * (rate.values + proj_cells_m['Surplus_Rate'].values)).sum()
            results.append({
                'Demographic_Scenario': scen,
                'Behavioral_Model': beh_label,
                'Total_Students_2050': total_students
            })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No results generated.")
        return

    grand_mean = df_res['Total_Students_2050'].mean()
    scen_means = df_res.groupby('Demographic_Scenario')['Total_Students_2050'].mean()
    beh_means = df_res.groupby('Behavioral_Model')['Total_Students_2050'].mean()

    value = df_res['Total_Students_2050']
    ss_total = ((value - grand_mean) ** 2).sum()
    ss_demographic = ((df_res.groupby('Demographic_Scenario')['Total_Students_2050'].transform('mean')
                       - grand_mean) ** 2).sum()
    ss_behavioral = ((df_res.groupby('Behavioral_Model')['Total_Students_2050'].transform('mean')
                      - grand_mean) ** 2).sum()
    ss_interaction = ss_total - ss_demographic - ss_behavioral

    print(f"\n--- Variance Decomposition of Total Students in {DEMAND_YEAR} (two-way ANOVA, sums of squares) ---")
    print(f"  Grand mean:                   {grand_mean:,.0f}")
    print(f"  Total sum of squares:         {ss_total:,.0f}")
    print(f"  Demographic scenario:         {ss_demographic:,.0f}  ({ss_demographic / ss_total * 100:.1f}%)")
    print(f"  Behavioral model:             {ss_behavioral:,.0f}  ({ss_behavioral / ss_total * 100:.1f}%)")
    print(f"  Interaction (residual):       {ss_interaction:,.0f}  ({ss_interaction / ss_total * 100:.2f}%)")

    print(f"\n--- Range by Source ---")
    print(f"  Demographic scenario range:  [{scen_means.min():,.0f}, {scen_means.max():,.0f}] "
          f"(span: {scen_means.max() - scen_means.min():,.0f})")
    print(f"  Behavioral model range:      [{beh_means.min():,.0f}, {beh_means.max():,.0f}] "
          f"(span: {beh_means.max() - beh_means.min():,.0f})")

    print(f"\n--- Summary by Behavioral Model ---")
    summary = df_res.groupby('Behavioral_Model')['Total_Students_2050'].agg(
        ['mean', 'std', 'min', 'max']
    )
    print(summary.to_string(float_format='%.0f'))

    print(f"\n--- Top/Bottom 5 Demographic Scenarios (by mean total students) ---")
    scen_summary = df_res.groupby('Demographic_Scenario')['Total_Students_2050'].mean().sort_values()
    print("  Bottom 5:")
    for s, v in scen_summary.head(5).items():
        print(f"    {s}: {v:,.0f}")
    print("  Top 5:")
    for s, v in scen_summary.tail(5).items():
        print(f"    {s}: {v:,.0f}")


def _year_slope_contrast(design_info, df_wide, nationality):
    probe = df_wide.iloc[[0, 0]].copy()
    probe['Nationality'] = nationality
    probe['Year_std'] = [1.0, 0.0]
    X = dmatrix(design_info, probe, return_type='dataframe')
    return X.iloc[0].values - X.iloc[1].values


def _project_teachers(model_res, model_des, cells, scen_type, a_stats, y_stats,
                      surplus, emp_dist, school_cols, str_lookup):
    rate = predict_enrollment(model_res, model_des, cells, a_stats, y_stats, scen_type)
    if rate is None:
        return None
    students = compute_projected_students(cells, rate, surplus, emp_dist, school_cols)
    df_long = pd.concat(
        [cells.assign(School_Type=col, Students=students[col]) for col in school_cols],
        ignore_index=True
    )
    df_long = df_long.merge(str_lookup, on=['NUTS3', 'School_Type'], how='left')
    ratio = df_long['Student-to-teacher'].values
    valid = (ratio > 0) & ~np.isnan(ratio)
    teachers = np.zeros(len(df_long))
    teachers[valid] = df_long['Students'].values[valid] / ratio[valid]
    df_long['Teachers'] = teachers
    return df_long


def test9_nationality_year_slope(df_ratio, df_teachers, df_projection, school_cols):
    print("\n" + "=" * 80)
    print("TEST 9: NATIONALITY-SPECIFIC YEAR SLOPE (TREND MODEL SENSITIVITY)")
    print("=" * 80)

    growth_start = 2025

    df_wide, a_stats, y_stats = prepare_wide_data(df_ratio, school_cols)
    surplus = build_surplus_lookup(df_wide)
    emp_dist = build_empirical_distributions(df_wide, school_cols)
    nats = sorted(df_wide['Nationality'].unique())

    specs = {
        'Trend (production, common slope)': FORMULA_TREND,
        'Trend + Nat x Year':               FORMULA_TREND_NAT,
    }
    fits = {}
    print("\n--- Model Comparison (dispersion = Pearson chi2 / df_resid) ---")
    print(f"{'Specification':>35s} {'AIC':>12s} {'BIC':>12s} {'Deviance':>12s} {'df_resid':>10s} "
          f"{'n_params':>10s} {'Dispersion':>11s}")
    for name, formula in specs.items():
        res, aic, design = fit_binomial_glm(df_wide, formula)
        if res is None:
            print(f"  {name:>33s} {'FAILED':>12s}")
            return
        disp = res.pearson_chi2 / res.df_resid
        fits[name] = (res, design, disp)
        print(f"  {name:>33s} {res.aic:12.1f} {res.bic_deviance:12.1f} {res.deviance:12.1f} "
              f"{res.df_resid:10.0f} {len(res.params):10d} {disp:11.2f}")
    aic_common, aic_nat = (fits[name][0].aic for name in specs)
    print(f"  Delta AIC (Nat x Year minus common slope): {aic_nat - aic_common:,.1f}")

    print(f"\n--- Year Slope by Nationality (logit; Year_std SD = {y_stats['std']:.4f} years; "
          f"SEs per year, SE_disp = SE x sqrt(dispersion); Shift = slope x {DEMAND_YEAR - BASE_YEAR} years) ---")
    print(f"{'Specification':>35s} {'Nationality':>26s} {'Coef_per_SD':>12s} {'Slope_per_yr':>13s} "
          f"{'SE_binom':>10s} {'SE_disp':>10s} {'z_disp':>8s} {'p_disp':>8s} "
          f"{f'Shift_{BASE_YEAR}_{DEMAND_YEAR}':>16s}")
    for name, (res, design, disp) in fits.items():
        contrasts = {nat: _year_slope_contrast(design, df_wide, nat) for nat in nats}
        diff = contrasts[nats[1]] - contrasts[nats[0]]
        if np.any(diff != 0):
            contrasts[f'{nats[1]} - {nats[0]}'] = diff
        cov = res.cov_params().values
        for label, L in contrasts.items():
            coef = L @ res.params.values
            se = np.sqrt(L @ cov @ L)
            se_disp = se * np.sqrt(disp)
            z = coef / se_disp
            p = 2 * scipy_stats.norm.sf(abs(z))
            slope = coef / y_stats['std']
            print(f"  {name:>33s} {label:>26s} {coef:12.5f} {slope:13.5f} "
                  f"{se / y_stats['std']:10.5f} {se_disp / y_stats['std']:10.5f} {z:8.2f} {p:8.4f} "
                  f"{slope * (DEMAND_YEAR - BASE_YEAR):16.4f}")

    print("\n--- Observed Crude Enrolment Ratio, Ages 6-18 (students / population) ---")
    obs = df_wide[df_wide['Age'].between(6, 18)].groupby(['Year', 'Nationality'])[
        ['Total_Students', 'Population']].sum()
    print((obs['Total_Students'] / obs['Population']).unstack('Nationality').to_string(float_format='%.4f'))

    res_sq, _, des_sq = fit_binomial_glm(df_wide, FORMULA_BASELINE)
    if res_sq is None:
        print("Model fitting failed.")
        return
    res_tr, des_tr, _ = fits['Trend (production, common slope)']
    res_nat, des_nat, _ = fits['Trend + Nat x Year']
    proj_models = {
        'Status quo (production)':       (res_sq, des_sq, 'statusquo'),
        'Trend (production)':            (res_tr, des_tr, 'trend'),
        'Trend + Nat x Year (uncapped)': (res_nat, des_nat, 'trend'),
    }
    labels = list(proj_models)

    str_lookup = df_teachers[df_teachers['Year'] == BASE_YEAR][['NUTS3', 'School_Type', 'Student-to-teacher']]
    proj_scenarios = sorted(df_projection.loc[df_projection['Year'] == DEMAND_YEAR, 'scenario'].unique())
    df_two = df_projection[df_projection['Year'].isin([growth_start, DEMAND_YEAR])]
    print(f"\nDemographic scenarios: {len(proj_scenarios)} (teachers = students / {BASE_YEAR} "
          f"NUTS3 x school-type student-to-teacher ratio)")

    cell_cols = ['Year', 'NUTS3', 'Age', 'Nationality', 'Population']
    rows_t, rows_er = [], []
    for scen in proj_scenarios:
        cells = df_two.loc[df_two['scenario'] == scen, cell_cols].drop_duplicates().reset_index(drop=True)
        is_end_6_18 = (cells['Year'] == DEMAND_YEAR) & cells['Age'].between(6, 18)
        pop_6_18 = cells[is_end_6_18].groupby('Nationality')['Population'].sum()
        for label, (model_res, model_des, scen_type) in proj_models.items():
            df_long = _project_teachers(model_res, model_des, cells, scen_type, a_stats, y_stats,
                                        surplus, emp_dist, school_cols, str_lookup)
            if df_long is None:
                print(f"  {scen} / {label}: prediction FAILED")
                continue
            df_long['State'] = df_long['NUTS3'].str[:4]
            g = df_long.groupby(['Year', 'Nationality', 'State'], as_index=False)['Teachers'].sum()
            g['Specification'] = label
            g['Scenario'] = scen
            rows_t.append(g)
            stu = df_long[(df_long['Year'] == DEMAND_YEAR) & df_long['Age'].between(6, 18)].groupby(
                'Nationality')['Students'].sum()
            rows_er.append({'Specification': label, 'Scenario': scen, **(stu / pop_6_18).to_dict()})

    if not rows_t:
        print("No results generated.")
        return
    df_t = pd.concat(rows_t, ignore_index=True)

    tot = df_t.groupby(['Specification', 'Scenario', 'Year'])['Teachers'].sum().unstack('Year')
    end = df_t[df_t['Year'] == DEMAND_YEAR].groupby(
        ['Specification', 'Scenario', 'Nationality'])['Teachers'].sum().unstack('Nationality')
    summary = pd.DataFrame({
        'Growth': (tot[DEMAND_YEAR] / tot[growth_start] - 1) * 100,
        'Teachers_end': tot[DEMAND_YEAR],
        nats[0]: end[nats[0]],
        nats[1]: end[nats[1]],
        'Foreign_share': end[nats[1]] / end.sum(axis=1) * 100,
    })

    print(f"\n--- National Teacher Demand, median [min, max] across {len(proj_scenarios)} scenarios ---")
    print(f"{'Specification':>35s} {'Metric':>36s} {'Median':>12s} {'Min':>12s} {'Max':>12s}")
    metrics = [
        (f'Growth {growth_start}-{DEMAND_YEAR} (%)', 'Growth', '12.2f'),
        (f'Teachers {DEMAND_YEAR}', 'Teachers_end', '12,.0f'),
        (f'Teachers {DEMAND_YEAR}, {nats[0]}', nats[0], '12,.0f'),
        (f'Teachers {DEMAND_YEAR}, {nats[1]}', nats[1], '12,.0f'),
        (f'Foreign share of {DEMAND_YEAR} teachers (%)', 'Foreign_share', '12.2f'),
    ]
    for metric, col, fmt in metrics:
        for label in labels:
            s = summary.loc[label, col]
            print(f"  {label:>33s} {metric:>36s} {s.median():{fmt}} {s.min():{fmt}} {s.max():{fmt}}")

    ref = summary.loc['Trend (production)']
    alt = summary.loc['Trend + Nat x Year (uncapped)']
    diffs = [
        (f'Growth {growth_start}-{DEMAND_YEAR} (pp)', alt['Growth'] - ref['Growth']),
        (f'Teachers {DEMAND_YEAR} (%)', (alt['Teachers_end'] / ref['Teachers_end'] - 1) * 100),
        (f'Teachers {DEMAND_YEAR}, {nats[0]} (%)', (alt[nats[0]] / ref[nats[0]] - 1) * 100),
        (f'Teachers {DEMAND_YEAR}, {nats[1]} (%)', (alt[nats[1]] / ref[nats[1]] - 1) * 100),
        (f'Foreign share of {DEMAND_YEAR} teachers (pp)', alt['Foreign_share'] - ref['Foreign_share']),
    ]
    print(f"\n--- Nat x Year (uncapped) minus Production Trend, per scenario: median [min, max] ---")
    print(f"{'Metric':>45s} {'Median':>10s} {'Min':>10s} {'Max':>10s}")
    for metric, s in diffs:
        print(f"  {metric:>43s} {s.median():+10.2f} {s.min():+10.2f} {s.max():+10.2f}")

    print(f"\n--- Projected Crude Enrolment Ratio, Ages 6-18, {DEMAND_YEAR} (median across scenarios) ---")
    er = pd.DataFrame(rows_er).groupby('Specification')[nats].median().reindex(labels)
    print(er.to_string(float_format='%.4f'))

    print(f"\n--- Teacher-Demand Growth {growth_start}-{DEMAND_YEAR} by Federal State (%), "
          f"median [min, max] across scenarios ---")
    st = df_t.groupby(['Specification', 'Scenario', 'State', 'Year'])['Teachers'].sum().unstack('Year')
    st_growth = ((st[DEMAND_YEAR] / st[growth_start] - 1) * 100).unstack('Specification')
    print(f"{'State':>7s} " + " ".join(f"{label:>31s}" for label in labels))
    for state, grp in st_growth.groupby(level='State'):
        txt = [f"{grp[lab].median():+7.2f} [{grp[lab].min():+7.2f}, {grp[lab].max():+7.2f}]" for lab in labels]
        print(f"  {state:>5s} " + " ".join(f"{t:>31s}" for t in txt))

    print(f"\n--- Per Scenario: Growth {growth_start}-{DEMAND_YEAR} (%) and Foreign Share of "
          f"{DEMAND_YEAR} Teachers (%) ---")
    per_scen = pd.DataFrame({
        'SQ_growth': summary.loc['Status quo (production)', 'Growth'],
        'Trend_growth': ref['Growth'],
        'NatYear_growth': alt['Growth'],
        'Diff_pp': alt['Growth'] - ref['Growth'],
        'Trend_fshare': ref['Foreign_share'],
        'NatYear_fshare': alt['Foreign_share'],
    })
    print(per_scen.to_string(float_format='%.2f'))
    return summary


def main():
    print("=" * 80)
    print("EDUCATION MODEL ROBUSTNESS TESTS")
    print("=" * 80)

    print("\nLoading data...")
    df_ratio, df_teachers = load_historic_data()
    school_cols = sorted(df_ratio['School_Type'].unique().tolist())
    print(f"School types: {school_cols}")
    print(f"Years: {sorted(df_ratio['Year'].unique())}")
    print(f"Regions: {sorted(df_ratio['NUTS3'].unique())}")
    print(f"Nationalities: {df_ratio['Nationality'].unique()}")
    print(f"Total rows (ratio): {len(df_ratio):,}")

    print("\nLoading projections...")
    df_projection = prepare_population_projections()
    print(f"Projection scenarios: {df_projection['scenario'].nunique()}")
    print(f"Projection years: {sorted(df_projection['Year'].unique())}")

    test1_loyo_cv(df_ratio, school_cols)
    test2_str_sensitivity(df_ratio, df_teachers, df_projection, school_cols)
    test3_bootstrap(df_ratio, school_cols, n_boot=N_BOOTSTRAP)
    test4_school_type_stability(df_ratio, school_cols)
    test5_alternative_specs(df_ratio, school_cols)
    test6_leave_region_out(df_ratio, school_cols)
    test7_surplus_sensitivity(df_ratio, school_cols)
    test8_variance_decomposition(df_ratio, df_projection, school_cols)
    test9_nationality_year_slope(df_ratio, df_teachers, df_projection, school_cols)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
