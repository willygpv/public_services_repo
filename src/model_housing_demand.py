"""Household formation rates projected in logit space and calibrated to the OEROK household projection, and projected dwelling demand."""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config
from oerok_projection import load_oerok_reference

ALLOW_SIMPLE_FALLBACK = (
    '--allow-simple-fallback' in sys.argv
    or os.environ.get('HOUSING_ALLOW_SIMPLE_FALLBACK') == '1'
)


BASE_YEAR = 2022
DEMAND_YEAR = 2050
BASE_ANCHOR_YEAR = 2021.5
KEY_YEARS = [2030, 2040, 2050]
DEMOGRAPHIC_COLS = ['NUTS3', 'Age', 'Sex', 'Nationality']


def convert_age(age_str) -> int:
    if age_str == 'Y_GE100':
        return 100
    elif age_str == 'Y_LT1':
        return 0
    else:
        return int(age_str.lstrip('Y'))


def create_nuts3_oerok_mapping() -> dict:
    return {
        'AT111': 'Mittelburgenland', 'AT112': 'Nordburgenland', 'AT113': 'Südburgenland',
        'AT121': 'Mostviertel-Eisenwurzen', 'AT122': 'Niederösterreich-Süd',
        'AT123': 'Sankt Pölten', 'AT124': 'Waldviertel', 'AT125': 'Weinviertel',
        'AT126': 'Wiener Umland-Nord', 'AT127': 'Wiener Umland-Süd',
        'AT130': 'Wien',
        'AT211': 'Klagenfurt-Villach', 'AT212': 'Oberkärnten', 'AT213': 'Unterkärnten',
        'AT221': 'Graz', 'AT222': 'Liezen', 'AT223': 'Östliche Obersteiermark',
        'AT224': 'Oststeiermark', 'AT225': 'West- und Südsteiermark',
        'AT226': 'Westliche Obersteiermark',
        'AT311': 'Innviertel', 'AT312': 'Linz-Wels', 'AT313': 'Mühlviertel',
        'AT314': 'Steyr-Kirchdorf', 'AT315': 'Traunviertel',
        'AT321': 'Lungau', 'AT322': 'Pinzgau-Pongau', 'AT323': 'Salzburg und Umgebung',
        'AT331': 'Außerfern', 'AT332': 'Innsbruck', 'AT333': 'Osttirol',
        'AT334': 'Tiroler Oberland', 'AT335': 'Tiroler Unterland',
        'AT341': 'Bludenz-Bregenzer Wald', 'AT342': 'Rheintal-Bodensee',
    }


def logit_transform(hhfr_rate):
    rate = np.clip(hhfr_rate / 100.0, 1e-6, 1.0 - 1e-6)
    return np.log(rate / (1.0 - rate))


def sigmoid_transform(logit_value):
    return (1.0 / (1.0 + np.exp(-logit_value))) * 100.0


def forecast_cohort_hhfr(group_df, target_years):
    if len(group_df) < 2:
        last_value = group_df['HHFR'].iloc[-1]
        return pd.Series([last_value] * len(target_years), index=target_years)

    group_df = group_df.sort_values('Year')
    X = group_df['Year'].values.reshape(-1, 1)
    y_logit = logit_transform(group_df['HHFR'].values)

    model = LinearRegression()
    model.fit(X, y_logit)

    predictions_logit = model.predict(np.array(target_years).reshape(-1, 1))
    predictions = sigmoid_transform(predictions_logit)
    return pd.Series(predictions, index=target_years)


def simple_hhfr_extrapolation(df_historical, forecast_years):
    print("Using simple HHFR trend extrapolation (in logit-space)...")

    cohort_groups = ['NUTS3', 'Age', 'Sex', 'Nationality']
    extrapolated_results = []

    for name, group in df_historical.groupby(cohort_groups):
        nuts3, age, sex, nationality = name

        if len(group) >= 2:
            years = group['Year'].values
            hhfr_values_logit = logit_transform(group['HHFR'].values)
            for target_year in forecast_years:
                extrapolated_logit = np.interp(target_year, years, hhfr_values_logit)
                extrapolated_results.append({
                    'NUTS3': nuts3, 'Age': age, 'Sex': sex,
                    'Nationality': nationality, 'Year': target_year,
                    'HHFR': sigmoid_transform(extrapolated_logit),
                })
        else:
            last_hhfr = group['HHFR'].iloc[-1]
            for target_year in forecast_years:
                extrapolated_results.append({
                    'NUTS3': nuts3, 'Age': age, 'Sex': sex,
                    'Nationality': nationality, 'Year': target_year,
                    'HHFR': last_hhfr,
                })

    print(f"Simple extrapolation completed: {len(extrapolated_results)} forecasts")
    return pd.DataFrame(extrapolated_results)


def compute_base_hhfr(df_hs_historical):
    recent = df_hs_historical[df_hs_historical['Year'].isin([2021, 2022])].copy()
    if recent.empty:
        return pd.DataFrame(columns=DEMOGRAPHIC_COLS + ['HHFR'])
    recent['implied_heads'] = recent['Population'] * recent['HHFR'] / 100.0
    grouped = recent.groupby(DEMOGRAPHIC_COLS).agg(
        implied_heads=('implied_heads', 'sum'), Population=('Population', 'sum'),
        HHFR_mean=('HHFR', 'mean'),
    ).reset_index()
    grouped['HHFR'] = np.where(
        grouped['Population'] > 0,
        (grouped['implied_heads'] / grouped['Population']) * 100.0, grouped['HHFR_mean'],
    )
    return grouped[DEMOGRAPHIC_COLS + ['HHFR']]


def interpolate_hhfr_between_years(df_hhfr_key_years, all_forecast_years,
                                   df_base_hhfr=None, base_year=None):
    print(f"Interpolating HHFR for years: {min(all_forecast_years)}-{max(all_forecast_years)}")

    cohort_cols = ['NUTS3', 'Age', 'Sex', 'Nationality']
    interpolated_results = []

    if df_base_hhfr is not None:
        if base_year is None:
            base_year = BASE_ANCHOR_YEAR
        cohorts = df_hhfr_key_years[cohort_cols].drop_duplicates()
        anchor = cohorts.merge(df_base_hhfr[cohort_cols + ['HHFR']], on=cohort_cols, how='inner')
        n_missing = len(cohorts) - len(anchor)
        if n_missing:
            print(f"Warning: {n_missing} cohorts have no base-period HHFR anchor")
        anchor['Year'] = base_year
        df_hhfr_key_years = pd.concat(
            [anchor[cohort_cols + ['Year', 'HHFR']],
             df_hhfr_key_years[cohort_cols + ['Year', 'HHFR']]],
            ignore_index=True,
        )
        print(f"Interpolation anchors: {base_year} (base) + "
              f"{sorted(df_hhfr_key_years['Year'].unique().tolist())[1:]}")

    for cohort_key, cohort_data in df_hhfr_key_years.groupby(cohort_cols):
        cohort_sorted = cohort_data.sort_values('Year')

        if len(cohort_sorted) >= 2:
            years = cohort_sorted['Year'].values
            hhfr_values = cohort_sorted['HHFR'].values
            interpolated_hhfr = np.clip(np.interp(all_forecast_years, years, hhfr_values), 0, 100)
        else:
            interpolated_hhfr = np.full(len(all_forecast_years), cohort_sorted['HHFR'].iloc[0])

        for i, year in enumerate(all_forecast_years):
            interpolated_results.append({
                'NUTS3': cohort_key[0], 'Age': cohort_key[1],
                'Sex': cohort_key[2], 'Nationality': cohort_key[3],
                'Year': year, 'HHFR': interpolated_hhfr[i],
            })

    return pd.DataFrame(interpolated_results)


def load_regional_clusters() -> dict:
    try:
        cluster_file = pd.read_excel(r'Data/OEROK/Budget_forecasts/nuts3_coding_adapted.xlsx')
        print(f"Loaded cluster mapping: {cluster_file.shape}")
        cluster_mapping = dict(zip(cluster_file['Code'], cluster_file['Cluster']))
        print(f"Found {len(cluster_mapping)} NUTS3 -> Cluster mappings")
        return cluster_mapping
    except Exception as e:
        raise FileNotFoundError(f"Could not load cluster file: {e}")


def create_age_group_mapping() -> dict:
    age_to_group = {age: None for age in range(0, 15)}
    for start in range(15, 95, 5):
        label = f"{start}-{start + 4}"
        for age in range(start, start + 5):
            age_to_group[age] = label
    for age in range(95, 101):
        age_to_group[age] = "95-100"
    return age_to_group


def get_weighted_average_hhfr(df_subset) -> float:
    heads = (df_subset['Population'] * df_subset['HHFR'] / 100.0).sum()
    total_pop = df_subset['Population'].sum()
    return (heads / total_pop) * 100.0 if total_pop > 0 else 0.0


def calculate_robust_trends(df_hs_historical, cluster_mapping, age_group_mapping):
    print("Calculating robust trends (Anchor: 2011 -> Avg 2021/22)...")

    df_2011 = df_hs_historical[df_hs_historical['Year'] == 2011].copy()
    df_recent = df_hs_historical[df_hs_historical['Year'].isin([2021, 2022])].copy()

    if df_2011.empty or df_recent.empty:
        raise ValueError("Missing required data (2011 or 2021/2022)")

    for df in [df_2011, df_recent]:
        df['Cluster'] = df['NUTS3'].map(cluster_mapping)
        df['AgeGroup'] = df['Age'].map(age_group_mapping)

    df_2011 = df_2011[df_2011['AgeGroup'].notna()]
    df_recent = df_recent[df_recent['AgeGroup'].notna()]

    group_cols = ['Cluster', 'AgeGroup', 'Sex', 'Nationality']
    unique_groups = df_recent[group_cols].drop_duplicates()

    print(f"Processing trends for {len(unique_groups)} demographic groups...")

    trend_results = []
    for _, row in unique_groups.iterrows():
        cluster, age_grp, sex, nat = row['Cluster'], row['AgeGroup'], row['Sex'], row['Nationality']

        mask_11 = (
            (df_2011['Cluster'] == cluster) & (df_2011['AgeGroup'] == age_grp)
            & (df_2011['Sex'] == sex) & (df_2011['Nationality'] == nat)
        )
        mask_recent = (
            (df_recent['Cluster'] == cluster) & (df_recent['AgeGroup'] == age_grp)
            & (df_recent['Sex'] == sex) & (df_recent['Nationality'] == nat)
        )
        group_11 = df_2011[mask_11]
        group_recent = df_recent[mask_recent]

        if group_recent.empty:
            continue

        hhfr_B = get_weighted_average_hhfr(group_recent)

        if not group_11.empty:
            hhfr_A = get_weighted_average_hhfr(group_11)
            logit_A = logit_transform(hhfr_A)
            logit_B = logit_transform(hhfr_B)
            slope = (logit_B - logit_A) / (2021.5 - 2011)
        else:
            slope = 0

        trend_results.append({
            'Cluster': cluster, 'AgeGroup': age_grp, 'Sex': sex,
            'Nationality': nat, 'trend_slope': slope,
            'anchor_hhfr_logit': logit_transform(hhfr_B), 'anchor_year': 2021.5,
        })

    return pd.DataFrame(trend_results)


def apply_trends_to_cohorts(df_hs_historical, df_trends, cluster_mapping,
                            age_group_mapping, target_year):
    print(f"Applying trends to cohorts for year {target_year}...")

    df_recent = df_hs_historical[df_hs_historical['Year'].isin([2021, 2022])].copy()
    cohort_cols = ['NUTS3', 'Age', 'Sex', 'Nationality']

    def get_cohort_stats(x):
        total_pop = x['Population'].sum()
        weighted_hhfr = (
            ((x['Population'] * x['HHFR'] / 100.0).sum() / total_pop) * 100.0
            if total_pop > 0 else x['HHFR'].mean()
        )
        nuts3_code = x['NUTS3'].iloc[0]
        age_val = x['Age'].iloc[0]
        return pd.Series({
            'base_hhfr': weighted_hhfr,
            'Cluster': cluster_mapping.get(nuts3_code),
            'AgeGroup': age_group_mapping.get(age_val),
        })

    base_stats = df_recent.groupby(cohort_cols).apply(get_cohort_stats).reset_index()

    trend_lookup = {}
    for _, row in df_trends.iterrows():
        key = (row['Cluster'], row['AgeGroup'], row['Sex'], row['Nationality'])
        trend_lookup[key] = row['trend_slope']

    forecast_results = []
    for _, cohort in base_stats.iterrows():
        nuts3, age, sex, nat = cohort['NUTS3'], cohort['Age'], cohort['Sex'], cohort['Nationality']
        cluster, age_group = cohort['Cluster'], cohort['AgeGroup']
        base_hhfr = cohort['base_hhfr']

        if age_group is None:
            forecasted_hhfr = 0
        else:
            slope = trend_lookup.get((cluster, age_group, sex, nat))
            if slope is None:
                forecasted_hhfr = base_hhfr
            else:
                base_logit = logit_transform(base_hhfr)
                forecasted_hhfr = sigmoid_transform(
                    base_logit + slope * (target_year - 2021.5)
                )

        forecast_results.append({
            'NUTS3': nuts3, 'Age': age, 'Sex': sex, 'Nationality': nat,
            'Year': target_year, 'HHFR': forecasted_hhfr,
            'Cluster': cluster, 'AgeGroup': age_group,
        })

    return pd.DataFrame(forecast_results)


def apply_oerok_constraints_robust(df_forecast, oerok_reference, target_year):
    population = oerok_reference['population']
    population = population[population['Year'] == target_year][DEMOGRAPHIC_COLS + ['Population']]
    households = oerok_reference['households']
    households = households[households['Year'] == target_year].set_index('NUTS3')['Households']
    if population.empty or households.empty:
        raise ValueError(f"No OEROK population or household projection for {target_year}")

    merged = df_forecast.merge(population, on=DEMOGRAPHIC_COLS, how='inner')
    model_households = (merged['Population'] * merged['HHFR'] / 100).groupby(merged['NUTS3']).sum()
    adjustment_factors = (households / model_households).reindex(df_forecast['NUTS3'].unique())
    if adjustment_factors.isna().any():
        raise ValueError(f"No OEROK factor for {adjustment_factors[adjustment_factors.isna()].index.tolist()}")
    adjustment_factors = adjustment_factors.clip(0.1, 5.0)

    df_out = df_forecast.copy()
    df_out['adjustment_factor'] = df_out['NUTS3'].map(adjustment_factors)
    df_out['HHFR'] = np.clip(df_out['HHFR'] * df_out['adjustment_factor'], 0, 100)

    check = df_out.merge(population, on=DEMOGRAPHIC_COLS, how='inner')
    check_households = (check['Population'] * check['HHFR'] / 100).sum()
    print(f"Applied OEROK adjustments to {len(adjustment_factors)} regions for {target_year}: "
          f"factors {adjustment_factors.min():.3f}-{adjustment_factors.max():.3f} "
          f"(median {adjustment_factors.median():.3f})")
    print(f"  Households on the OEROK population: unconstrained {model_households.sum():,.0f}, "
          f"constrained {check_households:,.0f}, OEROK {households.sum():,.0f}")
    return df_out


def apply_rolling_smooth(hhfr_values, window_size):
    return pd.Series(hhfr_values).rolling(
        window=int(window_size), center=True, min_periods=1,
    ).mean().values


def apply_gaussian_smooth(hhfr_values, sigma):
    from scipy import ndimage
    return ndimage.gaussian_filter1d(hhfr_values, sigma=sigma, mode='nearest')


def apply_spline_smooth(ages, hhfr_values, smoothing_factor):
    from scipy.interpolate import UnivariateSpline
    try:
        spline = UnivariateSpline(ages, hhfr_values, s=smoothing_factor * len(ages))
        return spline(ages)
    except Exception:
        return hhfr_values


def smooth_hhfr_age_transitions(df_forecast, smoothing_method='rolling', smoothing_strength=3):
    print(f"Smoothing HHFR age transitions using {smoothing_method} method (strength={smoothing_strength})...")

    df_smoothed = df_forecast.copy()
    cohort_cols = ['NUTS3', 'Sex', 'Nationality', 'Year']

    smoothed_count = 0
    total_cohorts = len(df_forecast.groupby(cohort_cols))

    for name, group in df_forecast.groupby(cohort_cols):
        group_sorted = group.sort_values('Age').copy()
        ages = group_sorted['Age'].values
        hhfr_values = group_sorted['HHFR'].values

        if len(hhfr_values) < 5:
            continue

        if smoothing_method == 'rolling':
            smoothed_hhfr = apply_rolling_smooth(hhfr_values, smoothing_strength)
        elif smoothing_method == 'gaussian':
            smoothed_hhfr = apply_gaussian_smooth(hhfr_values, smoothing_strength)
        elif smoothing_method == 'spline':
            smoothed_hhfr = apply_spline_smooth(ages, hhfr_values, smoothing_strength)
        else:
            raise ValueError(f"Unknown smoothing method: {smoothing_method}")

        smoothed_hhfr = np.clip(smoothed_hhfr, 0, 100)

        mask = (
            (df_smoothed['NUTS3'] == name[0])
            & (df_smoothed['Sex'] == name[1])
            & (df_smoothed['Nationality'] == name[2])
            & (df_smoothed['Year'] == name[3])
        )
        age_to_hhfr = dict(zip(ages, smoothed_hhfr))
        df_smoothed.loc[mask, 'HHFR'] = df_smoothed.loc[mask, 'Age'].map(age_to_hhfr)
        smoothed_count += 1

    print(f"Smoothed {smoothed_count}/{total_cohorts} cohort age profiles")
    return df_smoothed


def smooth_age_group_boundaries(df_forecast, boundary_ages=None, blend_width=2):
    if boundary_ages is None:
        boundary_ages = [25, 35, 45, 55, 65, 75, 85]

    print(f"Smoothing specific age group boundaries: {boundary_ages}")

    df_smoothed = df_forecast.copy()
    cohort_cols = ['NUTS3', 'Sex', 'Nationality', 'Year']

    for name, group in df_forecast.groupby(cohort_cols):
        group_sorted = group.sort_values('Age').copy()
        ages = group_sorted['Age'].values
        hhfr_values = group_sorted['HHFR'].values.copy()

        for boundary_age in boundary_ages:
            if boundary_age not in ages:
                continue
            boundary_idx = np.where(ages == boundary_age)[0]
            if len(boundary_idx) == 0:
                continue
            boundary_idx = boundary_idx[0]
            start_idx = max(0, boundary_idx - blend_width)
            end_idx = min(len(ages), boundary_idx + blend_width + 1)

            if end_idx - start_idx >= 3:
                segment = hhfr_values[start_idx:end_idx]
                for i in range(1, len(segment) - 1):
                    segment[i] = 0.5 * segment[i] + 0.25 * segment[i - 1] + 0.25 * segment[i + 1]
                hhfr_values[start_idx:end_idx] = segment

        hhfr_values = np.clip(hhfr_values, 0, 100)

        mask = (
            (df_smoothed['NUTS3'] == name[0])
            & (df_smoothed['Sex'] == name[1])
            & (df_smoothed['Nationality'] == name[2])
            & (df_smoothed['Year'] == name[3])
        )
        age_to_hhfr = dict(zip(ages, hhfr_values))
        df_smoothed.loc[mask, 'HHFR'] = df_smoothed.loc[mask, 'Age'].map(age_to_hhfr)

    return df_smoothed


def forecast_hhfr_robust(df_hs_historical, oerok_reference, target_year=2050):
    print(f"ROBUST HHFR FORECASTING FOR {target_year}")
    print("=" * 60)

    print("\nStep 1: Loading regional clusters and age groups...")
    cluster_mapping = load_regional_clusters()
    age_group_mapping = create_age_group_mapping()

    print("\nStep 2: Calculating robust trends (2011-2021 only)...")
    df_trends = calculate_robust_trends(df_hs_historical, cluster_mapping, age_group_mapping)

    print(f"\nStep 3: Applying trends to individual cohorts for {target_year}...")
    df_forecast_base = apply_trends_to_cohorts(
        df_hs_historical, df_trends, cluster_mapping, age_group_mapping, target_year,
    )

    print("\nStep 4: Applying OEROK household constraints...")
    try:
        df_forecast_constrained = apply_oerok_constraints_robust(
            df_forecast_base, oerok_reference, target_year,
        )
        print("OEROK constraints applied successfully")
    except Exception as e:
        if not ALLOW_SIMPLE_FALLBACK:
            raise
        print(f"OEROK constraint application failed: {e}")
        print("Using unconstrained forecasts (explicitly allowed)...")
        df_forecast_constrained = df_forecast_base

    result = df_forecast_constrained[['NUTS3', 'Age', 'Sex', 'Nationality', 'Year', 'HHFR']].copy()

    print(f"\nROBUST FORECASTING COMPLETED!")
    print(f"Result shape: {result.shape}")
    print(f"HHFR range: {result['HHFR'].min():.2f} - {result['HHFR'].max():.2f}")
    return result


def forecast_hhfr_robust_with_smoothing(df_hs_historical, oerok_reference, target_year=2050,
                                        apply_smoothing=True, smoothing_method='rolling',
                                        smoothing_strength=3):
    df_forecast = forecast_hhfr_robust(df_hs_historical, oerok_reference, target_year)

    if apply_smoothing:
        print(f"\nApplying {smoothing_method} smoothing to age transitions...")
        df_forecast = smooth_hhfr_age_transitions(
            df_forecast, smoothing_method=smoothing_method,
            smoothing_strength=smoothing_strength,
        )
        print("Age transition smoothing completed")
    return df_forecast


def calculate_dwelling_demand_with_scenarios(df_hs_historical, population_projections,
                                             df_hhfr_forecasts, base_year):
    print("Calculating absolute dwelling scenarios (Memory Optimized, No Demand Cols)...")

    print("Calculating status quo HHFR (2021-2022 Weighted Average)...")
    sq_grouped = compute_base_hhfr(df_hs_historical)

    if not sq_grouped.empty:
        status_quo_hhfr = sq_grouped.rename(columns={'HHFR': 'HHFR_statusquo'})
        status_quo_hhfr['HHFR_statusquo'] = status_quo_hhfr['HHFR_statusquo'].astype('float32')
    else:
        status_quo_hhfr = pd.DataFrame(columns=DEMOGRAPHIC_COLS + ['HHFR_statusquo'])

    historical_df = df_hs_historical.copy()
    if 'scenario' not in historical_df.columns:
        historical_df['scenario'] = 'historical'
    keep_cols = DEMOGRAPHIC_COLS + ['Year', 'Population', 'Dwellings', 'HHFR', 'scenario']
    historical_df = historical_df[keep_cols]
    results_dfs = [historical_df]

    future_years = [y for y in df_hhfr_forecasts['Year'].unique() if y >= base_year]
    population_scenarios = population_projections['scenario'].unique()
    df_hhfr_forecasts['HHFR'] = df_hhfr_forecasts['HHFR'].astype('float32')

    for pop_scenario in population_scenarios:
        print(f"Processing population scenario: {pop_scenario}")
        scenario_pop_all = population_projections[
            population_projections['scenario'] == pop_scenario
        ].copy()
        scenario_pop_all['Population'] = scenario_pop_all['Population'].astype('float32')

        for year in future_years:
            pop_data = scenario_pop_all[scenario_pop_all['Year'] == year]
            if pop_data.empty:
                continue

            hhfr_trend = df_hhfr_forecasts[df_hhfr_forecasts['Year'] == year]
            if not hhfr_trend.empty:
                merged_trend = pop_data.merge(hhfr_trend, on=DEMOGRAPHIC_COLS + ['Year'], how='inner')
                if not merged_trend.empty:
                    merged_trend['Dwellings'] = merged_trend['Population'] * (merged_trend['HHFR'] / 100.0)
                    merged_trend['scenario'] = f"{pop_scenario}--trend"
                    results_dfs.append(merged_trend)

            if not status_quo_hhfr.empty:
                merged_sq = pop_data.merge(status_quo_hhfr, on=DEMOGRAPHIC_COLS, how='left')
                if not merged_sq.empty:
                    merged_sq['HHFR_statusquo'] = merged_sq['HHFR_statusquo'].fillna(0)
                    merged_sq['Dwellings'] = merged_sq['Population'] * (merged_sq['HHFR_statusquo'] / 100.0)
                    merged_sq['scenario'] = f"{pop_scenario}--statusquo"
                    merged_sq.rename(columns={'HHFR_statusquo': 'HHFR'}, inplace=True)
                    results_dfs.append(merged_sq)

    print("Concatenating results...")
    if not results_dfs:
        print("No results generated")
        return {
            'dwelling_demand_detailed': pd.DataFrame(),
            'dwelling_demand_national': pd.DataFrame(),
        }

    df_combined = pd.concat(results_dfs, ignore_index=True)
    del results_dfs

    for col in ['Population', 'Dwellings', 'HHFR']:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna(0).astype('float32')

    national_totals = (
        df_combined.groupby(['Year', 'scenario'])
        .agg({'Population': 'sum', 'Dwellings': 'sum'})
        .reset_index()
    )

    print(f"Final results: {df_combined.shape} detailed, {national_totals.shape} national")
    return {
        'dwelling_demand_detailed': df_combined,
        'dwelling_demand_national': national_totals,
        'hhfr_forecasts': df_hhfr_forecasts,
        'status_quo_hhfr': status_quo_hhfr,
    }


def complete_dwelling_demand_workflow_robust(
    df_hs_historical, population_projections, oerok_reference, base_year,
    forecast_years=None, apply_smoothing=True,
    smoothing_method='rolling', smoothing_strength=3,
):
    if forecast_years is None:
        forecast_years = list(range(base_year + 1, 2051))

    print("COMPLETE DWELLING DEMAND WORKFLOW (ROBUST + SMOOTH VERSION)")
    print("=" * 70)

    print("\nStep 1: OEROK reference: "
          f"{oerok_reference['population']['Year'].nunique()} population years, "
          f"{oerok_reference['households']['Year'].nunique()} household years")

    print(f"\nStep 2: Generating ROBUST HHFR forecasts...")
    print("Using age grouping (8 groups) + regional clustering (5 clusters) + 2011-2021 trends only")
    if apply_smoothing:
        print(f"+ {smoothing_method} smoothing (strength={smoothing_strength})")

    try:
        all_hhfr_forecasts = []

        for year in KEY_YEARS:
            if year <= max(forecast_years):
                print(f"\n--- Robust forecasting for {year} ---")
                year_forecast = forecast_hhfr_robust(
                    df_hs_historical, oerok_reference, target_year=year,
                )
                if apply_smoothing:
                    print(f"Applying {smoothing_method} smoothing...")
                    year_forecast = smooth_hhfr_age_transitions(
                        year_forecast, smoothing_method=smoothing_method,
                        smoothing_strength=smoothing_strength,
                    )
                    print("Smoothing completed")
                all_hhfr_forecasts.append(year_forecast)

        if all_hhfr_forecasts:
            df_hhfr_constrained = pd.concat(all_hhfr_forecasts, ignore_index=True)
            df_hhfr_complete = interpolate_hhfr_between_years(
                df_hhfr_constrained, forecast_years,
                df_base_hhfr=compute_base_hhfr(df_hs_historical),
                base_year=BASE_ANCHOR_YEAR,
            )
            print(f"Robust + Smooth HHFR forecasts completed: {df_hhfr_complete.shape}")
        else:
            raise ValueError("No robust forecasts generated")

    except Exception as e:
        if not ALLOW_SIMPLE_FALLBACK:
            raise
        print(f"Robust forecasting failed: {e}")
        print("Falling back to simple extrapolation (explicitly allowed)...")
        df_hhfr_complete = simple_hhfr_extrapolation(df_hs_historical, forecast_years)

    print("\nStep 3: Calculating dwelling demand with trend and status quo scenarios...")
    results = calculate_dwelling_demand_with_scenarios(
        df_hs_historical, population_projections, df_hhfr_complete, base_year,
    )

    print("Robust + Smooth workflow completed successfully!")
    return results


def main():
    base_year = BASE_YEAR
    demand_year = DEMAND_YEAR

    df_hs = pd.read_csv(r'XLSX/df_hs_np_clean.csv')
    df_hs['scenario'] = 'BSL'
    df_hs.head()

    df_hs[df_hs['Year'] == 2011]['Dwellings'].sum()

    projected_path = config.PATH_PROJECTION
    projected_all = pd.read_parquet(projected_path)
    projected_all.rename(columns={
        'sex': 'Sex', 'age': 'Age', 'OBS_VALUE': 'Population',
        'geo': 'NUTS3', 'TIME_PERIOD': 'Year',
    }, inplace=True)

    projected_all['Age'] = projected_all['Age'].apply(convert_age)
    projected_all['Population'] = projected_all['Population'].apply(lambda x: max(x, 0))
    projected_all = projected_all.groupby(
        ['Year', 'NUTS3', 'Age', 'Nationality', 'Sex', 'scenario'], as_index=False,
    )['Population'].sum()

    df_proj_nosex = projected_all[projected_all['Year'] <= demand_year]
    print(df_proj_nosex.head())

    oerok_reference = load_oerok_reference(KEY_YEARS)
    for year in KEY_YEARS:
        pop = oerok_reference['population']
        hh = oerok_reference['households']
        print(f"OEROK {year}: population {pop.loc[pop['Year'] == year, 'Population'].sum():,.0f}, "
              f"households {hh.loc[hh['Year'] == year, 'Households'].sum():,.0f}")

    print("RUN THE COMPLETE WORKING WORKFLOW:")
    print("=" * 50)

    results = complete_dwelling_demand_workflow_robust(
        df_hs_historical=df_hs,
        population_projections=df_proj_nosex,
        oerok_reference=oerok_reference,
        base_year=base_year,
        forecast_years=list(range(2023, 2051)),
        apply_smoothing=True,
        smoothing_method='rolling',
        smoothing_strength=3,
    )

    df_dw_demand = results['dwelling_demand_detailed']
    df_dw_demand.head()

    df_dw_demand[df_dw_demand['Year'] == 2022][['Population', 'scenario']].sum()

    df_base = df_dw_demand[df_dw_demand['Year'] == base_year].copy()
    df_others = df_dw_demand[df_dw_demand['Year'] != base_year].copy()

    base_combos = set(df_base[['NUTS3', 'Age', 'Sex', 'Nationality']].itertuples(index=False, name=None))
    other_combos = set(df_others[['NUTS3', 'Age', 'Sex', 'Nationality']].itertuples(index=False, name=None))
    missing_combos = other_combos - base_combos

    print(f"Number of base-year combos: {len(base_combos)}")
    print(f"Number of other-year combos: {len(other_combos)}")
    print(f"Number of combos missing from base year: {len(missing_combos)}")

    if missing_combos:
        idx = pd.MultiIndex.from_frame(df_others[['NUTS3', 'Age', 'Sex', 'Nationality']])
        df_missing = df_others[idx.isin(missing_combos)]
        print("Missing combos DataFrame:")
        print(df_missing)
        print(df_missing['Age'].unique())
    else:
        print("Missing combos DataFrame: none")

    output_path = config.PATH_HOUSING_DEMAND

    df_dw_demand.round(4)
    df_dw_demand.to_parquet(output_path, index=False, compression='gzip')
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
