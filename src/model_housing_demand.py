#!/usr/bin/env python
"""
Housing Demand — EU Scenarios with Logit-Transformed HHFR Forecasting

Pipeline:
  1. Load historical headship-rate data (from previous script's output)
  2. Load population projections & OEROK household-size data
  3. Forecast HHFR via logit-space trend extrapolation with OEROK constraints
  4. Calculate dwelling demand under trend & status-quo scenarios
  5. Validate, attach geometries, and export
"""

import gc
import math
import os
import warnings
from io import BytesIO, StringIO

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

# Try importing optional dependencies (used in some smoothing / plotting paths)
try:
    import cartogram
except ImportError:
    pass


# =============================================================================
# 1. CONSTANTS
# =============================================================================

BASE_YEAR = 2022
DEMAND_YEAR = 2050
DEMOGRAPHIC_COLS = ['NUTS3', 'Age', 'Sex', 'Nationality']


# =============================================================================
# 2. PLOTTING HELPERS (carried over from headship-rate notebook)
# =============================================================================

def create_age_group_mapping_5yr() -> dict:
    """Map integer ages 0–100 to 5-year age-group labels for plotting."""
    age_to_group = {}
    for age in range(0, 15):
        age_to_group[age] = '0-14'
    for start in range(15, 95, 5):
        label = f"{start}-{start + 4}"
        for age in range(start, start + 5):
            age_to_group[age] = label
    for age in range(95, 101):
        age_to_group[age] = "95-100"
    return age_to_group


def plot_headship_rate_subplots(df, group_col, filter_col, title_prefix, color_map):
    """Plot HHFR time-series by *group_col* facets, coloured by *filter_col*."""
    print("Applying 5-year 'AgeGroup' mapping for plotting...")
    age_group_map = create_age_group_mapping_5yr()
    df_plot = df.copy()
    df_plot['AgeGroup'] = df_plot['Age'].map(age_group_map)
    df_plot = df_plot.dropna(subset=['AgeGroup'])

    unique_groups = df_plot[group_col].unique()
    if all(isinstance(g, str) for g in unique_groups) and any('-' in g for g in unique_groups):
        unique_groups = sorted(unique_groups, key=lambda x: int(x.split('-')[0]))

    cols = math.ceil(math.sqrt(len(unique_groups)))
    rows = math.ceil(len(unique_groups) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    df_plot['Year'] = pd.to_numeric(df_plot['Year'])

    handles, labels = [], []
    for i, group in enumerate(unique_groups):
        ax = axes[i]
        df_group = df_plot[df_plot[group_col] == group]
        for fval in df_group[filter_col].unique():
            df_filt = df_group[df_group[filter_col] == fval]
            df_agg = df_filt.groupby('Year')['HHFR'].mean().reset_index().sort_values('Year')
            line, = ax.plot(
                df_agg['Year'], df_agg['HHFR'],
                label=str(fval), color=color_map.get(fval, 'gray'),
                marker='o', markersize=3,
            )
            if fval not in labels:
                handles.append(line)
                labels.append(fval)
        ax.set_title(f'{title_prefix}: {group}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Headship Rate (HHFR)')
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(len(unique_groups), len(axes)):
        fig.delaxes(axes[j])

    fig.legend(handles, labels, loc='center left', fontsize='medium',
               bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()


# =============================================================================
# 3. POPULATION PROJECTION HELPERS
# =============================================================================

def convert_age(age_str) -> int:
    """Convert Eurostat-style age strings (Y_GE100, Y_LT1, Y25, …) to int."""
    if age_str == 'Y_GE100':
        return 100
    elif age_str == 'Y_LT1':
        return 0
    else:
        return int(age_str.lstrip('Y'))


# =============================================================================
# 4. OEROK DATA PROCESSING
# =============================================================================

def process_oerok_data(oerok_raw: pd.DataFrame) -> pd.DataFrame:
    """Melt wide-format OEROK data to long; handles int/string year columns."""
    if 'Kenn-zahl' in oerok_raw.columns:
        oerok_clean = oerok_raw.drop('Kenn-zahl', axis=1)
    else:
        oerok_clean = oerok_raw.copy()

    year_cols = [
        col for col in oerok_clean.columns
        if (isinstance(col, (int, float)) and 2000 <= col <= 2100)
        or (isinstance(col, str) and col.isdigit() and 2000 <= int(col) <= 2100)
    ]
    print(f"Found year columns: {year_cols}")

    oerok_long = pd.melt(
        oerok_clean,
        id_vars=['Region'],
        value_vars=year_cols,
        var_name='Year',
        value_name='avg_household_size',
    )
    oerok_long['Year'] = pd.to_numeric(oerok_long['Year'], errors='coerce').astype('Int64')
    oerok_long = oerok_long.dropna(subset=['Year'])
    return oerok_long


def process_oerok_data_complete(oerok_raw: pd.DataFrame) -> pd.DataFrame:
    """Process OEROK data — handles both integer and string year columns."""
    print("Processing OEROK data...")
    oerok_clean = oerok_raw.copy()
    if 'Kenn-zahl' in oerok_clean.columns:
        oerok_clean = oerok_clean.drop('Kenn-zahl', axis=1)

    year_cols = [
        col for col in oerok_clean.columns
        if (isinstance(col, (int, float)) and 2000 <= col <= 2100)
        or (isinstance(col, str) and col.isdigit() and 2000 <= int(col) <= 2100)
    ]
    print(f"Found year columns: {year_cols}")
    if not year_cols:
        raise ValueError("No year columns found in OEROK data!")

    id_cols = [col for col in oerok_clean.columns if col not in year_cols]
    oerok_long = pd.melt(
        oerok_clean, id_vars=id_cols, value_vars=year_cols,
        var_name='Year', value_name='avg_household_size',
    )
    oerok_long['Year'] = pd.to_numeric(oerok_long['Year'], errors='coerce').astype('Int64')
    oerok_long = oerok_long.dropna(subset=['Year'])

    print(f"Processed OEROK data: {oerok_long.shape}, years: {oerok_long['Year'].min()}-{oerok_long['Year'].max()}")
    return oerok_long


# =============================================================================
# 5. NUTS3 ↔ OEROK MAPPING
# =============================================================================

def create_nuts3_oerok_mapping() -> dict:
    """Static mapping from NUTS3 codes to OEROK region names."""
    return {
        # Burgenland
        'AT111': 'Mittelburgenland', 'AT112': 'Nordburgenland', 'AT113': 'Südburgenland',
        # Niederösterreich
        'AT121': 'Mostviertel-Eisenwurzen', 'AT122': 'Niederösterreich-Süd',
        'AT123': 'Sankt Pölten', 'AT124': 'Waldviertel', 'AT125': 'Weinviertel',
        'AT126': 'Wiener Umland-Nord', 'AT127': 'Wiener Umland-Süd',
        # Wien
        'AT130': 'Wien',
        # Kärnten
        'AT211': 'Klagenfurt-Villach', 'AT212': 'Oberkärnten', 'AT213': 'Unterkärnten',
        # Steiermark
        'AT221': 'Graz', 'AT222': 'Liezen', 'AT223': 'Östliche Obersteiermark',
        'AT224': 'Oststeiermark', 'AT225': 'West- und Südsteiermark',
        'AT226': 'Westliche Obersteiermark',
        # Oberösterreich
        'AT311': 'Innviertel', 'AT312': 'Linz-Wels', 'AT313': 'Mühlviertel',
        'AT314': 'Steyr-Kirchdorf', 'AT315': 'Traunviertel',
        # Salzburg
        'AT321': 'Lungau', 'AT322': 'Pinzgau-Pongau', 'AT323': 'Salzburg und Umgebung',
        # Tirol
        'AT331': 'Außerfern', 'AT332': 'Innsbruck', 'AT333': 'Osttirol',
        'AT334': 'Tiroler Oberland', 'AT335': 'Tiroler Unterland',
        # Vorarlberg
        'AT341': 'Bludenz-Bregenzer Wald', 'AT342': 'Rheintal-Bodensee',
    }


# =============================================================================
# 6. LOGIT / SIGMOID TRANSFORMS
# =============================================================================

def logit_transform(hhfr_rate):
    """Transform a 0–100 rate to logit-space (unbounded)."""
    rate = np.clip(hhfr_rate / 100.0, 1e-6, 1.0 - 1e-6)
    return np.log(rate / (1.0 - rate))


def sigmoid_transform(logit_value):
    """Transform an unbounded logit value back to a 0–100 rate."""
    return (1.0 / (1.0 + np.exp(-logit_value))) * 100.0


# =============================================================================
# 7. HHFR FORECASTING FUNCTIONS
# =============================================================================

def forecast_cohort_hhfr(group_df, target_years):
    """Forecast HHFR for a single cohort via linear regression in logit-space."""
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


def calculate_regional_household_size(hhfr_data, population_data, gqr_data):
    """Calculate average household size by region from HHFR, population, and GQR."""
    merged = hhfr_data.merge(population_data, on=['NUTS3', 'Age', 'Sex', 'Nationality'], how='inner')
    merged = merged.merge(
        gqr_data[['NUTS3', 'Age', 'Sex', 'Nationality', 'GQR']],
        on=['NUTS3', 'Age', 'Sex', 'Nationality'], how='left',
    )
    merged['GQR'] = merged['GQR'].fillna(0)
    merged['households'] = merged['Population'] * merged['HHFR'] / 100
    merged['private_pop'] = merged['Population'] * (1 - merged['GQR'])

    regional = merged.groupby('NUTS3').agg(
        {'households': 'sum', 'private_pop': 'sum'}
    ).reset_index()
    regional['avg_household_size'] = np.where(
        regional['households'] > 0,
        regional['private_pop'] / regional['households'],
        0,
    )
    return regional


def simple_hhfr_extrapolation(df_historical, forecast_years):
    """Fallback: simple linear extrapolation in logit-space."""
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


def interpolate_hhfr_between_years(df_hhfr_key_years, all_forecast_years):
    """Interpolate HHFR for years between key forecast years."""
    print(f"Interpolating HHFR for years: {min(all_forecast_years)}-{max(all_forecast_years)}")

    cohort_cols = ['NUTS3', 'Age', 'Sex', 'Nationality']
    interpolated_results = []

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


# =============================================================================
# 8. CLUSTER / AGE-GROUP MAPPINGS FOR ROBUST TRENDS
# =============================================================================

def load_regional_clusters() -> dict:
    """Load NUTS3 → cluster mapping from Excel file."""
    try:
        cluster_file = pd.read_excel(r'Data/OEROK/Budget_forecasts/nuts3_coding_adapted.xlsx')
        print(f"Loaded cluster mapping: {cluster_file.shape}")
        cluster_mapping = dict(zip(cluster_file['Code'], cluster_file['Cluster']))
        print(f"Found {len(cluster_mapping)} NUTS3 → Cluster mappings")
        return cluster_mapping
    except Exception as e:
        raise FileNotFoundError(f"Could not load cluster file: {e}")


def create_age_group_mapping() -> dict:
    """
    Map integer ages to 5-year age-group labels for trend calculation.

    Ages 0–14 → None (no household formation).
    """
    age_to_group = {age: None for age in range(0, 15)}
    for start in range(15, 95, 5):
        label = f"{start}-{start + 4}"
        for age in range(start, start + 5):
            age_to_group[age] = label
    for age in range(95, 101):
        age_to_group[age] = "95-100"
    return age_to_group


def get_weighted_average_hhfr(df_subset) -> float:
    """Population-weighted average HHFR for a data subset."""
    heads = (df_subset['Population'] * df_subset['HHFR'] / 100.0).sum()
    total_pop = df_subset['Population'].sum()
    return (heads / total_pop) * 100.0 if total_pop > 0 else 0.0


# =============================================================================
# 9. ROBUST TREND CALCULATION & APPLICATION
# =============================================================================

def calculate_robust_trends(df_hs_historical, cluster_mapping, age_group_mapping):
    """
    Calculate HHFR trends using 2011 as start and average(2021, 2022) as end.
    Fits in logit-space.
    """
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
    """Project individual cohorts starting from their weighted 2021/2022 average."""
    print(f"Applying trends to cohorts for year {target_year}...")

    df_recent = df_hs_historical[df_hs_historical['Year'].isin([2021, 2022])].copy()
    cohort_cols = ['NUTS3', 'Age', 'Sex', 'Nationality']

    def get_cohort_stats(x):
        total_pop = x['Population'].sum()
        weighted_hhfr = (
            ((x['Population'] * x['HHFR'] / 100.0).sum() / total_pop) * 100.0
            if total_pop > 0 else 0.0
        )
        nuts3_code = x['NUTS3'].iloc[0]
        age_val = x['Age'].iloc[0]
        return pd.Series({
            'base_hhfr': weighted_hhfr,
            'Cluster': cluster_mapping.get(nuts3_code),
            'AgeGroup': age_group_mapping.get(age_val),
        })

    base_stats = df_recent.groupby(cohort_cols).apply(get_cohort_stats).reset_index()

    # Build trend lookup
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


# =============================================================================
# 10. OEROK CONSTRAINT APPLICATION
# =============================================================================

def apply_oerok_constraints_robust(df_forecast, df_hs_historical, oerok_data, target_year):
    """Apply OEROK constraints at NUTS3 level."""
    nuts3_mapping = create_nuts3_oerok_mapping()

    latest_year = df_hs_historical['Year'].max()
    population_target = df_hs_historical[df_hs_historical['Year'] == latest_year][
        ['NUTS3', 'Age', 'Sex', 'Nationality', 'Population']
    ].copy()
    gqr_target = df_hs_historical[df_hs_historical['Year'] == latest_year][
        ['NUTS3', 'Age', 'Sex', 'Nationality', 'GQR']
    ].copy()

    current_sizes = calculate_regional_household_size(df_forecast, population_target, gqr_target)

    if 'Year' in oerok_data.columns and 'avg_household_size' in oerok_data.columns:
        oerok_targets = oerok_data[oerok_data['Year'] == target_year][['Region', 'avg_household_size']].copy()
        oerok_targets.columns = ['OEROK_Region', 'target_size']
        if oerok_targets.empty:
            print(f"No OEROK data for {target_year}, returning unconstrained forecasts")
            return df_forecast
    else:
        print("OEROK data format not recognized, returning unconstrained forecasts")
        return df_forecast

    current_sizes['OEROK_Region'] = current_sizes['NUTS3'].map(nuts3_mapping)
    current_sizes = current_sizes.dropna(subset=['OEROK_Region'])
    size_comparison = current_sizes.merge(oerok_targets, on='OEROK_Region', how='inner')

    if size_comparison.empty:
        print("No matching regions for OEROK constraints")
        return df_forecast

    adjustment_factors = {}
    for _, row in size_comparison.iterrows():
        nuts3 = row['NUTS3']
        current_size = row['avg_household_size']
        target_size = row['target_size']
        if current_size > 0:
            adjustment_factors[nuts3] = np.clip(current_size / target_size, 0.1, 5.0)
        else:
            adjustment_factors[nuts3] = 1.0

    df_out = df_forecast.copy()
    df_out['adjustment_factor'] = df_out['NUTS3'].map(adjustment_factors).fillna(1.0)
    df_out['HHFR'] = np.clip(df_out['HHFR'] * df_out['adjustment_factor'], 0, 100)
    print(f"Applied OEROK adjustments to {len(adjustment_factors)} regions")
    return df_out


# =============================================================================
# 11. SMOOTHING FUNCTIONS
# =============================================================================

def apply_rolling_smooth(hhfr_values, window_size):
    """Apply centred rolling-average smoothing."""
    return pd.Series(hhfr_values).rolling(
        window=int(window_size), center=True, min_periods=1,
    ).mean().values


def apply_gaussian_smooth(hhfr_values, sigma):
    """Apply Gaussian-filter smoothing."""
    from scipy import ndimage
    return ndimage.gaussian_filter1d(hhfr_values, sigma=sigma, mode='nearest')


def apply_spline_smooth(ages, hhfr_values, smoothing_factor):
    """Apply cubic-spline smoothing."""
    from scipy.interpolate import UnivariateSpline
    try:
        spline = UnivariateSpline(ages, hhfr_values, s=smoothing_factor * len(ages))
        return spline(ages)
    except Exception:
        return hhfr_values


def smooth_hhfr_age_transitions(df_forecast, smoothing_method='rolling', smoothing_strength=3):
    """Smooth HHFR transitions across age groups to reduce sharp jumps."""
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
    """Specifically smooth age-group boundaries with targeted blending."""
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


def compare_smoothing_methods(df_forecast, sample_cohort=None):
    """Compare different smoothing methods for a sample cohort."""
    if sample_cohort is None:
        cohort_cols = ['NUTS3', 'Sex', 'Nationality', 'Year']
        sample_cohort = next(iter(df_forecast.groupby(cohort_cols)))[0]

    print(f"Comparing smoothing methods for sample cohort: {sample_cohort}")

    mask = (
        (df_forecast['NUTS3'] == sample_cohort[0])
        & (df_forecast['Sex'] == sample_cohort[1])
        & (df_forecast['Nationality'] == sample_cohort[2])
        & (df_forecast['Year'] == sample_cohort[3])
    )
    sample_data = df_forecast[mask].sort_values('Age')
    ages = sample_data['Age'].values
    original_hhfr = sample_data['HHFR'].values

    methods = {
        'original': original_hhfr,
        'rolling_3': apply_rolling_smooth(original_hhfr, 3),
        'rolling_5': apply_rolling_smooth(original_hhfr, 5),
        'gaussian_1.5': apply_gaussian_smooth(original_hhfr, 1.5),
        'gaussian_2.0': apply_gaussian_smooth(original_hhfr, 2.0),
    }

    plt.figure(figsize=(12, 8))
    colors = ['black', 'blue', 'green', 'red', 'purple']
    for i, (method, values) in enumerate(methods.items()):
        plt.plot(
            ages, values, label=method, color=colors[i % len(colors)],
            linewidth=2 if method == 'original' else 1.5,
            alpha=0.8 if method != 'original' else 1.0,
        )
    plt.xlabel('Age')
    plt.ylabel('HHFR')
    plt.title(f'Smoothing Method Comparison\n{sample_cohort}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return methods


# =============================================================================
# 12. ROBUST FORECAST ORCHESTRATORS
# =============================================================================

def forecast_hhfr_robust(df_hs_historical, oerok_data, target_year=2050):
    """Main function: robust HHFR forecasting with age/regional grouping."""
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

    print("\nStep 4: Applying OEROK household size constraints...")
    try:
        df_forecast_constrained = apply_oerok_constraints_robust(
            df_forecast_base, df_hs_historical, oerok_data, target_year,
        )
        print("OEROK constraints applied successfully")
    except Exception as e:
        print(f"OEROK constraint application failed: {e}")
        print("Using unconstrained forecasts...")
        df_forecast_constrained = df_forecast_base

    result = df_forecast_constrained[['NUTS3', 'Age', 'Sex', 'Nationality', 'Year', 'HHFR']].copy()

    print(f"\nROBUST FORECASTING COMPLETED!")
    print(f"Result shape: {result.shape}")
    print(f"HHFR range: {result['HHFR'].min():.2f} - {result['HHFR'].max():.2f}")
    return result


def forecast_hhfr_robust_with_smoothing(df_hs_historical, oerok_data, target_year=2050,
                                        apply_smoothing=True, smoothing_method='rolling',
                                        smoothing_strength=3):
    """Enhanced: Robust HHFR forecasting with optional age-transition smoothing."""
    df_forecast = forecast_hhfr_robust(df_hs_historical, oerok_data, target_year)

    if apply_smoothing:
        print(f"\nApplying {smoothing_method} smoothing to age transitions...")
        df_forecast = smooth_hhfr_age_transitions(
            df_forecast, smoothing_method=smoothing_method,
            smoothing_strength=smoothing_strength,
        )
        print("Age transition smoothing completed")
    return df_forecast


# =============================================================================
# 13. DWELLING DEMAND CALCULATION
# =============================================================================

def calculate_dwelling_demand_with_scenarios(df_hs_historical, population_projections,
                                             df_hhfr_forecasts, base_year):
    """
    Calculate absolute dwelling stock/demand with trend & status-quo HHFR scenarios.
    Memory-optimised via list-of-DataFrames + pd.concat.
    """
    print("Calculating absolute dwelling scenarios (Memory Optimized, No Demand Cols)...")

    # 1. Status-quo HHFR (weighted avg 2021–2022)
    print("Calculating status quo HHFR (2021-2022 Weighted Average)...")
    status_quo_data = df_hs_historical[df_hs_historical['Year'].isin([2021, 2022])].copy()

    if not status_quo_data.empty:
        status_quo_data['implied_heads'] = status_quo_data['Population'] * status_quo_data['HHFR'] / 100.0
        sq_grouped = status_quo_data.groupby(DEMOGRAPHIC_COLS).agg(
            {'implied_heads': 'sum', 'Population': 'sum'}
        ).reset_index()
        sq_grouped['HHFR_statusquo'] = np.where(
            sq_grouped['Population'] > 0,
            (sq_grouped['implied_heads'] / sq_grouped['Population']) * 100.0, 0,
        )
        status_quo_hhfr = sq_grouped[DEMOGRAPHIC_COLS + ['HHFR_statusquo']].copy()
        status_quo_hhfr['HHFR_statusquo'] = status_quo_hhfr['HHFR_statusquo'].astype('float32')
    else:
        status_quo_hhfr = pd.DataFrame(columns=DEMOGRAPHIC_COLS + ['HHFR_statusquo'])

    # 2. Historical data
    historical_df = df_hs_historical.copy()
    if 'scenario' not in historical_df.columns:
        historical_df['scenario'] = 'historical'
    keep_cols = DEMOGRAPHIC_COLS + ['Year', 'Population', 'Dwellings', 'HHFR', 'scenario']
    historical_df = historical_df[keep_cols]
    results_dfs = [historical_df]

    # 3. Future projections
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

            # A. Trend scenario
            hhfr_trend = df_hhfr_forecasts[df_hhfr_forecasts['Year'] == year]
            if not hhfr_trend.empty:
                merged_trend = pop_data.merge(hhfr_trend, on=DEMOGRAPHIC_COLS + ['Year'], how='inner')
                if not merged_trend.empty:
                    merged_trend['Dwellings'] = merged_trend['Population'] * (merged_trend['HHFR'] / 100.0)
                    merged_trend['scenario'] = f"{pop_scenario}--trend"
                    results_dfs.append(merged_trend)

            # B. Status-quo scenario
            if not status_quo_hhfr.empty:
                merged_sq = pop_data.merge(status_quo_hhfr, on=DEMOGRAPHIC_COLS, how='left')
                if not merged_sq.empty:
                    merged_sq['HHFR_statusquo'] = merged_sq['HHFR_statusquo'].fillna(0)
                    merged_sq['Dwellings'] = merged_sq['Population'] * (merged_sq['HHFR_statusquo'] / 100.0)
                    merged_sq['scenario'] = f"{pop_scenario}--statusquo"
                    merged_sq.rename(columns={'HHFR_statusquo': 'HHFR'}, inplace=True)
                    results_dfs.append(merged_sq)

    # 4. Final assembly
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


# =============================================================================
# 14. COMPLETE WORKFLOW ORCHESTRATOR
# =============================================================================

def complete_dwelling_demand_workflow_robust(
    df_hs_historical, population_projections, oerok_raw_data, base_year,
    forecast_years=None, apply_smoothing=True,
    smoothing_method='rolling', smoothing_strength=3,
):
    """
    Full workflow: robust HHFR forecasting → dwelling demand with scenarios.
    """
    if forecast_years is None:
        forecast_years = list(range(base_year + 1, 2051))

    print("COMPLETE DWELLING DEMAND WORKFLOW (ROBUST + SMOOTH VERSION)")
    print("=" * 70)

    # Step 1: Process OEROK
    print("\nStep 1: Processing OEROK data...")
    try:
        oerok_processed = process_oerok_data_complete(oerok_raw_data)
    except Exception as e:
        print(f"OEROK processing failed: {e}")
        print("Using simple extrapolation without constraints...")
        df_hhfr_complete = simple_hhfr_extrapolation(df_hs_historical, forecast_years)
        return calculate_dwelling_demand_with_scenarios(
            df_hs_historical, population_projections, df_hhfr_complete, base_year,
        )

    # Step 2: Robust HHFR forecasts with optional smoothing
    print(f"\nStep 2: Generating ROBUST HHFR forecasts...")
    print("Using age grouping (8 groups) + regional clustering (5 clusters) + 2011-2021 trends only")
    if apply_smoothing:
        print(f"+ {smoothing_method} smoothing (strength={smoothing_strength})")

    try:
        key_years = [2030, 2040, 2050]
        all_hhfr_forecasts = []

        for year in key_years:
            if year <= max(forecast_years):
                print(f"\n--- Robust forecasting for {year} ---")
                year_forecast = forecast_hhfr_robust(
                    df_hs_historical, oerok_processed, target_year=year,
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
            df_hhfr_complete = interpolate_hhfr_between_years(df_hhfr_constrained, forecast_years)
            print(f"Robust + Smooth HHFR forecasts completed: {df_hhfr_complete.shape}")
        else:
            raise ValueError("No robust forecasts generated")

    except Exception as e:
        print(f"Robust forecasting failed: {e}")
        print("Falling back to simple extrapolation...")
        df_hhfr_complete = simple_hhfr_extrapolation(df_hs_historical, forecast_years)

    # Step 3: Dwelling demand with scenarios
    print("\nStep 3: Calculating dwelling demand with trend and status quo scenarios...")
    results = calculate_dwelling_demand_with_scenarios(
        df_hs_historical, population_projections, df_hhfr_complete, base_year,
    )

    print("Robust + Smooth workflow completed successfully!")
    return results


# =============================================================================
# 15. MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':

    base_year = BASE_YEAR
    demand_year = DEMAND_YEAR

    # -----------------------------------------------------------------
    # 15a. Load historical headship-rate data
    # -----------------------------------------------------------------
    df_hs = pd.read_csv(r'XLSX/df_hs_np_clean.csv')
    df_hs['scenario'] = 'BSL'
    df_hs.head()

    df_hs[df_hs['Year'] == 2011]['Dwellings'].sum()

    # Plot historical HHFR
    print("Plotting data from df_hs_np (your original historical data)")
    # plot_headship_rate_subplots(
    #     df=df_hs,
    #     group_col='AgeGroup',
    #     filter_col='Nationality',
    #     title_prefix='HHFR by 5-Year Age Group (from df_hs_np)',
    #     color_map={'Austria': 'blue', 'Foreign country': 'green'},
    # )

    # -----------------------------------------------------------------
    # 15b. Load population projections
    # -----------------------------------------------------------------
    projected_path = (
        r'PQT/negtest_projected_simEU_negtest_projected_mig_EU'
        r'_2025_2051_h61_20251204_adjfert_adjasfr.parquet'
    )
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

    # -----------------------------------------------------------------
    # 15c. Load OEROK data
    # -----------------------------------------------------------------
    oerok_data = pd.read_excel(r'Data/OEROK/Budget_forecasts/OEROK_haushalt_filtered.xlsx')

    a = process_oerok_data(oerok_data)
    a.head()

    df_struct = a.copy()
    for col in df_struct.columns:
        print(
            f"Column '{col}' (Type: {df_struct[col].dtype}, "
            f"Unique: {df_struct[col].nunique()}): "
            f"Unique values -> {df_struct[col].unique()[:15]}"
        )

    # -----------------------------------------------------------------
    # 15d. Run the complete dwelling-demand workflow
    # -----------------------------------------------------------------
    print("RUN THE COMPLETE WORKING WORKFLOW:")
    print("=" * 50)

    results = complete_dwelling_demand_workflow_robust(
        df_hs_historical=df_hs,
        population_projections=df_proj_nosex,
        oerok_raw_data=oerok_data,
        base_year=base_year,
        forecast_years=list(range(2023, 2051)),
        apply_smoothing=True,
        smoothing_method='rolling',
        smoothing_strength=3,
    )

    # -----------------------------------------------------------------
    # 15e. Inspect results
    # -----------------------------------------------------------------
    df_dw_demand = results['dwelling_demand_detailed']
    df_dw_demand.head()

    # Sanity check
    df_dw_demand[df_dw_demand['Year'] == 2022][['Population', 'scenario']].sum()

    # Check for missing base-year combos
    df_base = df_dw_demand[df_dw_demand['Year'] == base_year].copy()
    df_others = df_dw_demand[df_dw_demand['Year'] != base_year].copy()

    base_combos = set(df_base[['NUTS3', 'Age', 'Sex', 'Nationality']].itertuples(index=False, name=None))
    other_combos = set(df_others[['NUTS3', 'Age', 'Sex', 'Nationality']].itertuples(index=False, name=None))
    missing_combos = other_combos - base_combos

    print(f"Number of base-year combos: {len(base_combos)}")
    print(f"Number of other-year combos: {len(other_combos)}")
    print(f"Number of combos missing from base year: {len(missing_combos)}")

    df_missing = df_others[
        df_others.apply(
            lambda row: (row['NUTS3'], row['Age'], row['Sex'], row['Nationality']) in missing_combos,
            axis=1,
        )
    ]
    print("Missing combos DataFrame:")
    print(df_missing)
    df_missing['Age'].unique()

    # -----------------------------------------------------------------
    # 15f. Load geometries
    # -----------------------------------------------------------------
    eu_gdf = gpd.read_file(r'GEOJSON/NUTS_RG_03M_2024_3035.geojson')

    at_gdf = eu_gdf[eu_gdf['CNTR_CODE'] == 'AT']
    at_gdf = at_gdf[at_gdf['LEVL_CODE'] == 3]
    at_gdf = at_gdf[['NUTS_ID', 'geometry']]
    at_gdf = at_gdf.rename(columns={'NUTS_ID': 'NUTS3'})
    at_gdf.head()

    # -----------------------------------------------------------------
    # 15g. Export
    # -----------------------------------------------------------------
    filename_no_ext = os.path.splitext(os.path.basename(projected_path))[0]
    output_path = f'PQT/dwelling_demand_{filename_no_ext}_clean.parquet'

    df_dw_demand.round(4)
    df_dw_demand.to_parquet(output_path, index=False, compression='gzip')
    print(f"Saved to: {output_path}")