#!/usr/bin/env python
"""
Housing Demand Analysis — Per-Person Headship Rate Model

Pipeline:
  1. Load & clean population data (NUTS3 × age × sex × nationality)
  2. Load & clean dwelling data (housing census)
  3. Load & clean non-private-household (institutional) population
  4. Calculate Group Quarters Rate (GQR) with B-spline interpolation
  5. Calculate headship rates at multiple aggregation levels
  6. Compute Household Formation Rate (HHFR) = (1 − GQR) × headship_rate
  7. Plot HHFR by 5-year age groups
  8. Export results
"""

import math
from io import BytesIO, StringIO

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.interpolate import CubicSpline, splev, splrep
from scipy.stats import shapiro, ttest_rel, wilcoxon


# =============================================================================
# 1. DATA LOADING & CLEANING FUNCTIONS
# =============================================================================

def load_and_clean_population_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean population data from a StatCube CSV export.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Cleaned population DataFrame with columns:
        [Year, NUTS3, Age, Sex, Nationality, Population]
    """
    df = pd.read_csv(file_path, skiprows=8, encoding='latin-1')

    # Truncate at the metadata footer
    symbol_index = df[df['Values'] == "Symbol"].index[0]
    df = df.iloc[:symbol_index]

    # Clean the numeric column
    df['Number'] = df['Number'].replace('-', '0')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce').fillna(0)

    df.columns = [
        'Values', 'Year', 'NUTS3', 'Sex', 'Age',
        'Nationality', 'Population', 'Annotations', 'Unnamed',
    ]

    # Filter out unclassifiable regions
    df = df[df['NUTS3'] != 'Not classifiable <0>']

    df['Year'] = df['Year'].astype(float)

    # Extract NUTS3 code from angle brackets
    df['NUTS3'] = df['NUTS3'].str.extract(r'<(.*?)>')

    # Harmonise age values
    df = df[df['Age'] != 'Not applicable']
    df['Age'] = df['Age'].replace({
        '100 plus years old': 100,
        'under 1 year old': 0,
        '1 year old': 1,
    })
    df['Age'] = df['Age'].astype(str).str.replace(' years old', '')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Standardise sex labels
    df['Sex'] = df['Sex'].map({'male': 'M', 'female': 'F'})

    df = df.drop(columns=['Unnamed', 'Annotations', 'Values'], errors='ignore')
    return df


def load_and_clean_nonprivate_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean institutional (non-private) household population data.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Cleaned DataFrame with columns:
        [Year, NUTS3, Age, Sex, Nationality, NPPopulation]
    """
    df = pd.read_csv(file_path, skiprows=8, encoding='latin-1')

    symbol_index = df[df['Counting'] == "Symbol"].index[0]
    df = df.iloc[:symbol_index]

    df['Number'] = df['Number'].replace('-', '0')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce').fillna(0)

    df.columns = [
        'Values', 'Year', 'NUTS3', 'Age', 'Sex', 'Type',
        'Nationality', 'NPPopulation', 'Annotations', 'Unnamed',
    ]

    df = df[df['NUTS3'] != 'Not classifiable <0>']
    df['Year'] = df['Year'].astype(float)

    # The 2011 census data uses 2012 for institutional households; remap.
    df.loc[df['Year'] == 2012, 'Year'] = 2011

    df['NUTS3'] = df['NUTS3'].str.extract(r'<(.*?)>')
    df = df[df['Age'] != 'Not applicable']

    df['Sex'] = df['Sex'].map({'Male': 'M', 'Female': 'F'})
    df['Nationality'] = df['Nationality'].replace({
        'Not Austria (incl. Stateless/Unsettled/Unknown)': 'Foreign country',
    })

    df = df.drop(columns=['Unnamed', 'Annotations', 'Values', 'Type'], errors='ignore')
    return df


# =============================================================================
# 2. AGE-GROUP MAPPING HELPERS
# =============================================================================

def map_age_to_group(age) -> str:
    """Map an integer age to a 5-year age-group string used by StatCube."""
    if pd.isna(age):
        return np.nan
    try:
        age = int(age)
    except (ValueError, TypeError):
        return np.nan

    if age < 0:
        return np.nan

    boundaries = list(range(0, 100, 5))  # [0, 5, 10, ..., 95]
    for low in boundaries:
        high = low + 4
        if low <= age <= high:
            return f'{low} to {high} years'
    if age >= 100:
        return '100 years and over'
    return np.nan


def get_age_midpoint(age_group_str) -> float:
    """Return the numerical midpoint for a StatCube age-group string."""
    if pd.isna(age_group_str) or not isinstance(age_group_str, str):
        return np.nan
    try:
        if 'to' in age_group_str:
            parts = age_group_str.replace(' years', '').split(' to ')
            if len(parts) != 2:
                return np.nan
            return (int(parts[0]) + int(parts[1])) / 2
        elif 'and over' in age_group_str:
            return int(age_group_str.replace(' years and over', ''))
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan


def categorize_age(age: int) -> str:
    """Categorise an individual age into the broad dwelling-census age groups."""
    if age < 15:
        return 'Under 15'
    elif age <= 29:
        return '15 to 29 years'
    elif age <= 49:
        return '30 to 49 years'
    elif age <= 64:
        return '50 to 64 years'
    elif age <= 84:
        return '65 to 84 years'
    else:
        return '85 years and over'


def create_age_group_mapping() -> dict:
    """Create a mapping from integer age (0–100) to 5-year age-group labels."""
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


# =============================================================================
# 3. GQR CALCULATION WITH SPLINE INTERPOLATION
# =============================================================================

def calculate_gqr_by_age_and_spline(
    df_np: pd.DataFrame,
    df_pop: pd.DataFrame,
    spline_ages: np.ndarray = np.arange(0, 101),
) -> pd.DataFrame:
    """
    Calculate Group Quarters Rate (GQR) per cohort with B-spline interpolation.

    Computes age-group-level GQR from institutional-population and total-population
    data, then interpolates smoothly across individual ages 0–100.

    Args:
        df_np:  Non-private household population (columns incl. NPPopulation).
        df_pop: Total population (columns incl. Population, Age as int).
        spline_ages: Array of integer ages for spline evaluation.

    Returns:
        DataFrame with columns:
        [Year, NUTS3, Sex, Nationality, Age, GQR, GQPop, Population]
    """
    df_np = df_np.copy()
    df_pop = df_pop.copy()

    # Ensure integer years
    df_np['Year'] = pd.to_numeric(df_np['Year'], errors='coerce').astype('Int64')
    df_pop['Year'] = pd.to_numeric(df_pop['Year'], errors='coerce').astype('Int64')
    df_np = df_np.dropna(subset=['Year'])
    df_pop = df_pop.dropna(subset=['Year'])

    # Standardise age columns to age-group strings
    df_pop['Age_Group'] = df_pop['Age'].apply(map_age_to_group)
    df_pop = df_pop.dropna(subset=['Age_Group'])

    df_np = df_np.rename(columns={'Age': 'Age_Group'})
    df_np = df_np.dropna(subset=['Age_Group'])

    group_keys = ['Year', 'NUTS3', 'Age_Group', 'Sex', 'Nationality']

    # Aggregate GQ population and total population
    gq_pop_summary = (
        df_np.groupby(group_keys, dropna=False)['NPPopulation']
        .sum().reset_index()
        .rename(columns={'NPPopulation': 'GQPop'})
    )
    tp_pop_summary = (
        df_pop.groupby(group_keys, dropna=False)['Population']
        .sum().reset_index()
        .rename(columns={'Population': 'TPop'})
    )

    df_gqr_by_cohort = pd.merge(
        gq_pop_summary, tp_pop_summary, on=group_keys, how='outer'
    ).fillna(0)

    df_gqr_by_cohort['GQR_by_Age'] = np.clip(
        np.where(df_gqr_by_cohort['TPop'] > 0,
                 df_gqr_by_cohort['GQPop'] / df_gqr_by_cohort['TPop'], 0),
        0, 1,
    )

    df_gqr_by_cohort['Age_Midpoint'] = df_gqr_by_cohort['Age_Group'].apply(get_age_midpoint)
    df_gqr_by_cohort = df_gqr_by_cohort.dropna(subset=['Age_Midpoint'])

    # Lookup for exact population by individual age
    pop_lookup_by_age = (
        df_pop.set_index(['Year', 'NUTS3', 'Sex', 'Nationality', 'Age'])['Population']
        .to_dict()
    )

    # B-spline interpolation per cohort
    splined_results = []
    spline_iter_keys = ['Year', 'NUTS3', 'Sex', 'Nationality']

    for cohort_keys, group_df in df_gqr_by_cohort.groupby(spline_iter_keys, dropna=False):
        year, nuts3, sex, nationality = cohort_keys
        group_df_sorted = group_df.sort_values(by='Age_Midpoint')
        spline_data = group_df_sorted[group_df_sorted['TPop'] > 0]

        if len(spline_data) == 0:
            continue

        x_values = spline_data['Age_Midpoint'].values
        y_values = spline_data['GQR_by_Age'].values

        # Deduplicate x values
        _, unique_idx = np.unique(x_values, return_index=True)
        unique_idx = np.sort(unique_idx)
        x_unique = x_values[unique_idx]
        y_unique = y_values[unique_idx]

        k_degree = min(3, len(x_unique) - 1)

        if k_degree < 1:
            if len(x_unique) == 1:
                splined_gqrs = np.clip(np.full(len(spline_ages), y_unique[0]), 0, 1)
            else:
                splined_gqrs = np.zeros(len(spline_ages))
        else:
            try:
                sort_idx = np.argsort(x_unique)
                tck = splrep(x_unique[sort_idx], y_unique[sort_idx], k=k_degree)
                splined_gqrs = np.clip(splev(spline_ages, tck), 0, 1)
            except ValueError as e:
                print(
                    f"Warning: Spline interpolation failed for Year {year}, "
                    f"NUTS3 {nuts3}, Sex {sex}, Nationality {nationality} "
                    f"due to '{e}'."
                )
                print("Falling back to linear interpolation if enough points, otherwise zeros/constant.")
                if len(x_unique) >= 2:
                    splined_gqrs = np.interp(spline_ages, x_unique, y_unique)
                elif len(x_unique) == 1:
                    splined_gqrs = np.full(len(spline_ages), y_unique[0])
                else:
                    splined_gqrs = np.zeros(len(spline_ages))

        for i, age in enumerate(spline_ages):
            pop_key = (year, nuts3, sex, nationality, age)
            exact_pop = pop_lookup_by_age.get(pop_key, 0)
            gqr_val = splined_gqrs[i]
            splined_results.append({
                'Year': year,
                'NUTS3': nuts3,
                'Sex': sex,
                'Nationality': nationality,
                'Age': age,
                'GQR': gqr_val,
                'GQPop': exact_pop * gqr_val,
                'Population': exact_pop,
            })

    if not splined_results:
        return pd.DataFrame(
            columns=['Year', 'NUTS3', 'Sex', 'Nationality', 'Age', 'GQR', 'GQPop', 'Population']
        )
    return pd.DataFrame(splined_results)


# =============================================================================
# 4. HEADSHIP RATE CALCULATION & INTERPOLATION
# =============================================================================

def aggregate_and_calculate_headship_rate(
    df_dw: pd.DataFrame,
    df_pop: pd.DataFrame,
    grouping_vars: list[str],
) -> pd.DataFrame:
    """
    Aggregate dwelling & population data and compute the headship rate.

    Args:
        df_dw:  Dwelling counts with an 'Age_group' column.
        df_pop: Private-household population with an 'Age_group' column.
        grouping_vars: Columns to group by.

    Returns:
        Merged DataFrame with headship_rate (%).
    """
    df_dw_agg = df_dw.groupby(grouping_vars, as_index=False)['Dwellings'].sum()
    df_pop_agg = df_pop.groupby(grouping_vars, as_index=False)['HHPopulation'].sum()

    df_hs = pd.merge(df_dw_agg, df_pop_agg, on=grouping_vars, how='inner')
    df_hs['headship_rate'] = np.where(
        df_hs['HHPopulation'] == 0, 0,
        (df_hs['Dwellings'] / df_hs['HHPopulation']) * 100,
    )
    df_hs['headship_rate'] = df_hs['headship_rate'].fillna(0)
    return df_hs


def interpolate_headship_rate(
    df_hs: pd.DataFrame,
    full_age_range: np.ndarray = np.arange(15, 101),
) -> pd.DataFrame:
    """
    Cubic-spline interpolation of headship rate over individual ages.

    Uses age-group midpoints as knots and evaluates the spline at each
    integer age in *full_age_range*.
    """
    age_mapping = {
        "15 to 29 years": (15, 29),
        "30 to 49 years": (30, 49),
        "50 to 64 years": (50, 64),
        "65 to 84 years": (65, 84),
        "85 years and over": (85, 100),
        "Under 15": (0, 14),
    }

    results = []
    grouping_vars = [
        col for col in df_hs.columns
        if col not in ['Age_group', 'headship_rate', 'Dwellings', 'HHPopulation']
    ]

    for keys, group in df_hs.groupby(grouping_vars):
        midpoints, rates = [], []
        for _, row in group.iterrows():
            age_group = str(row['Age_group'])
            if age_group in age_mapping:
                low, high = age_mapping[age_group]
                midpoints.append(np.mean([low, high]))
                rates.append(row['headship_rate'])

        if len(midpoints) >= 2:
            cs = CubicSpline(midpoints, rates, bc_type='natural')
            interp_rates = np.clip(cs(full_age_range), 0, 100)
        elif len(midpoints) == 1:
            interp_rates = np.full_like(full_age_range, rates[0], dtype=float)
        else:
            interp_rates = np.full_like(full_age_range, 0, dtype=float)

        if isinstance(keys, tuple):
            key_dict = dict(zip(grouping_vars, keys))
        elif len(grouping_vars) == 1:
            key_dict = {grouping_vars[0]: keys}
        else:
            key_dict = {}

        for age, rate in zip(full_age_range, interp_rates):
            row_result = key_dict.copy()
            row_result['Age'] = age
            row_result['headship_rate'] = rate
            results.append(row_result)

    return pd.DataFrame(results)


# =============================================================================
# 5. HHFR MERGE HELPER
# =============================================================================

def calculate_hhfr_and_dwellings(
    df_base: pd.DataFrame,
    df_rate: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge a headship-rate table onto the base population/GQR table and
    compute HHFR and dwelling counts.

    HHFR  = (1 − GQR) × headship_rate
    Dwellings = Population × HHFR / 100

    Args:
        df_base: Granular population/GQR table.
        df_rate: Aggregated headship-rate table.

    Returns:
        Merged DataFrame with headship_rate, HHFR, and Dwellings columns.
    """
    value_cols = {'headship_rate', 'Dwellings', 'HHPopulation'}
    merge_keys = list(set(df_rate.columns) - value_cols)

    cols_to_merge = [c for c in merge_keys + ['headship_rate'] if c in df_rate.columns]
    if 'headship_rate' not in cols_to_merge:
        print("Warning: 'headship_rate' column not found in df_rate. Returning base df.")
        return df_base.copy()

    df_merged = df_base.merge(df_rate[cols_to_merge], on=merge_keys, how='left')

    # Ages < 15 have no headship → fill with 0
    df_merged.loc[
        (df_merged['Age'] < 15) & (df_merged['headship_rate'].isna()),
        'headship_rate',
    ] = 0
    df_merged['headship_rate'] = df_merged['headship_rate'].fillna(0)

    df_merged['HHFR'] = (1 - df_merged['GQR']) * df_merged['headship_rate']
    df_merged['Dwellings'] = df_merged['Population'] * (df_merged['HHFR'] / 100)
    return df_merged


# =============================================================================
# 6. REPORTING HELPERS
# =============================================================================

def report_population_by_nationality_year(
    dataframe: pd.DataFrame, population_col: str, title: str,
) -> None:
    """Print population totals grouped by year and nationality."""
    pop_by_nat_year = dataframe.groupby(['Year', 'Nationality'])[population_col].sum()
    print(f"\n--- {title} ---")
    for (year, nationality), count in pop_by_nat_year.items():
        print(f" - Year {year}, Nationality {nationality}: {count} people")


# =============================================================================
# 7. PLOTTING
# =============================================================================

def plot_headship_rate_subplots(df, group_col, filter_col, title_prefix, color_map):
    """
    Plot HHFR time-series by *group_col* facets, coloured by *filter_col*.

    Internally maps individual ages to 5-year age groups for the subplot grid.
    """
    print("Applying 5-year 'AgeGroup' mapping for plotting...")
    age_group_map = create_age_group_mapping()
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
                label=str(fval),
                color=color_map.get(fval, 'gray'),
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
# 8. MAIN PIPELINE
# =============================================================================

if __name__ == '__main__':

    # -----------------------------------------------------------------
    # 8a. Load population data
    # -----------------------------------------------------------------
    years = [2022, 2021, 2011]
    age_groups = ['0-74', '75-100']
    sexes = ['M', 'F']
    nats = ['AT', 'F']

    pop_file_template = "Data/Statcube/population-{year}_nuts3_age{age}_{Sex}_{nat}.csv"

    df_pop = pd.DataFrame()
    for year in years:
        for age in age_groups:
            for sex in sexes:
                for nat in nats:
                    fpath = pop_file_template.format(year=year, age=age, Sex=sex, nat=nat)
                    df_pop = pd.concat(
                        [df_pop, load_and_clean_population_data(fpath)],
                        ignore_index=True,
                    )

    df_pop = df_pop.groupby(
        ['Year', 'NUTS3', 'Age', 'Sex', 'Nationality'], as_index=False
    )['Population'].sum()

    df_pop.head()
    print('2022 population after cleaning dataset ', df_pop[df_pop['Year'] == 2022]['Population'].sum())

    df_pop.to_csv('Data/Statcube/cleaned_population_data_housing.csv', index=False)

    # -----------------------------------------------------------------
    # 8b. Load dwelling data
    # -----------------------------------------------------------------
    df_dw = pd.read_csv(
        r'Data/Statcube/housing_census_dwellings-2011-2021-2022_nuts3_age_Sex_ATF.csv',
        skiprows=8, encoding='latin-1',
    )
    symbol_index = df_dw[df_dw['Counting'] == "Symbol"].index[0]
    df_dw = df_dw.iloc[:symbol_index]
    df_dw.tail()

    df_dw['Number'] = df_dw['Number'].replace('-', '0')
    df_dw['Number'] = pd.to_numeric(df_dw['Number'], errors='coerce').fillna(0)
    print('Dwellings before cleaning dataset ', df_dw[df_dw['Year'] == '2011']['Number'].sum())

    df_dw.columns = [
        'Counting', 'Year', 'NUTS3', 'Age_group', 'Sex',
        'Nationality', 'Dwellings', 'Annotations', 'Unnamed',
    ]
    df_dw['Year'] = df_dw['Year'].astype(float)
    df_dw['NUTS3'] = df_dw['NUTS3'].str.extract(r'<(.*?)>')

    # Dwellings without usual residents have no household reference person
    df_dw = df_dw[df_dw['Age_group'] != 'Not applicable']
    df_dw['Nationality'] = df_dw['Nationality'].replace({
        'Not Austria (incl. Stateless/Unsettled/Unknown)': 'Foreign country',
    })
    df_dw['Sex'] = df_dw['Sex'].map({'Male': 'M', 'Female': 'F'})

    df_dw = df_dw.groupby(
        ['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'], as_index=False
    )['Dwellings'].sum()
    df_dw = df_dw.drop(columns=['Unnamed', 'Annotations', 'Counting'], errors='ignore')
    df_dw.head()

    df_dw['Age_group'].unique()

    print(df_dw[df_dw['Nationality'] == 'Not applicable']['Dwellings'].sum())
    df_dw = df_dw[df_dw['Nationality'] != 'Not applicable']

    # Report dwellings by nationality and year
    dwellings_by_nat_year = df_dw.groupby(['Year', 'Nationality'])['Dwellings'].sum()
    for (year, nationality), count in dwellings_by_nat_year.items():
        print(f" - Year {year}, Nationality {nationality}: {count} dwellings")

    # -----------------------------------------------------------------
    # 8c. Load non-private (institutional) household population
    # -----------------------------------------------------------------
    np_years = [2012, 2022, 2021]
    np_file_template = "Data/Statcube/institutional_households-{year}_nuts3_agegroups_sex_{nat}.csv"

    df_np = pd.DataFrame()
    for year in np_years:
        for nat in nats:
            fpath = np_file_template.format(year=year, nat=nat)
            df_np = pd.concat(
                [df_np, load_and_clean_nonprivate_data(fpath)],
                ignore_index=True,
            )

    df_np = df_np.groupby(
        ['Year', 'NUTS3', 'Age', 'Sex', 'Nationality'], as_index=False
    )['NPPopulation'].sum()
    df_np.tail()

    np_pop_by_nat_year = df_np.groupby(['Year', 'Nationality'])['NPPopulation'].sum()
    for (year, nationality), count in np_pop_by_nat_year.items():
        print(f" - Year {year}, Nationality {nationality}: {count} peoople in non-private households")

    # -----------------------------------------------------------------
    # 8d. Calculate splined GQR
    # -----------------------------------------------------------------
    df_splined_gqr = calculate_gqr_by_age_and_spline(df_np, df_pop)
    print(df_splined_gqr)

    df_splined_gqr[df_splined_gqr['Year'] == 2022]['Population'].sum()

    # -----------------------------------------------------------------
    # 8e. Prepare household population (total − institutional)
    # -----------------------------------------------------------------
    df_pop_agroup = df_pop.copy()
    df_pop_agroup['Age_group'] = df_pop_agroup['Age'].apply(categorize_age)
    df_pop_agroup = df_pop_agroup.groupby(
        ['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'], as_index=False
    )['Population'].sum()
    print("Aggregated df_pop_agroup head:")
    print(df_pop_agroup.head())

    df_np_agroup = df_np.copy()
    age_bins = [0, 15, 30, 50, 65, 85, np.inf]
    age_labels = [
        'Under 15', '15 to 29 years', '30 to 49 years',
        '50 to 64 years', '65 to 84 years', '85 years and over',
    ]
    df_np_agroup['Age_group'] = pd.cut(
        df_np_agroup['Age'].apply(
            lambda x: int(str(x).split(' ')[0]) if str(x).split(' ')[0].isdigit() else 0
        ),
        bins=age_bins, labels=age_labels, right=False, include_lowest=True,
    )
    df_np_agroup = df_np_agroup.groupby(
        ['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'], as_index=False
    )['NPPopulation'].sum()
    print("\nAggregated df_np_agroup head:")
    print(df_np_agroup.head())

    # Merge population and non-private population
    df_hhpop = pd.merge(
        df_pop_agroup, df_np_agroup,
        on=['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'],
        how='outer', suffixes=('_pop', '_np'),
    ).rename(columns={'Population': 'Total_population', 'NPPopulation': 'NPPopulation'})

    df_hhpop['NPPopulation'] = df_hhpop['NPPopulation'].fillna(0)
    df_hhpop['Total_population'] = df_hhpop['Total_population'].fillna(0)

    # For 2011 census the population already excludes institutional HHs
    df_hhpop['HHPopulation'] = df_hhpop.apply(
        lambda row: row['Total_population']
        if row['Year'] == 2011
        else row['Total_population'] - row['NPPopulation'],
        axis=1,
    )
    print("\ndf_hhpop head after merging and calculation:")
    print(df_hhpop.head())

    report_population_by_nationality_year(
        df_hhpop, 'HHPopulation',
        'Population by Nationality and Year (After Non-Private Households Adjustment)',
    )
    report_population_by_nationality_year(
        df_hhpop, 'Total_population',
        'Total Population by Nationality and Year (Before Non-Private Households Adjustment)',
    )

    # -----------------------------------------------------------------
    # 8f. Headship rates at multiple aggregation levels
    # -----------------------------------------------------------------
    aggregations = {
        'df_hs':           ['Year', 'Age_group', 'NUTS3', 'Sex', 'Nationality'],
        'df_hs_age_cob':   ['Year', 'Age_group', 'Nationality'],
        'df_hs_nuts3_cob': ['Year', 'NUTS3', 'Nationality'],
        'df_hs_sex_cob':   ['Year', 'Sex', 'Nationality'],
        'df_hs_nuts3_age': ['Year', 'NUTS3', 'Age_group'],
        'df_hs_age':       ['Year', 'Age_group'],
        'df_hs_nuts3':     ['Year', 'NUTS3'],
        'df_hs_cob':       ['Year', 'Nationality'],
        'df_hs_Sex':       ['Year', 'Sex'],
    }

    results = {}
    for key, grouping_vars in aggregations.items():
        print(f"\n--- Processing aggregation: {key} ---")
        df_agg = aggregate_and_calculate_headship_rate(
            df_dw=df_dw, df_pop=df_hhpop, grouping_vars=grouping_vars,
        )
        if 'Age_group' in grouping_vars:
            results[key] = interpolate_headship_rate(df_agg)
        else:
            results[key] = df_agg
        print(f"Sample of {key} results:")
        print(results[key].head())

    df_hs = results['df_hs']
    df_hs_age_cob = results['df_hs_age_cob']
    df_hs_nuts3_cob = results['df_hs_nuts3_cob']
    df_hs_nuts3_age = results['df_hs_nuts3_age']
    df_hs_sex_cob = results['df_hs_sex_cob']
    df_hs_age = results['df_hs_age']
    df_hs_nuts3 = results['df_hs_nuts3']
    df_hs_cob = results['df_hs_cob']
    df_hs_Sex = results['df_hs_Sex']
    print("\nAll aggregations and interpolations complete.")

    # -----------------------------------------------------------------
    # 8g. Merge population with headship rates → full per-person table
    # -----------------------------------------------------------------
    df_hs_np = df_splined_gqr[
        ['Year', 'NUTS3', 'Nationality', 'Age', 'Sex', 'Population', 'GQPop', 'GQR']
    ].merge(
        df_hs[['Year', 'NUTS3', 'Nationality', 'Age', 'Sex', 'headship_rate']],
        on=['Year', 'NUTS3', 'Nationality', 'Age', 'Sex'],
        how='left',
    )
    df_hs_np.loc[
        (df_hs_np['Age'] < 15) & (df_hs_np['headship_rate'].isna()),
        'headship_rate',
    ] = 0
    df_hs_np['Dwellings'] = (
        df_hs_np['Population']
        * (1 - df_hs_np['GQR'])
        * (df_hs_np['headship_rate'] / 100)
    )

    # -----------------------------------------------------------------
    # 8h. HHFR variants at different aggregation levels
    # -----------------------------------------------------------------

    # --- 2-Variable ---
    df_hs_age_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_age_cob'],
    )
    df_hs_nuts3_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_nuts3_cob'],
    )
    df_hs_sex_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_sex_cob'],
    )
    df_hs_nuts3_age_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_nuts3_age'],
    )

    # --- 1-Variable ---
    df_hs_age_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_age'],
    )
    df_hs_nuts3_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_nuts3'],
    )
    df_hs_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_cob'],
    )

    # -----------------------------------------------------------------
    # 8i. Household Formation Rate on the main table
    # -----------------------------------------------------------------
    df_hs_np['HHFR'] = (1 - df_hs_np['GQR']) * df_hs_np['headship_rate']
    df_hs_np.head()

    # -----------------------------------------------------------------
    # 8j. Plot HHFR by 5-year age group
    # -----------------------------------------------------------------
    print("Plotting data from df_hs_np (your original historical data)")
    plot_headship_rate_subplots(
        df=df_hs_np,
        group_col='AgeGroup',
        filter_col='Nationality',
        title_prefix='HHFR by 5-Year Age Group (from df_hs_np)',
        color_map={'Austria': 'blue', 'Foreign country': 'green'},
    )

    # -----------------------------------------------------------------
    # 8k. Export
    # -----------------------------------------------------------------
    df_hs_np.to_csv(r'XLSX/df_hs_np_clean.csv', index=False)