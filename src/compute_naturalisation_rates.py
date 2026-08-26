#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Naturalisation Rate Redistributor

This script calculates naturalisation rates based on historical Statcube data
(aggregated by age bin and NUTS2 region) and broadcasts these rates to
single-age populations at the NUTS3 level. It then converts fixed income-group
shares of naturalisation events into group-specific relative propensities,
preserving the observed national total.

Output: the naturalisation schedule named by config.PATH_NATURALISATION,
which project_migrant_population.py reads back in.
"""

import pandas as pd
import os
import sys

import config

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# File Paths
PATH_NAT_DATA = r'Data/Statcube/naturalisations-2013-2023_nuts2_sex_age_groups.csv'
PATH_POP_DATA = r'XLSX/redistributed_ig_pop_2024.csv'
PATH_OUTPUT   = config.PATH_NATURALISATION

# Age Bin Definitions (Statistik Austria Standard)
NAT_BIN_MAP = {
    'up to 14 years old':   list(range(0, 15)),
    '15 to 17 years old':   list(range(15, 18)),
    '18 to 24 years old':   list(range(18, 25)),
    '25 to 34 years old':   list(range(25, 35)),
    '35 to 44 years old':   list(range(35, 45)),
    '45 to 54 years old':   list(range(45, 55)),
    '55 to 59 years old':   list(range(55, 60)),
    '60 to 64 years old':   list(range(60, 65)),
    '65 to 74 years old':   list(range(65, 75)),
    '75 plus years old':    list(range(75, 101))
}

# NUTS2 Name to Code Mapping
NUTS2_NAME_TO_CODE = {
    'Burgenland': 'AT11', 'Carinthia': 'AT21', 'Lower Austria': 'AT12',
    'Upper Austria': 'AT31', 'Salzburg': 'AT32', 'Styria': 'AT22',
    'Tyrol': 'AT33', 'Vorarlberg': 'AT34', 'Vienna': 'AT13'
}

# Fixed Shares by Income Group
INCOME_SHARES = {
    "High income": 0.30,
    "Upper middle income": 0.35,
    "Lower middle income": 0.05,
    "Low income": 0.30
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_age(age_str):
    """Parses Eurostat/Statistik Austria age strings into integers."""
    if age_str == 'Y_LT1': return 0
    if age_str == 'Y_GE100': return 100
    # specific fix for "112 plus years old" or similar if they appear in this specific column format
    # though the logic below mostly handles standard codes.
    try:
        return int(str(age_str).replace('Y', '').replace('_', ''))
    except ValueError:
        return None

def get_age_to_bin_map():
    """Reverses the NAT_BIN_MAP to map single integer ages to bin labels."""
    age_to_bin = {}
    for label, ages in NAT_BIN_MAP.items():
        for age in ages:
            age_to_bin[age] = label
    return age_to_bin

# ==========================================
# DATA LOADING
# ==========================================

def load_and_clean_naturalisation_data(file_path):
    """
    Loads and cleans the Statcube naturalisation CSV.
    Specific parsing logic required due to metadata headers/footers.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Read blindly first to find structure if needed, but fixed logic provided:
    # Skip first 8 rows based on original script
    df = pd.read_csv(file_path, skiprows=8, encoding='latin-1')
    
    # Locate the footer start (where "Values" == "Symbol")
    if "Symbol" in df['Values'].values:
        symbol_index = df[df['Values'] == "Symbol"].index[0]
        df = df.iloc[:symbol_index]

    # Clean 'Number' column
    df['Number'] = df['Number'].replace('-', '0')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce').fillna(0)

    # Standardize Columns
    df.columns = ['Values', 'TIME_PERIOD', 'age', 'NUTS2', 'sex', 'naturalisation', 'annotations', 'unnamed']

    # Filters
    df = df[df['NUTS2'] != 'Not classifiable <0>']
    df = df[df['NUTS2'] != 'Foreign country']
    df = df[df['age'] != 'Not applicable']
    
    # Type conversion
    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(float)
    df['sex'] = df['sex'].replace({'male': 'M', 'female': 'F'})

    # Drop unused
    df = df.drop(columns=['unnamed', 'annotations', 'Values'], errors='ignore')

    return df

def load_population_data(file_path):
    """Loads the target 2024 population data."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    return df

# ==========================================
# CORE PROCESSING
# ==========================================

def calculate_block_rates(df_nat, df_pop):
    """
    Calculates naturalisation rates per Age-Bin and NUTS2 region.
    Returns a DataFrame of rates broadcasted to single ages.
    """
    age_to_bin_label = get_age_to_bin_map()

    # --- 1. Prepare Population (Denominator) ---
    df_pop['age_num'] = df_pop['age'].apply(parse_age)
    df_pop['NUTS2'] = df_pop['geo'].str[:4]
    df_pop['population'] = df_pop['OBS_VALUE']
    
    # Map single ages to bins
    df_pop['bin_label'] = df_pop['age_num'].map(age_to_bin_label)

    # Aggregate Population by NUTS2, Sex, and BIN
    pop2_binned = (
        df_pop
        .groupby(['NUTS2', 'sex', 'bin_label'], as_index=False)['population']
        .sum()
        .rename(columns={'population': 'pop_bin_total'})
    )

    # --- 2. Prepare Naturalisations (Numerator) ---
    # Map NUTS2 names to codes if necessary
    if df_nat['NUTS2'].str.len().max() > 4:
        df_nat['NUTS2'] = df_nat['NUTS2'].map(NUTS2_NAME_TO_CODE)

    # Filter for relevant years (2022-2024) and Average
    df_nat_avg = (
        df_nat[df_nat['TIME_PERIOD'].isin([2022, 2023, 2024])]
        .groupby(['NUTS2', 'sex', 'age'], as_index=False)['naturalisation']
        .mean()
        .rename(columns={'naturalisation': 'avg_nats_bin', 'age': 'bin_label'})
    )

    # --- 3. Calculate Rates ---
    df_rates = pd.merge(
        df_nat_avg,
        pop2_binned,
        on=['NUTS2', 'sex', 'bin_label'],
        how='inner'
    )
    
    df_rates['block_rate'] = df_rates['avg_nats_bin'] / df_rates['pop_bin_total']
    df_rates['block_rate'] = df_rates['block_rate'].fillna(0)

    # --- 4. Broadcast to Single Ages ---
    expanded_rows = []
    for _, row in df_rates.iterrows():
        bin_lbl = row['bin_label']
        rate = row['block_rate']
        nuts2 = row['NUTS2']
        sex = row['sex']
        
        target_ages = NAT_BIN_MAP.get(bin_lbl, [])
        
        for age in target_ages:
            expanded_rows.append({
                'NUTS2': nuts2,
                'sex': sex,
                'age_num': age,
                'base_nat_rate': rate
            })

    return pd.DataFrame(expanded_rows), df_pop

def apply_income_distribution(df_final):
    """Merges income shares and calculates final weighted rates.

    INCOME_SHARES gives the share of naturalisation EVENTS accruing to each
    income group. Applied directly as a rate multiplier it deflates the national
    total to the population-weighted mean of the shares (~31% of the observed
    figure). Dividing by the group's share of the foreign POPULATION converts the
    event share into a relative propensity, so that

        sum(pop_k * rate_k) == base_nat_rate * pop_total
    """
    income_df = pd.DataFrame([{"IncomeGroup": ig, "share": share}
                              for ig, share in INCOME_SHARES.items()])

    # Merge Shares
    df_final = pd.merge(df_final, income_df, on='IncomeGroup', how='left')

    # Convert event shares into relative propensities
    pop_share = (df_final.groupby('IncomeGroup')['OBS_VALUE'].transform('sum')
                 / df_final['OBS_VALUE'].sum())
    df_final['nat_rate_by_income'] = df_final['base_nat_rate'] * df_final['share'] / pop_share

    # Calibrate so the base-year national total reproduces base_nat_rate exactly
    target = (df_final['OBS_VALUE'] * df_final['base_nat_rate']).sum()
    actual = (df_final['OBS_VALUE'] * df_final['nat_rate_by_income']).sum()
    df_final['nat_rate_by_income'] *= target / actual

    df_final['nat_rate'] = df_final['base_nat_rate'] # Keep total rate for reference

    return df_final

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting naturalisation rate redistribution...")

    # 1. Load Data
    print(f"Loading naturalisation data from: {PATH_NAT_DATA}")
    df_nat = load_and_clean_naturalisation_data(PATH_NAT_DATA)
    
    print(f"Loading population data from: {PATH_POP_DATA}")
    df_pop_raw = load_population_data(PATH_POP_DATA)

    # 2. Calculate Rates (Broadcasting Bin Rates to Single Ages)
    print("Calculating block rates and broadcasting to single ages...")
    df_smooth_rates, df_pop_processed = calculate_block_rates(df_nat, df_pop_raw)

    # 3. Merge Rates onto Detailed NUTS3 Population
    # Start with the detailed population frame
    final_nat = df_pop_processed.copy()

    # Merge the smooth single-age rates
    final_nat = pd.merge(
        final_nat,
        df_smooth_rates,
        on=['NUTS2', 'sex', 'age_num'],
        how='left'
    )

    # Fill gaps with 0 (e.g. if a specific bin had no naturalisations)
    final_nat['base_nat_rate'] = final_nat['base_nat_rate'].fillna(0)

    # 4. Apply Income Group Shares
    print("Applying income group distribution shares...")
    final_nat = apply_income_distribution(final_nat)

    # 5. Export
    print(f"Exporting results to: {PATH_OUTPUT}")
    final_nat.to_csv(PATH_OUTPUT, index=False)
    print("Done.")

if __name__ == "__main__":
    main()