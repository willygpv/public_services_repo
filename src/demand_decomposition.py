"""National demand by sector and nationality: shares, representation index, cumulative growth, and the volume, ageing and behaviour effects."""

import pandas as pd
import numpy as np
import os

import config


SCENARIO_FOCUS = 'Trend'

PATHS = {
    'edu': config.PATH_EDUCATION_DEMAND,
    'hous': config.PATH_HOUSING_DEMAND,
    'health': config.PATH_HEALTHCARE_DEMAND,
    'pop': config.PATH_PROJECTION,
    'output_csv': config.PATH_NATIONAL_METRICS,
}

METRIC_MAP = {
    'M1_Absolute_Index': 'M1_Absolute_Index',
    'M2_Growth_Index': 'M2_Growth_Index',
    'M3_Market_Share': 'M3_Share_of_Total_Demand',
    'M4_Intensity_Index': 'M4_Per_Capita_Demand_Index',
    'M6_RI': 'M6_Proportional_Demand_Index',
    'M7_Volume_Contribution': 'M7_Population_Volume_Effect',
    'M8_Structure_Contribution': 'M8_Demographic_Composition_Effect',
    'M9_Behavior_Contribution': 'M9_Policy_Behavior_Effect',
    'M10_Demand_Index': 'M10_Cumulative_Demand_Growth'
}

os.makedirs('CSV', exist_ok=True)
os.makedirs(os.path.dirname(PATHS['output_csv']), exist_ok=True)

def load_and_calculate_metrics():
    print("Processing...")
    print("   Loading raw files...")
    edu_df = pd.read_parquet(PATHS['edu'])
    hous_df = pd.read_parquet(PATHS['hous'])
    health_df = pd.read_parquet(PATHS['health'])

    try:
        pop_master_df = pd.read_parquet(PATHS['pop'])
        if len(pop_master_df.columns) <= 2: raise ValueError("Likely wrong delimiter")
    except:
        print("   Standard CSV read failed, trying with semi-colon separator...")
        pop_master_df = pd.read_csv(PATHS['pop'], sep=';')

    print("   Processing demographics...")
    pop_master_df = pop_master_df.rename(columns={'TIME_PERIOD': 'Year', 'OBS_VALUE': 'Population'})

    def age_sort_key(age):
        if age == 'Y_LT1': return 0
        if age == 'Y_GE100': return 100
        try: return int(age.replace('Y', ''))
        except: return -1

    pop_master_df['Age_Clean'] = pop_master_df['age'].apply(age_sort_key)
    pop_master_df = pop_master_df[pop_master_df['Age_Clean'] >= 0].copy()

    def get_base_scenario(scen_str):
        if '--' in str(scen_str): return str(scen_str).split('--')[0]
        return str(scen_str)

    pop_scen_total = pop_master_df.groupby(['Year', 'Nationality', 'scenario'])['Population'].sum().reset_index() \
        .rename(columns={'Population': 'Master_Pop_Total', 'scenario': 'base_scenario'})

    pop_work = pop_master_df[(pop_master_df['Age_Clean'] >= 20) & (pop_master_df['Age_Clean'] <= 64)]
    pop_scen_work = pop_work.groupby(['Year', 'Nationality', 'scenario'])['Population'].sum().reset_index() \
        .rename(columns={'Population': 'Master_Pop_Workforce', 'scenario': 'base_scenario'})

    pop_global = pop_master_df.groupby(['Year', 'scenario'])['Population'].sum().reset_index() \
        .rename(columns={'Population': 'Global_Pop', 'scenario': 'base_scenario'})

    def process_service_scenarios(df, service_name, value_col, year_col='Year', exclude_patterns=None):
        if year_col != 'Year': df = df.rename(columns={year_col: 'Year'})
        df = df[df['Year'] >= 2025].copy()
        if exclude_patterns:
            for pattern in exclude_patterns:
                df = df[~df['scenario'].str.startswith(pattern)]
        df_agg = df.groupby(['Year', 'Nationality', 'scenario'], observed=False)[value_col].sum().reset_index()
        df_agg['base_scenario'] = df_agg['scenario'].apply(get_base_scenario)
        df_merged = pd.merge(df_agg, pop_scen_total, on=['Year', 'Nationality', 'base_scenario'], how='left')
        df_merged = pd.merge(df_merged, pop_scen_work, on=['Year', 'Nationality', 'base_scenario'], how='left')
        df_merged = pd.merge(df_merged, pop_global, on=['Year', 'base_scenario'], how='left')
        df_merged['Service'] = service_name
        df_merged = df_merged.rename(columns={value_col: 'Demand'})
        return df_merged

    print("   Merging datasets (Scenario Level)...")
    edu_processed = process_service_scenarios(edu_df, 'Education', 'Demand_teachers', exclude_patterns=['historic', 'extrapolated'])
    hous_processed = process_service_scenarios(hous_df, 'Housing', 'Dwellings')
    health_processed = process_service_scenarios(health_df, 'Healthcare', 'Bed_Days_Mean', year_col='year')

    all_scenarios = pd.concat([edu_processed, hous_processed, health_processed], ignore_index=True)
    all_scenarios = all_scenarios.dropna(subset=['Master_Pop_Total', 'Global_Pop'])

    print("   Calculating metrics per scenario...")
    all_scenarios = all_scenarios.sort_values(['Service', 'Nationality', 'scenario', 'Year'])
    all_scenarios['Global_Demand'] = all_scenarios.groupby(['Year', 'Service', 'scenario'])['Demand'].transform('sum')
    all_scenarios['Intensity_Total'] = all_scenarios['Demand'] / all_scenarios['Master_Pop_Total']

    all_scenarios['I0'] = all_scenarios.groupby(['Service', 'Nationality', 'scenario'])['Intensity_Total'].transform('first')
    all_scenarios['D0'] = all_scenarios.groupby(['Service', 'Nationality', 'scenario'])['Demand'].transform('first')
    all_scenarios['P0'] = all_scenarios.groupby(['Service', 'Nationality', 'scenario'])['Master_Pop_Total'].transform('first')
    all_scenarios['D0'] = all_scenarios['D0'].replace(0, np.nan)

    all_scenarios['M3_Market_Share'] = (all_scenarios['Demand'] / all_scenarios['Global_Demand']) * 100
    all_scenarios['M4_Intensity_Index'] = (all_scenarios['Intensity_Total'] / all_scenarios['I0']) * 100
    all_scenarios['M6_RI'] = (all_scenarios['Demand'] / all_scenarios['Global_Demand']) / \
                             (all_scenarios['Master_Pop_Total'] / all_scenarios['Global_Pop'])

    all_scenarios['M10_Demand_Index'] = ((all_scenarios['Demand'] / all_scenarios['D0']) - 1) * 100

    print("   Calculating 3-component decomposition...")
    sq_mask = all_scenarios['scenario'].str.contains('statusquo|sq', case=False, regex=True)
    sq_df = all_scenarios[sq_mask].copy()

    join_cols = ['Year', 'Nationality', 'Service', 'base_scenario']
    sq_cols = ['Demand', 'Master_Pop_Total', 'Intensity_Total', 'P0', 'I0'] + join_cols

    if sq_df.empty:
        all_scenarios['Effect_Behavior_Abs'] = 0
        all_scenarios['Effect_Volume_Abs'] = (all_scenarios['Master_Pop_Total'] - all_scenarios['P0']) * ((all_scenarios['I0'] + all_scenarios['Intensity_Total']) / 2)
        all_scenarios['Effect_Structure_Abs'] = (all_scenarios['Intensity_Total'] - all_scenarios['I0']) * ((all_scenarios['P0'] + all_scenarios['Master_Pop_Total']) / 2)
    else:
        all_scenarios = pd.merge(all_scenarios, sq_df[sq_cols], on=join_cols, suffixes=('', '_SQ'), how='left')

        for c in ['Demand', 'Master_Pop_Total', 'Intensity_Total', 'P0', 'I0']:
            if f'{c}_SQ' in all_scenarios.columns:
                all_scenarios[f'{c}_SQ'] = all_scenarios[f'{c}_SQ'].fillna(all_scenarios[c])

        all_scenarios['Effect_Volume_Abs'] = (
            (all_scenarios['Master_Pop_Total_SQ'] - all_scenarios['P0_SQ']) * ((all_scenarios['I0_SQ'] + all_scenarios['Intensity_Total_SQ']) / 2)
        )

        all_scenarios['Effect_Structure_Abs'] = (
            (all_scenarios['Intensity_Total_SQ'] - all_scenarios['I0_SQ']) * ((all_scenarios['P0_SQ'] + all_scenarios['Master_Pop_Total_SQ']) / 2)
        )

        raw_behavior = all_scenarios['Demand'] - all_scenarios['Demand_SQ']

        gap_map = all_scenarios[all_scenarios['Year'] == 2025].set_index(['Service', 'Nationality', 'scenario'])['Demand'] - \
                  all_scenarios[all_scenarios['Year'] == 2025].set_index(['Service', 'Nationality', 'scenario'])['Demand_SQ']

        all_scenarios = all_scenarios.join(gap_map.rename('legacy_gap'), on=['Service', 'Nationality', 'scenario'])
        all_scenarios['legacy_gap'] = all_scenarios['legacy_gap'].fillna(0)

        all_scenarios['Effect_Behavior_Abs'] = raw_behavior - all_scenarios['legacy_gap']

    all_scenarios['M7_Volume_Contribution'] = (all_scenarios['Effect_Volume_Abs'] / all_scenarios['D0']) * 100
    all_scenarios['M8_Structure_Contribution'] = (all_scenarios['Effect_Structure_Abs'] / all_scenarios['D0']) * 100
    all_scenarios['M9_Behavior_Contribution'] = (all_scenarios['Effect_Behavior_Abs'] / all_scenarios['D0']) * 100

    print(f"   Aggregating statistics (Mode: {SCENARIO_FOCUS})...")
    subset_df = all_scenarios.copy()

    if SCENARIO_FOCUS == 'StatusQuo':
        subset_df = subset_df[subset_df['scenario'].str.contains('statusquo|sq', case=False, regex=True)].copy()
    elif SCENARIO_FOCUS == 'Trend':
        subset_df = subset_df[~subset_df['scenario'].str.contains('statusquo|sq', case=False, regex=True)].copy()
        if subset_df.empty:
            subset_df = all_scenarios.copy()

    metric_cols = [
        'M3_Market_Share', 'M4_Intensity_Index', 'M6_RI',
        'M10_Demand_Index',
        'M7_Volume_Contribution', 'M8_Structure_Contribution', 'M9_Behavior_Contribution',
        'Master_Pop_Total', 'I0', 'Demand', 'D0',
        'Effect_Volume_Abs', 'Effect_Structure_Abs', 'Effect_Behavior_Abs'
    ]

    existing_cols = [c for c in metric_cols if c in subset_df.columns]

    agg_df = subset_df.groupby(['Year', 'Service', 'Nationality'])[existing_cols].agg(['median', 'min', 'max']).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

    rename_dict = {}
    for old_base, new_base in METRIC_MAP.items():
        if old_base in existing_cols:
            rename_dict[f'{old_base}_median'] = new_base
            rename_dict[f'{old_base}_min'] = f'{new_base}_min'
            rename_dict[f'{old_base}_max'] = f'{new_base}_max'

    standard_map = {
        'Master_Pop_Total_median': 'Master_Pop_Total',
        'I0_median': 'I0',
        'Demand_median': 'Demand',
        'D0_median': 'D0',
        'Demand_min': 'Demand_min',
        'Demand_max': 'Demand_max'
    }
    rename_dict.update(standard_map)

    agg_df = agg_df.rename(columns=rename_dict)

    print(f"   Saving processed data to {PATHS['output_csv']}...")
    agg_df.to_csv(PATHS['output_csv'], index=False)
    print("Done.\n")
    return agg_df

def main():
    load_and_calculate_metrics()


if __name__ == "__main__":
    main()
