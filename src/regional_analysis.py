"""Regional decomposition of demand growth by nationality, and regressions of regional demand growth on the foreign population share and on foreign population growth."""

import pandas as pd
import numpy as np
from scipy import stats
import os

import config

TARGET_YEAR = 2050
AGGREGATION_MODE = 'MIXED'

PATHS = {
    'edu': config.PATH_EDUCATION_DEMAND,
    'hous': config.PATH_HOUSING_DEMAND,
    'health': config.PATH_HEALTHCARE_DEMAND,
    'pop': config.PATH_PROJECTION,
}

SUFFIX = f"_{AGGREGATION_MODE}"
CSV_OUT = config.PATH_REGIONAL_DECOMPOSITION

os.makedirs('CSV', exist_ok=True)

def get_nuts2(region_code): return str(region_code)[:4]
def get_base_scenario(scen_str): return str(scen_str).split('--')[0]

def run_regional_analysis():
    print(f"Regional analysis ({AGGREGATION_MODE})...")

    print("   Loading population...")
    try:
        pop_df = pd.read_parquet(PATHS['pop'])
        if len(pop_df.columns) < 3: pop_df = pd.read_csv(PATHS['pop'], sep=';')
    except:
        pop_df = pd.read_csv(PATHS['pop'], sep=';')

    pop_df = pop_df.rename(columns={'TIME_PERIOD': 'Year', 'OBS_VALUE': 'Population', 'geo': 'Region_Code'})
    pop_df['Nationality'] = pop_df['Nationality'].astype(str).str.strip()
    pop_df['NUTS2'] = pop_df['Region_Code'].apply(get_nuts2)

    if 'scenario' in pop_df.columns: pop_df['base_scenario'] = pop_df['scenario'].apply(get_base_scenario)
    else: pop_df['base_scenario'] = 'Main'

    for c in ['Region_Code', 'NUTS2', 'Nationality', 'base_scenario']: pop_df[c] = pop_df[c].astype('category')

    def aggregate_pop(level):
        if level == 'NUTS3':
            grp = ['Year', 'Region_Code', 'Nationality', 'base_scenario']
        else:
            grp = ['Year', 'NUTS2', 'Nationality', 'base_scenario']

        df = pop_df.groupby(grp, observed=False)['Population'].sum().reset_index()
        if level == 'NUTS2': df = df.rename(columns={'NUTS2': 'Region_Code'})
        return df.rename(columns={'Population': 'Region_Pop'})

    pop_nuts2 = aggregate_pop('NUTS2')
    pop_nuts3 = aggregate_pop('NUTS3')

    def process_service(path, service_name, value_col, geo_col_in_file):
        print(f"   Processing {service_name}...")
        df = pd.read_parquet(path)
        if 'year' in df.columns: df = df.rename(columns={'year': 'Year'})
        df = df[df['Year'] >= 2025].copy()
        df = df.rename(columns={geo_col_in_file: 'Region_Code'})

        if AGGREGATION_MODE == 'NUTS2' or service_name == 'Healthcare':
            df['Region_Code'] = df['Region_Code'].apply(get_nuts2)
            pop_lookup = pop_nuts2
        else:
            pop_lookup = pop_nuts3

        df['Nationality'] = df['Nationality'].astype(str).str.strip()
        df['base_scenario'] = df['scenario'].apply(get_base_scenario)

        df_agg = df.groupby(['Year', 'Region_Code', 'Nationality', 'scenario', 'base_scenario'], observed=False)[value_col].sum().reset_index()
        df_agg = df_agg.rename(columns={value_col: 'Demand'})
        df_agg = df_agg[df_agg['Demand'] > 1e-6]

        df_merged = pd.merge(df_agg, pop_lookup, on=['Year', 'Region_Code', 'Nationality', 'base_scenario'], how='left')
        df_merged = df_merged.dropna(subset=['Region_Pop'])

        df_merged['Intensity'] = df_merged['Demand'] / df_merged['Region_Pop']
        df_merged['Service'] = service_name

        base_df = df_merged[df_merged['Year'] == 2025].copy()
        idx_cols = ['Region_Code', 'Nationality', 'scenario']

        d0_map = base_df.groupby(idx_cols, observed=False)['Demand'].first()
        p0_map = base_df.groupby(idx_cols, observed=False)['Region_Pop'].first()
        i0_map = base_df.groupby(idx_cols, observed=False)['Intensity'].first()

        temp_idx = df_merged.set_index(idx_cols).index
        df_merged['D0'] = temp_idx.map(d0_map)
        df_merged['P0'] = temp_idx.map(p0_map)
        df_merged['I0'] = temp_idx.map(i0_map)

        return df_merged.dropna(subset=['D0'])

    d_edu = process_service(PATHS['edu'], 'Education', 'Demand_teachers', 'NUTS3')
    d_hous = process_service(PATHS['hous'], 'Housing', 'Dwellings', 'NUTS3')
    d_health = process_service(PATHS['health'], 'Healthcare', 'Bed_Days_Mean', 'NUTS2')

    master_df = pd.concat([d_edu, d_hous, d_health], ignore_index=True)

    print("   Calculating Decomposition...")
    sq_mask = master_df['scenario'].str.contains('statusquo|sq', case=False)
    df_sq = master_df[sq_mask].copy()

    join_cols = ['Year', 'Region_Code', 'Nationality', 'Service', 'base_scenario']

    sq_value_cols = ['Demand', 'Region_Pop', 'Intensity', 'D0', 'P0', 'I0']
    if df_sq.empty:
        merged = master_df.copy()
        for c in sq_value_cols:
            merged[f'{c}_SQ'] = merged[c]
    else:
        df_sq = df_sq.groupby(join_cols, observed=False)[sq_value_cols].first().reset_index()
        merged = pd.merge(master_df, df_sq, on=join_cols, suffixes=('', '_SQ'), how='left')
        for c in sq_value_cols:
            merged[f'{c}_SQ'] = merged[f'{c}_SQ'].fillna(merged[c])

    raw_behaviour = merged['Demand'] - merged['Demand_SQ']
    base_rows = merged[merged['Year'] == 2025]
    gap_idx = ['Region_Code', 'Nationality', 'Service', 'scenario']
    gap_map = (base_rows.set_index(gap_idx)['Demand'] - base_rows.set_index(gap_idx)['Demand_SQ'])
    merged = merged.join(gap_map.rename('legacy_gap'), on=gap_idx)
    merged['legacy_gap'] = merged['legacy_gap'].fillna(0)
    merged['Beh_Abs'] = raw_behaviour - merged['legacy_gap']

    merged['Vol_Abs'] = (merged['Region_Pop_SQ'] - merged['P0_SQ']) * ((merged['Intensity_SQ'] + merged['I0_SQ']) / 2)
    merged['Struct_Abs'] = (merged['Intensity_SQ'] - merged['I0_SQ']) * ((merged['Region_Pop_SQ'] + merged['P0_SQ']) / 2)
    merged['Net_Abs'] = merged['Vol_Abs'] + merged['Struct_Abs'] + merged['Beh_Abs']

    is_sq = merged['scenario'].astype(str).str.contains('statusquo|sq', case=False, regex=True)
    condition = ~is_sq
    trend_subset = merged[condition].copy()

    print("   Calculating National Aggregates...")
    nat_cols = ['Demand', 'Region_Pop', 'D0', 'Vol_Abs', 'Struct_Abs', 'Beh_Abs', 'Net_Abs']
    nat_grp = ['Service', 'Year', 'Nationality', 'scenario']

    nat_df = trend_subset.groupby(nat_grp, observed=False)[nat_cols].sum().reset_index()
    nat_df['Region_Code'] = 'NAT'

    print("   Aggregating Statistics...")
    combined_data = pd.concat([trend_subset, nat_df], ignore_index=True)
    metric_cols = ['Demand', 'Region_Pop', 'D0', 'Vol_Abs', 'Struct_Abs', 'Beh_Abs', 'Net_Abs']
    final_grp = ['Year', 'Region_Code', 'Nationality', 'Service']

    scen_out = combined_data.loc[combined_data['Year'].isin([2025, TARGET_YEAR]),
                                 final_grp + ['scenario'] + metric_cols]
    scen_out.to_csv(config.PATH_REGIONAL_DECOMPOSITION_SCENARIOS, index=False)
    print(f"   Saved {config.PATH_REGIONAL_DECOMPOSITION_SCENARIOS}")

    agg_df = combined_data.groupby(final_grp, observed=False)[metric_cols].agg(['median', 'min', 'max']).reset_index()
    agg_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg_df.columns]

    for col in metric_cols:
        agg_df = agg_df.rename(columns={f'{col}_median': col})

    agg_df['Vol_Contrib_Pct'] = (agg_df['Vol_Abs'] / agg_df['D0']) * 100
    agg_df['Struct_Contrib_Pct'] = (agg_df['Struct_Abs'] / agg_df['D0']) * 100
    agg_df['Beh_Contrib_Pct'] = (agg_df['Beh_Abs'] / agg_df['D0']) * 100

    pop_pivot = agg_df.groupby(['Year', 'Region_Code', 'Nationality'])['Region_Pop'].mean().unstack(fill_value=0).reset_index()
    if 'Foreign country' in pop_pivot.columns:
        pop_pivot['Total_Region'] = pop_pivot.drop(columns=['Year']).sum(axis=1, numeric_only=True)
        pop_pivot['Foreign_Share'] = (pop_pivot['Foreign country'] / pop_pivot['Total_Region']) * 100
        agg_df = pd.merge(agg_df, pop_pivot[['Year', 'Region_Code', 'Foreign_Share']], on=['Year', 'Region_Code'], how='left')
    else:
        agg_df['Foreign_Share'] = np.nan

    natives = agg_df[agg_df['Nationality'] == 'Austria'].copy()
    natives['Native_Aging_Ratio'] = np.divide(
        natives['Struct_Abs'], natives['Vol_Abs'],
        out=np.zeros_like(natives['Struct_Abs']), where=np.abs(natives['Vol_Abs']) > 1e-9
    )

    final_df = pd.merge(agg_df, natives[['Year', 'Region_Code', 'Service', 'Native_Aging_Ratio']],
                  on=['Year', 'Region_Code', 'Service'], how='left')

    print("   Adding Per-Scenario Totals...")
    tot_grp = ['Year', 'Region_Code', 'Service', 'scenario']
    per_scen = combined_data[tot_grp + ['Nationality', 'Demand']].copy()
    per_scen['Demand_Total'] = per_scen.groupby(tot_grp, observed=True)['Demand'].transform('sum')
    per_scen['Demand_Share'] = (per_scen['Demand'] / per_scen['Demand_Total']) * 100
    tot = per_scen.drop_duplicates(tot_grp)[tot_grp + ['Demand_Total']].copy()
    base_idx = ['Region_Code', 'Service', 'scenario']
    t0_map = tot[tot['Year'] == 2025].set_index(base_idx)['Demand_Total']
    tot['T0'] = tot.set_index(base_idx).index.map(t0_map)
    tot['Demand_Total_Change'] = tot['Demand_Total'] - tot['T0']
    tot['Demand_Total_Growth_Pct'] = (tot['Demand_Total_Change'] / tot['T0']) * 100
    tot_df = tot.groupby(['Year', 'Region_Code', 'Service'], observed=True).agg(
        Demand_Total=('Demand_Total', 'median'),
        Demand_Total_min=('Demand_Total', 'min'),
        Demand_Total_max=('Demand_Total', 'max'),
        Demand_Total_Change=('Demand_Total_Change', 'median'),
        Demand_Total_Growth_Pct=('Demand_Total_Growth_Pct', 'median'),
        Demand_Total_Growth_Pct_min=('Demand_Total_Growth_Pct', 'min'),
        Demand_Total_Growth_Pct_max=('Demand_Total_Growth_Pct', 'max'),
    ).reset_index()
    share_df = per_scen.groupby(final_grp, observed=True)['Demand_Share'].median().reset_index()
    final_df = pd.merge(final_df, tot_df, on=['Year', 'Region_Code', 'Service'], how='left')
    final_df = pd.merge(final_df, share_df, on=final_grp, how='left')
    tot['NUTS2'] = tot['Region_Code'].str[:4]
    st = tot.groupby(['Year', 'NUTS2', 'Service', 'scenario'], observed=True)['Demand_Total'].sum().reset_index()
    st_idx = ['NUTS2', 'Service', 'scenario']
    s0_map = st[st['Year'] == 2025].set_index(st_idx)['Demand_Total']
    st['S0'] = st.set_index(st_idx).index.map(s0_map)
    st['NUTS2_Total_Growth_Pct'] = ((st['Demand_Total'] - st['S0']) / st['S0']) * 100
    st_df = st.groupby(['Year', 'NUTS2', 'Service'], observed=True).agg(
        NUTS2_Total_Growth_Pct=('NUTS2_Total_Growth_Pct', 'median'),
        NUTS2_Total_Growth_Pct_min=('NUTS2_Total_Growth_Pct', 'min'),
        NUTS2_Total_Growth_Pct_max=('NUTS2_Total_Growth_Pct', 'max'),
    ).reset_index()
    final_df['NUTS2'] = final_df['Region_Code'].astype(str).str[:4]
    final_df = pd.merge(final_df, st_df, on=['Year', 'NUTS2', 'Service'], how='left').drop(columns='NUTS2')

    final_df.to_csv(CSV_OUT, index=False)
    print(f"   Saved {CSV_OUT}")
    return final_df

def growth_vs_foreign_share(df):
    print("Regression of demand growth on foreign share...")
    df_clean = df[df['Region_Code'] != 'NAT']

    demand_agg = df_clean.groupby(['Year', 'Region_Code', 'Service'])['Demand'].sum().reset_index()
    growth_df = demand_agg.pivot_table(index=['Region_Code', 'Service'], columns='Year', values='Demand').reset_index()

    growth_df['Total_Growth_Pct'] = ((growth_df[TARGET_YEAR] - growth_df[2025]) / growth_df[2025]) * 100
    share_df = df_clean[df_clean['Year'] == TARGET_YEAR][['Region_Code', 'Service', 'Foreign_Share']].drop_duplicates()

    plot_df = pd.merge(growth_df, share_df, on=['Region_Code', 'Service'], how='inner')
    plot_df.to_csv(f'CSV/regression_input_data{SUFFIX}.csv', index=False)

    stats_list = []
    for service in plot_df['Service'].unique():
        subset = plot_df[plot_df['Service'] == service]
        if len(subset) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(subset['Foreign_Share'], subset['Total_Growth_Pct'])
            stats_list.append({
                'Service': service, 'Slope': slope, 'Intercept': intercept,
                'R2': r_value**2, 'P_Value': p_value, 'Std_Err': std_err
            })

    if stats_list: pd.DataFrame(stats_list).to_csv(f'CSV/regression_statistics{SUFFIX}.csv', index=False)

def temporal_regression(df):
    print("Regression of demand growth on foreign share, by year...")

    df_clean = df[df['Region_Code'] != 'NAT']

    df_agg = df_clean.groupby(['Year', 'Region_Code', 'Service'])['Demand'].sum().reset_index()
    share_lookup = df_clean[['Year', 'Region_Code', 'Foreign_Share']].drop_duplicates()
    df_agg = pd.merge(df_agg, share_lookup, on=['Year', 'Region_Code'], how='left')

    base_demand = df_agg[df_agg['Year'] == 2025][['Region_Code', 'Service', 'Demand']].rename(columns={'Demand': 'Base_Demand'})

    stats_history = []
    years = sorted([y for y in df['Year'].unique() if y > 2025 and y <= TARGET_YEAR])

    for year in years:
        current = df_agg[df_agg['Year'] == year].copy()
        merged = pd.merge(current, base_demand, on=['Region_Code', 'Service'])
        merged['Growth'] = ((merged['Demand'] - merged['Base_Demand']) / merged['Base_Demand']) * 100

        for service in merged['Service'].unique():
            sub = merged[merged['Service'] == service].dropna()
            if len(sub) > 5:
                slope, intercept, r_value, p_value, std_err = stats.linregress(sub['Foreign_Share'], sub['Growth'])
                stats_history.append({
                    'Year': year, 'Service': service, 'Slope': slope, 'Std_Err': std_err
                })

    reg_df = pd.DataFrame(stats_history)
    reg_df.to_csv(f"CSV/temporal_regression_stats{SUFFIX}.csv", index=False)

def growth_vs_foreign_growth(df):
    print("Regression of demand growth on foreign population growth...")
    df_clean = df[df['Region_Code'] != 'NAT']
    subset = df_clean[df_clean['Year'].isin([2025, TARGET_YEAR])].copy()

    f_pop = subset[subset['Nationality'] == 'Foreign country'].pivot_table(index='Region_Code', columns='Year', values='Region_Pop').reset_index()
    f_pop['F_Growth'] = ((f_pop[TARGET_YEAR] - f_pop[2025]) / f_pop[2025]) * 100

    dem = df_clean.groupby(['Year', 'Region_Code', 'Service'])['Demand'].sum().reset_index().pivot_table(index=['Region_Code', 'Service'], columns='Year', values='Demand').reset_index()
    dem['S_Growth'] = ((dem[TARGET_YEAR] - dem[2025]) / dem[2025]) * 100

    plot_df = pd.merge(dem, f_pop[['Region_Code', 'F_Growth']], on='Region_Code')

    stats_list = []
    for s in plot_df['Service'].unique():
        sub = plot_df[plot_df['Service'] == s]
        if len(sub) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(sub['F_Growth'], sub['S_Growth'])
            stats_list.append({'Service': s, 'Slope': slope, 'R2': r_value**2, 'Std_Err': std_err})

    if stats_list: pd.DataFrame(stats_list).to_csv(f'CSV/growth_vs_growth_stats{SUFFIX}.csv', index=False)

def temporal_growth_regression(df):
    print("Regression of demand growth on foreign population growth, by year...")

    df_clean = df[df['Region_Code'] != 'NAT']

    pop_df = df_clean[df_clean['Nationality'] == 'Foreign country'].groupby(['Year', 'Region_Code'])['Region_Pop'].sum().reset_index()
    pop_base = pop_df[pop_df['Year'] == 2025].rename(columns={'Region_Pop': 'Base_Foreign'})

    dem_df = df_clean.groupby(['Year', 'Region_Code', 'Service'])['Demand'].sum().reset_index()
    dem_base = dem_df[dem_df['Year'] == 2025].rename(columns={'Demand': 'Base_Demand'})

    stats_history = []
    years = sorted([y for y in df['Year'].unique() if y > 2025 and y <= TARGET_YEAR])

    for year in years:
        p_curr = pd.merge(pop_df[pop_df['Year']==year], pop_base, on='Region_Code')
        p_curr['F_Growth'] = ((p_curr['Region_Pop'] - p_curr['Base_Foreign'])/p_curr['Base_Foreign'])*100

        d_curr = pd.merge(dem_df[dem_df['Year']==year], dem_base, on=['Region_Code', 'Service'])
        d_curr['S_Growth'] = ((d_curr['Demand'] - d_curr['Base_Demand'])/d_curr['Base_Demand'])*100

        merged = pd.merge(d_curr, p_curr[['Region_Code', 'F_Growth']], on='Region_Code')
        for s in merged['Service'].unique():
            sub = merged[merged['Service']==s].dropna()
            sub = sub[sub['F_Growth'] < 500]
            if len(sub) > 5:
                slope, intercept, r_value, p_value, std_err = stats.linregress(sub['F_Growth'], sub['S_Growth'])
                stats_history.append({
                    'Year': year, 'Service': s, 'Slope': slope, 'Std_Err': std_err
                })

    reg_df = pd.DataFrame(stats_history)
    reg_df.to_csv(f"CSV/temporal_growth_growth_stats{SUFFIX}.csv", index=False)

def main():
    df = run_regional_analysis()
    growth_vs_foreign_share(df)
    temporal_regression(df)
    growth_vs_foreign_growth(df)
    temporal_growth_regression(df)
    print("Done.")


if __name__ == "__main__":
    main()
