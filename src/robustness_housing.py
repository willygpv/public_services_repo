"""Robustness tests of the housing model."""

import warnings
import numpy as np
import pandas as pd

import config
from oerok_projection import load_oerok_reference

warnings.filterwarnings('ignore')

BASE_YEAR = 2022
DEMAND_YEAR = 2050
DEMOGRAPHIC_COLS = ['NUTS3', 'Age', 'Sex', 'Nationality']

PATH_HISTORICAL = r'XLSX/df_hs_np_clean.csv'
PATH_PROJECTION = config.PATH_PROJECTION
PATH_CLUSTERS = r'Data/OEROK/Budget_forecasts/nuts3_coding_adapted.xlsx'
OEROK_YEARS = [2030, 2040, 2050]


def logit_transform(hhfr_rate):
    rate = np.clip(hhfr_rate / 100.0, 1e-6, 1.0 - 1e-6)
    return np.log(rate / (1.0 - rate))


def sigmoid_transform(logit_value):
    return (1.0 / (1.0 + np.exp(-logit_value))) * 100.0


def convert_age(age_str):
    if age_str == 'Y_GE100': return 100
    elif age_str == 'Y_LT1': return 0
    return int(str(age_str).lstrip('Y'))


def get_weighted_average_hhfr(df_subset, cohort=False):
    heads = (df_subset['Population'] * df_subset['HHFR'] / 100.0).sum()
    total_pop = df_subset['Population'].sum()
    if total_pop > 0:
        return (heads / total_pop) * 100.0
    return df_subset['HHFR'].mean() if cohort and len(df_subset) else 0.0


def two_way_anova_ss(df, value, factor_a, factor_b):
    grand = df[value].mean()
    ss_total = ((df[value] - grand) ** 2).sum()
    ss_a = ((df.groupby(factor_a)[value].transform('mean') - grand) ** 2).sum()
    ss_b = ((df.groupby(factor_b)[value].transform('mean') - grand) ** 2).sum()
    return {'total': ss_total, 'a': ss_a, 'b': ss_b, 'ab': ss_total - ss_a - ss_b}


def load_all_data():
    print("--- Loading all data ---")

    df_hs = pd.read_csv(PATH_HISTORICAL)
    df_hs['scenario'] = 'BSL'
    print(f"  Historical: {len(df_hs):,} rows, years={sorted(df_hs['Year'].unique())}")
    print(f"  Regions: {df_hs['NUTS3'].nunique()}, Nationalities: {df_hs['Nationality'].unique()}")

    proj = pd.read_parquet(PATH_PROJECTION)
    proj.rename(columns={
        'sex': 'Sex', 'age': 'Age', 'OBS_VALUE': 'Population',
        'geo': 'NUTS3', 'TIME_PERIOD': 'Year',
    }, inplace=True)
    proj['Age'] = proj['Age'].apply(convert_age)
    proj['Population'] = proj['Population'].clip(lower=0)
    proj = proj.groupby(
        ['Year', 'NUTS3', 'Age', 'Nationality', 'Sex', 'scenario'], as_index=False
    )['Population'].sum()
    proj = proj[proj['Year'] <= DEMAND_YEAR]
    print(f"  Projections: {len(proj):,} rows, scenarios={proj['scenario'].nunique()}")

    oerok_reference = load_oerok_reference(OEROK_YEARS)
    hh = oerok_reference['households']
    print(f"  OEROK: households {DEMAND_YEAR} = "
          f"{hh.loc[hh['Year'] == DEMAND_YEAR, 'Households'].sum():,.0f}")

    cluster_df = pd.read_excel(PATH_CLUSTERS)
    cluster_mapping = dict(zip(cluster_df['Code'], cluster_df['Cluster']))
    print(f"  Clusters: {len(cluster_mapping)} regions -> {len(set(cluster_mapping.values()))} clusters")

    return df_hs, proj, oerok_reference, cluster_mapping


def create_age_group_mapping():
    age_to_group = {age: None for age in range(0, 15)}
    for start in range(15, 95, 5):
        label = f"{start}-{start + 4}"
        for age in range(start, start + 5):
            age_to_group[age] = label
    for age in range(95, 101):
        age_to_group[age] = "95-100"
    return age_to_group


def create_nuts3_oerok_mapping():
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


def calculate_trends(df_hs, cluster_mapping, age_group_mapping,
                     start_year=2011, end_years=[2021, 2022]):
    df_start = df_hs[df_hs['Year'] == start_year].copy()
    df_end = df_hs[df_hs['Year'].isin(end_years)].copy()

    for df in [df_start, df_end]:
        df['Cluster'] = df['NUTS3'].map(cluster_mapping)
        df['AgeGroup'] = df['Age'].map(age_group_mapping)

    df_start = df_start[df_start['AgeGroup'].notna()]
    df_end = df_end[df_end['AgeGroup'].notna()]

    group_cols = ['Cluster', 'AgeGroup', 'Sex', 'Nationality']
    unique_groups = df_end[group_cols].drop_duplicates()
    end_year_mid = np.mean(end_years)

    trends = []
    for _, row in unique_groups.iterrows():
        cluster, ag, sex, nat = row['Cluster'], row['AgeGroup'], row['Sex'], row['Nationality']
        mask_s = ((df_start['Cluster'] == cluster) & (df_start['AgeGroup'] == ag) &
                  (df_start['Sex'] == sex) & (df_start['Nationality'] == nat))
        mask_e = ((df_end['Cluster'] == cluster) & (df_end['AgeGroup'] == ag) &
                  (df_end['Sex'] == sex) & (df_end['Nationality'] == nat))
        g_start = df_start[mask_s]
        g_end = df_end[mask_e]
        if g_end.empty:
            continue
        hhfr_end = get_weighted_average_hhfr(g_end)
        if not g_start.empty:
            hhfr_start = get_weighted_average_hhfr(g_start)
            slope = (logit_transform(hhfr_end) - logit_transform(hhfr_start)) / (end_year_mid - start_year)
        else:
            slope = 0
        trends.append({
            'Cluster': cluster, 'AgeGroup': ag, 'Sex': sex,
            'Nationality': nat, 'trend_slope': slope,
            'hhfr_start': hhfr_start if not g_start.empty else np.nan,
            'hhfr_end': hhfr_end,
            'pop_start': g_start['Population'].sum(),
            'pop_end': g_end['Population'].sum(),
        })
    return pd.DataFrame(trends)


def oerok_adjustment_factors(df_hhfr, oerok_reference, target_year):
    pop = oerok_reference['population']
    pop = pop[pop['Year'] == target_year][DEMOGRAPHIC_COLS + ['Population']]
    oerok_hh = oerok_reference['households']
    oerok_hh = oerok_hh[oerok_hh['Year'] == target_year].set_index('NUTS3')['Households']
    merged = df_hhfr.merge(pop, on=DEMOGRAPHIC_COLS, how='inner')
    model_hh = (merged['Population'] * merged['HHFR'] / 100).groupby(merged['NUTS3']).sum()
    factors = (oerok_hh / model_hh).reindex(df_hhfr['NUTS3'].unique())
    if factors.isna().any():
        raise ValueError(f"No OEROK factor for {factors[factors.isna()].index.tolist()}")
    return factors.clip(0.1, 5.0), model_hh, oerok_hh


def project_dwellings(df_hs, df_proj, cluster_mapping, age_group_mapping,
                      df_trends, target_year, oerok_data=None,
                      use_trend=True, apply_oerok=True):
    df_recent = df_hs[df_hs['Year'].isin([2021, 2022])].copy()
    cohort_cols = DEMOGRAPHIC_COLS

    base_stats = df_recent.groupby(cohort_cols).apply(
        lambda x: pd.Series({
            'base_hhfr': get_weighted_average_hhfr(x, cohort=True),
            'Cluster': cluster_mapping.get(x['NUTS3'].iloc[0]),
            'AgeGroup': age_group_mapping.get(x['Age'].iloc[0]),
        })
    ).reset_index()

    trend_lookup = {}
    if use_trend and df_trends is not None:
        for _, row in df_trends.iterrows():
            key = (row['Cluster'], row['AgeGroup'], row['Sex'], row['Nationality'])
            trend_lookup[key] = row['trend_slope']

    hhfr_proj = []
    for _, cohort in base_stats.iterrows():
        nuts3, age, sex, nat = cohort['NUTS3'], cohort['Age'], cohort['Sex'], cohort['Nationality']
        base_hhfr = cohort['base_hhfr']
        cluster, ag = cohort['Cluster'], cohort['AgeGroup']

        if ag is None:
            forecasted = 0
        elif use_trend:
            slope = trend_lookup.get((cluster, ag, sex, nat), 0)
            base_logit = logit_transform(base_hhfr)
            forecasted = sigmoid_transform(base_logit + slope * (target_year - 2021.5))
        else:
            forecasted = base_hhfr

        hhfr_proj.append({
            'NUTS3': nuts3, 'Age': age, 'Sex': sex, 'Nationality': nat,
            'HHFR': forecasted
        })

    df_hhfr = pd.DataFrame(hhfr_proj)

    if apply_oerok and oerok_data is not None and use_trend:
        adj_factors, _, _ = oerok_adjustment_factors(df_hhfr, oerok_data, target_year)
        df_hhfr['adj'] = df_hhfr['NUTS3'].map(adj_factors)
        df_hhfr['HHFR'] = np.clip(df_hhfr['HHFR'] * df_hhfr['adj'], 0, 100)
        df_hhfr.drop(columns=['adj'], inplace=True)

    proj_cells = df_proj[df_proj['Year'] == target_year].copy()
    merged = proj_cells.merge(df_hhfr, on=DEMOGRAPHIC_COLS, how='left')
    merged['HHFR'] = merged['HHFR'].fillna(0)
    merged['Dwellings'] = merged['Population'] * merged['HHFR'] / 100

    return merged['Dwellings'].sum(), df_hhfr


def test1_trend_anchor(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 1: TREND ANCHOR SENSITIVITY")
    print("=" * 80)

    one_scen = df_proj['scenario'].unique()[0]
    df_proj_one = df_proj[df_proj['scenario'] == one_scen]

    configs = {
        'Baseline (2011 -> avg 2021/22)': {'start': 2011, 'end': [2021, 2022]},
        '2011 -> 2021 only':              {'start': 2011, 'end': [2021]},
        '2011 -> 2022 only':              {'start': 2011, 'end': [2022]},
    }

    avail_years = sorted(df_hs['Year'].unique())
    print(f"  Available historical years: {avail_years}")

    print(f"\n--- Trend Slopes: Summary Statistics ---")
    print(f"{'Configuration':>35s} {'Mean_slope':>12s} {'Std_slope':>12s} {'Min':>10s} {'Max':>10s} {'Dwellings_2050':>16s}")

    ref_dwellings = None
    for name, cfg in configs.items():
        trends = calculate_trends(df_hs, cluster_mapping, age_group_mapping,
                                  start_year=cfg['start'], end_years=cfg['end'])
        dwellings, _ = project_dwellings(df_hs, df_proj_one, cluster_mapping,
                                         age_group_mapping, trends, DEMAND_YEAR,
                                         oerok_data, use_trend=True, apply_oerok=True)
        slopes = trends['trend_slope']
        if ref_dwellings is None:
            ref_dwellings = dwellings

        change = ((dwellings / ref_dwellings) - 1) * 100 if ref_dwellings else 0
        print(f"  {name:>33s} {slopes.mean():12.5f} {slopes.std():12.5f} "
              f"{slopes.min():10.5f} {slopes.max():10.5f} {dwellings:16,.0f} ({change:+.2f}%)")

    print(f"\n--- Trend Slope Distribution (Baseline config) ---")
    trends_base = calculate_trends(df_hs, cluster_mapping, age_group_mapping)
    print(f"  Total trend groups: {len(trends_base)}")
    print(f"  Positive slopes:    {(trends_base['trend_slope'] > 0).sum()} "
          f"({(trends_base['trend_slope'] > 0).mean() * 100:.1f}%)")
    print(f"  Negative slopes:    {(trends_base['trend_slope'] < 0).sum()} "
          f"({(trends_base['trend_slope'] < 0).mean() * 100:.1f}%)")
    print(f"  Zero slopes:        {(trends_base['trend_slope'] == 0).sum()}")

    print(f"\n--- Mean Slope by Nationality ---")
    for nat in trends_base['Nationality'].unique():
        sub = trends_base[trends_base['Nationality'] == nat]
        print(f"  {nat:>20s}: mean={sub['trend_slope'].mean():.5f}, "
              f"std={sub['trend_slope'].std():.5f}")

    print(f"\n--- Mean Slope by Cluster ---")
    for cl in sorted(trends_base['Cluster'].dropna().unique()):
        sub = trends_base[trends_base['Cluster'] == cl]
        print(f"  {str(cl):>30s}: mean={sub['trend_slope'].mean():.5f}, "
              f"std={sub['trend_slope'].std():.5f}")

    print(f"\n--- HHFR Start (2011) vs End (2021.5): National Weighted Averages ---")
    for nat in trends_base['Nationality'].unique():
        sub = trends_base[trends_base['Nationality'] == nat].dropna(subset=['hhfr_start'])
        if not sub.empty:
            w_start = np.average(sub['hhfr_start'], weights=sub['pop_start'])
            w_end = np.average(sub['hhfr_end'], weights=sub['pop_end'])
            print(f"  {nat:>20s}: start={w_start:.2f}, "
                  f"end={w_end:.2f}, "
                  f"change={w_end - w_start:+.2f}")
    print(f"  (weights: trend-group population, 2011 at the start, 2021+2022 at the end)")
    print(f"  Unweighted means over the trend groups (each cluster x age group x sex counts once):")
    for nat in trends_base['Nationality'].unique():
        sub = trends_base[trends_base['Nationality'] == nat].dropna(subset=['hhfr_start'])
        if not sub.empty:
            print(f"  {nat:>20s}: unweighted start={sub['hhfr_start'].mean():.2f}, "
                  f"end={sub['hhfr_end'].mean():.2f}, "
                  f"change={sub['hhfr_end'].mean() - sub['hhfr_start'].mean():+.2f}")


def test2_oerok_impact(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 2: ÖROK CONSTRAINT IMPACT")
    print("=" * 80)

    trends = calculate_trends(df_hs, cluster_mapping, age_group_mapping)
    one_scen = df_proj['scenario'].unique()[0]
    df_proj_one = df_proj[df_proj['scenario'] == one_scen]

    configs = {
        'Trend + ÖROK constrained':   {'use_trend': True, 'apply_oerok': True},
        'Trend unconstrained':         {'use_trend': True, 'apply_oerok': False},
        'Status-quo (no trend)':       {'use_trend': False, 'apply_oerok': False},
    }

    print(f"\n--- National Dwelling Demand at {DEMAND_YEAR} ---")
    print(f"{'Configuration':>35s} {'Dwellings':>15s} {'vs_constrained':>15s}")

    ref = None
    hhfr_results = {}
    for name, cfg in configs.items():
        dw, df_hhfr = project_dwellings(
            df_hs, df_proj_one, cluster_mapping, age_group_mapping, trends,
            DEMAND_YEAR, oerok_data, **cfg
        )
        if ref is None:
            ref = dw
        change = ((dw / ref) - 1) * 100
        print(f"  {name:>33s} {dw:15,.0f} {change:+14.2f}%")
        hhfr_results[name] = df_hhfr

    print(f"\n--- Regional ÖROK Adjustment Factors ---")
    print(f"  (households in {DEMAND_YEAR} on ÖROK's projected population: "
          f"unconstrained trend rates vs ÖROK)")
    nuts3_oerok = create_nuts3_oerok_mapping()

    df_hhfr_unc = hhfr_results.get('Trend unconstrained')
    if df_hhfr_unc is not None:
        factors, model_hh, oerok_hh = oerok_adjustment_factors(df_hhfr_unc, oerok_data, DEMAND_YEAR)
        reg = pd.DataFrame({'model_hh': model_hh, 'oerok_hh': oerok_hh, 'adj_factor': factors})
        reg = reg.sort_values('adj_factor')
        print(f"{'NUTS3':>8s} {'OEROK_Region':>30s} {'Model_HH':>14s} "
              f"{'OEROK_HH':>14s} {'Adj_factor':>12s}")
        for nuts3, r in reg.iterrows():
            print(f"  {nuts3:>6s} {str(nuts3_oerok.get(nuts3)):>30s} "
                  f"{r['model_hh']:14,.0f} {r['oerok_hh']:14,.0f} {r['adj_factor']:12.3f}")
        print(f"  {'Total':>6s} {'':>30s} {reg['model_hh'].sum():14,.0f} "
              f"{reg['oerok_hh'].sum():14,.0f} {reg['oerok_hh'].sum() / reg['model_hh'].sum():12.3f}")

    print(f"\n--- Mean HHFR by Nationality: Constrained vs Unconstrained ---")
    for key_name, df_h in hhfr_results.items():
        for nat in df_h['Nationality'].unique():
            sub = df_h[df_h['Nationality'] == nat]
            mean_hhfr = sub['HHFR'].mean()
            print(f"  {key_name:>35s} | {nat:>20s}: mean HHFR = {mean_hhfr:.3f}")


def test3_cluster_sensitivity(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 3: CLUSTER AGGREGATION SENSITIVITY")
    print("=" * 80)

    one_scen = df_proj['scenario'].unique()[0]
    df_proj_one = df_proj[df_proj['scenario'] == one_scen]

    trends_cluster = calculate_trends(df_hs, cluster_mapping, age_group_mapping)
    dw_cluster, _ = project_dwellings(
        df_hs, df_proj_one, cluster_mapping, age_group_mapping,
        trends_cluster, DEMAND_YEAR, oerok_data, use_trend=True
    )

    national_mapping = {k: 'National' for k in cluster_mapping}
    trends_national = calculate_trends(df_hs, national_mapping, age_group_mapping)
    dw_national, _ = project_dwellings(
        df_hs, df_proj_one, national_mapping, age_group_mapping,
        trends_national, DEMAND_YEAR, oerok_data, use_trend=True
    )

    region_mapping = {k: k for k in cluster_mapping}
    trends_region = calculate_trends(df_hs, region_mapping, age_group_mapping)
    dw_region, _ = project_dwellings(
        df_hs, df_proj_one, region_mapping, age_group_mapping,
        trends_region, DEMAND_YEAR, oerok_data, use_trend=True
    )

    print(f"\n--- Dwelling Demand at {DEMAND_YEAR} by Cluster Aggregation Level ---")
    print(f"{'Level':>25s} {'N_groups':>10s} {'Dwellings':>15s} {'vs_5cluster':>15s}")
    for label, dw, n_grp in [
        ('National (1 cluster)', dw_national, len(trends_national)),
        ('5 ÖROK clusters', dw_cluster, len(trends_cluster)),
        ('35 NUTS3 regions', dw_region, len(trends_region)),
    ]:
        change = ((dw / dw_cluster) - 1) * 100
        print(f"  {label:>23s} {n_grp:10d} {dw:15,.0f} {change:+14.2f}%")

    print(f"\n--- Slope Distribution by Aggregation Level ---")
    for label, tr in [('National', trends_national), ('5 clusters', trends_cluster),
                      ('35 regions', trends_region)]:
        slopes = tr['trend_slope']
        print(f"  {label:>12s}: mean={slopes.mean():.5f}, std={slopes.std():.5f}, "
              f"range=[{slopes.min():.5f}, {slopes.max():.5f}]")

    print(f"\n--- Slope Comparison: 5-Cluster vs Region-Level ---")
    region_slopes = trends_region.rename(columns={'Cluster': 'NUTS3'})
    region_slopes['Cluster'] = region_slopes['NUTS3'].map(cluster_mapping)
    merged = trends_cluster.merge(
        region_slopes, on=['Cluster', 'AgeGroup', 'Sex', 'Nationality'],
        suffixes=('_cluster', '_region'), how='inner'
    )
    if not merged.empty:
        corr = merged['trend_slope_cluster'].corr(merged['trend_slope_region'])
        mae = np.mean(np.abs(merged['trend_slope_cluster'] - merged['trend_slope_region']))
        print(f"  Correlation: {corr:.4f}")
        print(f"  MAE of slopes: {mae:.5f}")
        rank_corr = merged['trend_slope_cluster'].corr(merged['trend_slope_region'],
                                                        method='spearman')
        print(f"  Pairs (region x own cluster): {len(merged)} of {len(trends_region)} region groups")
        print(f"  Rank correlation (Spearman): {rank_corr:.4f}")


def test4_smoothing_sensitivity(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 4: SMOOTHING METHOD SENSITIVITY")
    print("=" * 80)

    trends = calculate_trends(df_hs, cluster_mapping, age_group_mapping)
    one_scen = df_proj['scenario'].unique()[0]
    df_proj_one = df_proj[df_proj['scenario'] == one_scen]

    _, df_hhfr_raw = project_dwellings(
        df_hs, df_proj_one, cluster_mapping, age_group_mapping,
        trends, DEMAND_YEAR, oerok_data, use_trend=True
    )

    def apply_smoothing_to_hhfr(df_hhfr, method, strength):
        df_s = df_hhfr.copy()
        cohort_cols = ['NUTS3', 'Sex', 'Nationality']
        for _, group in df_s.groupby(cohort_cols):
            g_sorted = group.sort_values('Age')
            ages = g_sorted['Age'].values
            hhfr = g_sorted['HHFR'].values
            if len(hhfr) < 5:
                continue
            if method == 'rolling':
                smoothed = pd.Series(hhfr).rolling(
                    window=int(strength), center=True, min_periods=1
                ).mean().values
            elif method == 'gaussian':
                from scipy import ndimage
                smoothed = ndimage.gaussian_filter1d(hhfr, sigma=strength, mode='nearest')
            else:
                smoothed = hhfr
            smoothed = np.clip(smoothed, 0, 100)
            mask = df_s.index.isin(g_sorted.index)
            df_s.loc[g_sorted.index, 'HHFR'] = smoothed
        return df_s

    def compute_dwellings_from_hhfr(df_hhfr, df_proj_yr):
        proj_cells = df_proj_yr[df_proj_yr['Year'] == DEMAND_YEAR].copy()
        merged = proj_cells.merge(df_hhfr, on=DEMOGRAPHIC_COLS, how='left')
        merged['HHFR'] = merged['HHFR'].fillna(0)
        return (merged['Population'] * merged['HHFR'] / 100).sum()

    configs = [
        ('No smoothing', 'none', 0),
        ('Rolling(3)', 'rolling', 3),
        ('Rolling(5)', 'rolling', 5),
        ('Rolling(7)', 'rolling', 7),
        ('Gaussian(1.0)', 'gaussian', 1.0),
        ('Gaussian(2.0)', 'gaussian', 2.0),
    ]

    print(f"\n--- Dwelling Demand at {DEMAND_YEAR} by Smoothing Method ---")
    print(f"{'Method':>20s} {'Dwellings':>15s} {'vs_no_smooth':>15s} {'HHFR_mean':>10s} {'HHFR_std':>10s}")

    ref_dw = None
    for name, method, strength in configs:
        if method == 'none':
            df_h = df_hhfr_raw.copy()
        else:
            df_h = apply_smoothing_to_hhfr(df_hhfr_raw, method, strength)

        dw = compute_dwellings_from_hhfr(df_h, df_proj_one)
        if ref_dw is None:
            ref_dw = dw
        change = ((dw / ref_dw) - 1) * 100
        print(f"  {name:>18s} {dw:15,.0f} {change:+14.2f}% "
              f"{df_h['HHFR'].mean():10.3f} {df_h['HHFR'].std():10.3f}")

    print(f"\n--- Age-to-Age Discontinuity (Mean Absolute HHFR Jump) ---")
    for name, method, strength in configs:
        if method == 'none':
            df_h = df_hhfr_raw.copy()
        else:
            df_h = apply_smoothing_to_hhfr(df_hhfr_raw, method, strength)

        jumps = []
        for _, group in df_h.groupby(['NUTS3', 'Sex', 'Nationality']):
            g = group.sort_values('Age')
            diffs = np.abs(np.diff(g['HHFR'].values))
            jumps.extend(diffs)
        mean_jump = np.mean(jumps)
        max_jump = np.max(jumps)
        print(f"  {name:>18s}: mean_jump={mean_jump:.4f}, max_jump={max_jump:.2f}")


def test5_nationality_decomposition(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 5: NATIONALITY HHFR DECOMPOSITION AND CONVERGENCE COUNTERFACTUAL")
    print("=" * 80)

    trends = calculate_trends(df_hs, cluster_mapping, age_group_mapping)
    one_scen = df_proj['scenario'].unique()[0]
    df_proj_one = df_proj[df_proj['scenario'] == one_scen]

    dw_actual, df_hhfr_actual = project_dwellings(
        df_hs, df_proj_one, cluster_mapping, age_group_mapping,
        trends, DEMAND_YEAR, oerok_data, use_trend=True
    )

    print(f"\n--- HHFR Profiles by Nationality and Age (base period, pop-weighted) ---")
    df_recent = df_hs[df_hs['Year'].isin([2021, 2022])].copy()
    age_group_map_5yr = {}
    for age in range(0, 15): age_group_map_5yr[age] = '0-14'
    for start in range(15, 100, 10):
        label = f"{start}-{start+9}"
        for a in range(start, min(start+10, 101)):
            age_group_map_5yr[a] = label

    df_recent['AG'] = df_recent['Age'].map(age_group_map_5yr)
    profile = df_recent.groupby(['AG', 'Nationality']).apply(
        lambda g: pd.Series({
            'w_hhfr': (g['Population'] * g['HHFR']).sum() / g['Population'].sum()
                      if g['Population'].sum() > 0 else 0,
            'pop': g['Population'].sum()
        })
    ).reset_index()

    pivot = profile.pivot(index='AG', columns='Nationality', values='w_hhfr')
    pop_pivot = profile.pivot(index='AG', columns='Nationality', values='pop')
    nats = pivot.columns.tolist()

    print(f"{'Age Group':>12s}", end='')
    for n in nats:
        print(f" {'HHFR_' + n[:3]:>12s}", end='')
    if len(nats) == 2:
        print(f" {'Ratio':>8s}", end='')
    print()

    for ag in sorted(pivot.index, key=lambda x: int(x.split('-')[0]) if '-' in x else 0):
        print(f"  {ag:>10s}", end='')
        vals = []
        for n in nats:
            v = pivot.loc[ag, n] if ag in pivot.index else np.nan
            print(f" {v:12.2f}", end='')
            vals.append(v)
        if len(vals) == 2 and vals[0] > 0:
            print(f" {vals[1]/vals[0]:8.3f}", end='')
        print()

    print(f"\n--- Counterfactual: Foreign HHFR = Austrian HHFR ---")
    df_hhfr_cf = df_hhfr_actual.copy()
    austrian_hhfr = df_hhfr_cf[df_hhfr_cf['Nationality'] == 'Austria'][
        ['NUTS3', 'Age', 'Sex', 'HHFR']
    ].rename(columns={'HHFR': 'HHFR_Austrian'})

    df_hhfr_cf = df_hhfr_cf.merge(austrian_hhfr, on=['NUTS3', 'Age', 'Sex'], how='left')
    mask_foreign = df_hhfr_cf['Nationality'] == 'Foreign country'
    df_hhfr_cf.loc[mask_foreign, 'HHFR'] = df_hhfr_cf.loc[mask_foreign, 'HHFR_Austrian'].fillna(
        df_hhfr_cf.loc[mask_foreign, 'HHFR']
    )

    proj_cells = df_proj_one[df_proj_one['Year'] == DEMAND_YEAR].copy()
    merged_cf = proj_cells.merge(
        df_hhfr_cf[DEMOGRAPHIC_COLS + ['HHFR']], on=DEMOGRAPHIC_COLS, how='left'
    )
    merged_cf['HHFR'] = merged_cf['HHFR'].fillna(0)
    dw_cf = (merged_cf['Population'] * merged_cf['HHFR'] / 100).sum()

    merged_actual = proj_cells.merge(
        df_hhfr_actual[DEMOGRAPHIC_COLS + ['HHFR']], on=DEMOGRAPHIC_COLS, how='left'
    )
    merged_actual['HHFR'] = merged_actual['HHFR'].fillna(0)
    merged_actual['Dw'] = merged_actual['Population'] * merged_actual['HHFR'] / 100

    dw_aus = merged_actual[merged_actual['Nationality'] == 'Austria']['Dw'].sum()
    dw_for = merged_actual[merged_actual['Nationality'] == 'Foreign country']['Dw'].sum()

    print(f"  Actual total:        {dw_actual:,.0f}")
    print(f"    Austrian:          {dw_aus:,.0f} ({dw_aus/dw_actual*100:.1f}%)")
    print(f"    Foreign:           {dw_for:,.0f} ({dw_for/dw_actual*100:.1f}%)")
    print(f"  Counterfactual:      {dw_cf:,.0f}")
    print(f"  Difference:          {dw_cf - dw_actual:+,.0f} ({(dw_cf/dw_actual - 1)*100:+.2f}%)")

    print(f"\n--- Implied Avg Persons per Dwelling by Nationality (2050, one scenario) ---")
    for nat in merged_actual['Nationality'].unique():
        sub = merged_actual[merged_actual['Nationality'] == nat]
        total_pop = sub['Population'].sum()
        total_dw = sub['Dw'].sum()
        if total_dw > 0:
            print(f"  {nat:>20s}: {total_pop/total_dw:.3f} persons/dwelling "
                  f"(pop={total_pop:,.0f}, dw={total_dw:,.0f})")


def test6_sq_vs_trend(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 6: STATUS-QUO vs TREND DIVERGENCE ACROSS SCENARIOS (OPTIMIZED)")
    print("=" * 80)

    print("  Calculating HHFR rates (Trend & SQ)...")
    trends = calculate_trends(df_hs, cluster_mapping, age_group_mapping)

    one_scen = df_proj['scenario'].unique()[0]
    df_proj_dummy = df_proj[df_proj['scenario'] == one_scen]

    _, df_hhfr_trend = project_dwellings(
        df_hs, df_proj_dummy, cluster_mapping, age_group_mapping,
        trends, DEMAND_YEAR, oerok_data, use_trend=True, apply_oerok=True
    )

    _, df_hhfr_sq = project_dwellings(
        df_hs, df_proj_dummy, cluster_mapping, age_group_mapping,
        trends, DEMAND_YEAR, oerok_data, use_trend=False, apply_oerok=False
    )

    print("  Filtering population data for target year...")
    pop_target = df_proj[df_proj['Year'] == DEMAND_YEAR].copy()

    print("  Merging rates and calculating dwellings...")
    rates_trend = df_hhfr_trend[DEMOGRAPHIC_COLS + ['HHFR']].rename(columns={'HHFR': 'HHFR_trend'})
    rates_sq = df_hhfr_sq[DEMOGRAPHIC_COLS + ['HHFR']].rename(columns={'HHFR': 'HHFR_sq'})

    merged = pop_target.merge(rates_trend, on=DEMOGRAPHIC_COLS, how='left')
    merged = merged.merge(rates_sq, on=DEMOGRAPHIC_COLS, how='left')

    merged['HHFR_trend'] = merged['HHFR_trend'].fillna(0)
    merged['HHFR_sq'] = merged['HHFR_sq'].fillna(0)

    merged['Dw_trend'] = merged['Population'] * merged['HHFR_trend'] / 100.0
    merged['Dw_sq']    = merged['Population'] * merged['HHFR_sq']    / 100.0

    print("  Aggregating results...")
    df_res = merged.groupby('scenario')[['Dw_trend', 'Dw_sq']].sum().reset_index()

    df_res['Diff'] = df_res['Dw_trend'] - df_res['Dw_sq']
    df_res['Diff_pct'] = ((df_res['Dw_trend'] / df_res['Dw_sq']) - 1) * 100

    print(f"\n--- Status-Quo vs Trend at {DEMAND_YEAR} ---")
    print(f"  Scenarios:           {len(df_res)}")
    print(f"  Mean Dw (trend):     {df_res['Dw_trend'].mean():,.0f}")
    print(f"  Mean Dw (statusquo): {df_res['Dw_statusquo'].mean():,.0f}" if 'Dw_statusquo' in df_res else f"  Mean Dw (statusquo): {df_res['Dw_sq'].mean():,.0f}")
    print(f"  Mean difference:     {df_res['Diff'].mean():+,.0f} ({df_res['Diff_pct'].mean():+.2f}%)")
    print(f"  Std difference:      {df_res['Diff'].std():,.0f}")
    print(f"  Range trend:         [{df_res['Dw_trend'].min():,.0f}, {df_res['Dw_trend'].max():,.0f}]")
    print(f"  Range statusquo:     [{df_res['Dw_sq'].min():,.0f}, {df_res['Dw_sq'].max():,.0f}]")

    print(f"\n  Trend > SQ in {(df_res['Diff'] > 0).sum()}/{len(df_res)} scenarios")
    print(f"  Trend < SQ in {(df_res['Diff'] < 0).sum()}/{len(df_res)} scenarios")

    return df_res


def test7_leave_cluster_out(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 7: LEAVE-ONE-CLUSTER-OUT")
    print("=" * 80)

    clusters = sorted(set(cluster_mapping.values()))
    print(f"Clusters: {clusters}")

    one_scen = df_proj['scenario'].unique()[0]
    df_proj_one = df_proj[df_proj['scenario'] == one_scen]

    trends_full = calculate_trends(df_hs, cluster_mapping, age_group_mapping)
    dw_full, _ = project_dwellings(
        df_hs, df_proj_one, cluster_mapping, age_group_mapping,
        trends_full, DEMAND_YEAR, oerok_data, use_trend=True
    )

    dw_full_nat = dw_full

    results = []
    for held_out_cluster in clusters:
        held_out_regions = [k for k, v in cluster_mapping.items() if v == held_out_cluster]

        est_mapping = {
            k: ('POOL' if v != held_out_cluster else v)
            for k, v in cluster_mapping.items()
        }
        trends_est = calculate_trends(df_hs, est_mapping, age_group_mapping)
        trends_pool = trends_est[trends_est['Cluster'] == 'POOL']

        pool_mapping = {k: 'POOL' for k in cluster_mapping}

        df_proj_held = df_proj_one[df_proj_one['NUTS3'].isin(held_out_regions)]

        dw_held, _ = project_dwellings(
            df_hs, df_proj_held, pool_mapping, age_group_mapping,
            trends_pool, DEMAND_YEAR, oerok_data, use_trend=True
        )

        dw_held_full, _ = project_dwellings(
            df_hs, df_proj_held, cluster_mapping, age_group_mapping,
            trends_full, DEMAND_YEAR, oerok_data, use_trend=True
        )

        pct_diff = ((dw_held / dw_held_full) - 1) * 100 if dw_held_full > 0 else 0

        dw_nat_loco = dw_full_nat - dw_held_full + dw_held
        nat_diff = ((dw_nat_loco / dw_full_nat) - 1) * 100 if dw_full_nat > 0 else 0

        results.append({
            'Held_Out_Cluster': held_out_cluster,
            'N_regions': len(held_out_regions),
            'Dw_full_model': dw_held_full,
            'Dw_pooled_other_trends': dw_held,
            'Diff_pct': pct_diff,
            'National_full': dw_full_nat,
            'National_LOCO': dw_nat_loco,
            'National_Diff_pct': nat_diff,
        })

    df_res = pd.DataFrame(results)
    print(f"\n--- Leave-One-Cluster-Out Results ---")
    print(df_res.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
    print(f"\nMean absolute difference (held-out regions): {df_res['Diff_pct'].abs().mean():.2f}%")
    print(f"Max absolute difference:  {df_res['Diff_pct'].abs().max():.2f}% "
          f"({df_res.loc[df_res['Diff_pct'].abs().idxmax(), 'Held_Out_Cluster']})")
    print(f"National total: mean absolute difference {df_res['National_Diff_pct'].abs().mean():.3f}%, "
          f"max {df_res['National_Diff_pct'].abs().max():.3f}% "
          f"({df_res.loc[df_res['National_Diff_pct'].abs().idxmax(), 'Held_Out_Cluster']})")


def test8_variance_decomposition(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data):
    print("\n" + "=" * 80)
    print("TEST 8: VARIANCE DECOMPOSITION (Demographic Scenario vs Behavioral Model)")
    print("=" * 80)

    trends = calculate_trends(df_hs, cluster_mapping, age_group_mapping)
    scenarios = df_proj['scenario'].unique()
    print(f"  Demographic scenarios: {len(scenarios)}")

    results = []
    for i, scen in enumerate(scenarios):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(scenarios)}...", end="\r")
        df_proj_s = df_proj[df_proj['scenario'] == scen]

        dw_trend, _ = project_dwellings(
            df_hs, df_proj_s, cluster_mapping, age_group_mapping,
            trends, DEMAND_YEAR, oerok_data, use_trend=True, apply_oerok=True
        )
        dw_sq, _ = project_dwellings(
            df_hs, df_proj_s, cluster_mapping, age_group_mapping,
            trends, DEMAND_YEAR, oerok_data, use_trend=False, apply_oerok=False
        )
        results.append({
            'scenario': scen, 'Behavioral': 'trend', 'Dwellings': dw_trend
        })
        results.append({
            'scenario': scen, 'Behavioral': 'statusquo', 'Dwellings': dw_sq
        })

    print()
    df_res = pd.DataFrame(results)

    grand_mean = df_res['Dwellings'].mean()
    scen_means = df_res.groupby('scenario')['Dwellings'].mean()
    beh_means = df_res.groupby('Behavioral')['Dwellings'].mean()

    ss = two_way_anova_ss(df_res, 'Dwellings', 'scenario', 'Behavioral')

    print(f"\n--- Variance Decomposition at {DEMAND_YEAR} (two-way ANOVA, sums of squares) ---")
    print(f"  Grand mean:                      {grand_mean:,.0f}")
    print(f"  Total sum of squares:            {ss['total']:,.0f}")
    print(f"  Demographic scenario:            {ss['a']:,.0f} ({ss['a'] / ss['total'] * 100:.1f}%)")
    print(f"  Behavioral model:                {ss['b']:,.0f} ({ss['b'] / ss['total'] * 100:.1f}%)")
    print(f"  Interaction (residual):          {ss['ab']:,.0f} ({ss['ab'] / ss['total'] * 100:.2f}%)")

    print(f"\n--- Range by Source ---")
    print(f"  Demographic range:  [{scen_means.min():,.0f}, {scen_means.max():,.0f}] "
          f"(span: {scen_means.max() - scen_means.min():,.0f})")
    print(f"  Behavioral range:   [{beh_means.min():,.0f}, {beh_means.max():,.0f}] "
          f"(span: {beh_means.max() - beh_means.min():,.0f})")

    demo_span_pct = (scen_means.max() - scen_means.min()) / grand_mean * 100
    beh_span_pct = (beh_means.max() - beh_means.min()) / grand_mean * 100
    print(f"\n  Demographic span as % of mean: {demo_span_pct:.1f}%")
    print(f"  Behavioral span as % of mean:  {beh_span_pct:.1f}%")

    print(f"\n--- Top/Bottom 5 Demographic Scenarios ---")
    sorted_scen = scen_means.sort_values()
    print("  Bottom 5:")
    for s, v in sorted_scen.head(5).items():
        print(f"    {s}: {v:,.0f}")
    print("  Top 5:")
    for s, v in sorted_scen.tail(5).items():
        print(f"    {s}: {v:,.0f}")

    return df_res


def main():
    print("=" * 80)
    print("HOUSING MODEL ROBUSTNESS TESTS")
    print("=" * 80)

    df_hs, df_proj, oerok_data, cluster_mapping = load_all_data()
    age_group_mapping = create_age_group_mapping()

    test1_trend_anchor(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)
    test2_oerok_impact(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)
    test3_cluster_sensitivity(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)
    test4_smoothing_sensitivity(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)
    test5_nationality_decomposition(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)
    test6_sq_vs_trend(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)
    test7_leave_cluster_out(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)
    test8_variance_decomposition(df_hs, df_proj, cluster_mapping, age_group_mapping, oerok_data)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
