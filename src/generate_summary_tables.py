"""Tables of national and regional demand, growth and decomposition, printed to the console."""

import pandas as pd
import os

import config

AGGREGATION_MODE = 'MIXED'
DECOMP_CSV = config.PATH_REGIONAL_DECOMPOSITION_SCENARIOS
POP_FILE = config.PATH_PROJECTION

NUTS2_NAMES = {
    'AT11': 'Burgenland', 'AT12': 'Lower Austria', 'AT13': 'Vienna',
    'AT21': 'Carinthia',  'AT22': 'Styria',
    'AT31': 'Upper Austria', 'AT32': 'Salzburg', 'AT33': 'Tyrol', 'AT34': 'Vorarlberg'
}

def load_data():
    if not os.path.exists(DECOMP_CSV):
        raise FileNotFoundError(f"{DECOMP_CSV} not found (run plot_regional_analysis.py first)")

    df_main = pd.read_csv(DECOMP_CSV)
    df_main['Year'] = pd.to_numeric(df_main['Year'], errors='coerce').fillna(0).astype(int)

    if 'Beh_Abs' not in df_main.columns: df_main['Beh_Abs'] = df_main.get('Effect_Behavior_Abs', 0.0)
    if 'Vol_Abs' not in df_main.columns: df_main['Vol_Abs'] = df_main.get('Effect_Volume_Abs', 0.0)
    if 'Struct_Abs' not in df_main.columns: df_main['Struct_Abs'] = df_main.get('Effect_Structure_Abs', 0.0)

    if 'scenario' not in df_main.columns:
        raise ValueError(f"{DECOMP_CSV} has no 'scenario' column (re-run plot_regional_analysis.py)")
    df_main['scenario'] = df_main['scenario'].astype(str)

    df_main['NUTS2'] = df_main['Region_Code'].str[:4]

    try:
        df_pop = pd.read_parquet(POP_FILE, columns=['TIME_PERIOD', 'age', 'Nationality', 'scenario', 'OBS_VALUE'],
                                 filters=[('TIME_PERIOD', 'in', [2025, 2050])])
        if len(df_pop.columns) < 3: df_pop = pd.read_csv(POP_FILE, sep=';')
        if 'TIME_PERIOD' in df_pop.columns:
            df_pop['TIME_PERIOD'] = pd.to_numeric(df_pop['TIME_PERIOD'], errors='coerce').fillna(0).astype(int)
        df_pop['base_scenario'] = df_pop['scenario'].astype(str).str.split('--').str[0]
    except Exception as e:
        raise RuntimeError(f'Cannot load the population file {POP_FILE}') from e

    return df_main, df_pop

def scen_stats(df, by, cols):
    st = df.groupby(by)[cols].agg(['median', 'min', 'max'])
    st.columns = [c if s == 'median' else f'{c}_{s}' for c, s in st.columns]
    return st.reset_index()

def table_1_summary(df):
    print("\nTable 1 (national scenarios)...")
    df = df[df['Year'].isin([2025, 2050])].copy()
    if df.empty: return

    df_nat = df[df['Region_Code'] == 'NAT'].copy()
    if df_nat.empty:
        print("WARNING: 'NAT' rows not found in CSV. Falling back to sum.")
        df_nat = df[df['Region_Code'] != 'NAT'].groupby(['Service', 'Year', 'Nationality', 'scenario'])[['Demand', 'Region_Pop']].sum().reset_index()

    agg_serv = df_nat.groupby(['Service', 'Year', 'Nationality', 'scenario'])['Demand'].sum().reset_index()

    pop_ref = df_nat[df_nat['Service'] == 'Housing'].copy()
    agg_pop = pop_ref.groupby(['Year', 'Nationality', 'scenario'])['Region_Pop'].sum().reset_index()
    agg_pop['Service'] = 'Population'
    agg_pop = agg_pop.rename(columns={'Region_Pop': 'Demand'})

    combined = pd.concat([agg_serv, agg_pop], ignore_index=True)
    totals = combined.groupby(['Service', 'Year', 'scenario'])['Demand'].sum().reset_index()
    totals['Nationality'] = 'Total'
    final = pd.concat([combined, totals], ignore_index=True)

    w = final.pivot_table(index=['Service', 'Nationality', 'scenario'], columns='Year', values='Demand').reset_index()
    w['Growth'] = ((w[2050] - w[2025]) / w[2025]) * 100
    w = w.rename(columns={2025: 'Demand_2025', 2050: 'Demand_2050'})
    p = scen_stats(w, ['Service', 'Nationality'], ['Demand_2025', 'Demand_2050', 'Growth'])
    p = p.rename(columns={'Demand_2050_min': 'Demand_min_2050', 'Demand_2050_max': 'Demand_max_2050',
                          'Growth_min': 'Growth_Min', 'Growth_max': 'Growth_Max'})

    s_map = {'Population':0, 'Education':1, 'Housing':2, 'Healthcare':3}
    n_map = {'Austria':1, 'Austrian':1, 'Foreign country':2, 'Foreign':2, 'Total':3}
    p['S'] = p['Service'].map(s_map)
    p['N'] = p['Nationality'].map(n_map)
    p = p.sort_values(['S', 'N'])

    print("\n" + "="*120)
    print(f"{'Table 1: Projected Demand (2050) with Uncertainty Ranges':^120}")
    print("="*120)
    print(f"{'Sector':<12} {'Nationality':<15} {'2025 (Abs)':>12} {'2050 Median [Min, Max]':>35} {'Growth % [Range]':>35}")
    print("-" * 120)

    for _, row in p.iterrows():
        s = row['Service']
        n = row['Nationality'].replace('Foreign country', 'Foreign').replace('Austria', 'Austrian')

        scale = 1e6 if s in ['Population', 'Housing', 'Healthcare'] else 1000
        if s == 'Healthcare' and row['Demand_2025'] < 1e6:
            scale = 1000

        suffix = "m" if scale == 1e6 else "k"

        val_fmt = "{:.1f}"

        v25 = f"{val_fmt.format(row['Demand_2025']/scale)}{suffix}"

        v50_med = val_fmt.format(row['Demand_2050']/scale)
        v50_min = val_fmt.format(row['Demand_min_2050']/scale)
        v50_max = val_fmt.format(row['Demand_max_2050']/scale)
        v50_str = f"{v50_med} [{v50_min}, {v50_max}]"

        gr_str = f"{row['Growth']:+.1f}% [{row['Growth_Min']:+.1f}, {row['Growth_Max']:+.1f}]"

        if n == 'Total':
            print("-" * 120)
            n = "TOTAL"

        print(f"{s:<12} {n:<15} {v25:>12} {v50_str:>35} {gr_str:>35}")
    print("="*120 + "\n")

def table_2_group_decomp(df):
    print("Table 2...")
    df_nat = df[(df['Year'] == 2050) & (df['Region_Code'] == 'NAT')].copy()
    if df_nat.empty: df_nat = df[df['Year'] == 2050].copy()

    comps = ['Vol_Abs', 'Struct_Abs', 'Beh_Abs']
    agg = df_nat.groupby(['Service', 'Nationality', 'scenario'])[['D0'] + comps].sum().reset_index()
    d0 = agg['D0'].where(agg['D0'] != 0, 1.0)
    for c in comps: agg[c] = (agg[c] / d0) * 100
    agg = scen_stats(agg, ['Service', 'Nationality'], comps)
    agg = agg.sort_values('Service')

    print("\n" + "="*140)
    print(f"{'Table 2: Growth Drivers (Median % [Min, Max])':^140}")
    print("="*140)
    print(f"{'Sector':<12} {'Nationality':<15} {'Volume':>35} {'Behavior':>35} {'Structure':>35}")
    print("-" * 140)

    curr = ""
    for _, row in agg.iterrows():
        s = row['Service']
        n = row['Nationality'].replace('Foreign country', 'Foreign').replace('Austria', 'Austrian')

        def fmt_comp(name):
            med, c_min, c_max = row[name], row[f'{name}_min'], row[f'{name}_max']
            return f"{med:+.1f}% [{min(c_min, c_max):+.1f}, {max(c_min, c_max):+.1f}]"

        if s != curr:
            if curr: print("-" * 140)
            curr = s
        print(f"{s:<12} {n:<15} {fmt_comp('Vol_Abs'):>35} {fmt_comp('Beh_Abs'):>35} {fmt_comp('Struct_Abs'):>35}")
    print("="*140 + "\n")

def table_3_waterfall_text(df):
    print("Table 3...")
    df_nat = df[(df['Year'] == 2050) & (df['Region_Code'] == 'NAT')].copy()
    if df_nat.empty: df_nat = df[df['Year'] == 2050].copy()

    comps = ['Vol_Abs', 'Struct_Abs', 'Beh_Abs']
    agg = df_nat.groupby(['Service', 'Nationality', 'scenario'])[['D0'] + comps].sum().reset_index()
    nat_base = agg.groupby(['Service', 'scenario'])['D0'].sum().reset_index().rename(columns={'D0': 'D0_Total'})
    merged = pd.merge(agg, nat_base, on=['Service', 'scenario'])
    d0_tot = merged['D0_Total'].where(merged['D0_Total'] != 0, 1.0)
    for c in comps: merged[c] = (merged[c] / d0_tot) * 100
    merged = scen_stats(merged, ['Service', 'Nationality'], comps)

    print("\n" + "="*140)
    print(f"{'Table 3: System-Weighted Contribution (pp) [Min, Max]':^140}")
    print("="*140)
    print(f"{'Sector':<12} {'Nationality':<15} {'Volume':>35} {'Behavior':>35} {'Structure':>35}")
    print("-" * 140)

    curr = ""
    for _, row in merged.iterrows():
        s = row['Service']
        n = row['Nationality'].replace('Foreign country', 'Foreign').replace('Austria', 'Austrian')

        def fmt_pp(name):
            med, c_min, c_max = row[name], row[f'{name}_min'], row[f'{name}_max']
            return f"{med:+.1f} [{min(c_min, c_max):+.1f}, {max(c_min, c_max):+.1f}]"

        if s != curr:
            if curr: print("-" * 140)
            curr = s
        print(f"{s:<12} {n:<15} {fmt_pp('Vol_Abs'):>35} {fmt_pp('Beh_Abs'):>35} {fmt_pp('Struct_Abs'):>35}")
    print("="*140 + "\n")

def table_4_m5_m6(df_main, df_pop):
    print("\n" + "="*200)
    print(f"{'Table 4: Pop Share, Demand Share, M6 (Fairness) and M5 (Sustainability) with Uncertainty':^200}")
    print("="*200)

    df_nat = df_main[df_main['Region_Code'] == 'NAT'].copy()
    if df_nat.empty: df_nat = df_main.copy()

    wf_data = {}
    if df_pop is not None:
        def clean_age(a):
            try: return int(str(a).replace('Y',''))
            except: return -1
        df_pop['Age_Num'] = df_pop['age'].apply(clean_age)
        wf_df = df_pop[(df_pop['Age_Num'] >= 20) & (df_pop['Age_Num'] <= 64)]
        wf_grp = wf_df.groupby(['TIME_PERIOD', 'Nationality', 'base_scenario'], observed=True)['OBS_VALUE'].sum()

        for year in [2025, 2050]:
            for nat in ['Austria', 'Foreign country']:
                try: wf_data[(year, nat)] = wf_grp.loc[(year, nat)]
                except: pass
            try: wf_data[(year, 'Total')] = wf_grp.loc[year].groupby(level='base_scenario').sum()
            except: pass

    df_agg = df_nat[df_nat['Year'].isin([2025, 2050])].copy()
    cols = ['Demand', 'Region_Pop']
    agg_nat = df_agg.groupby(['Service', 'Year', 'Nationality', 'scenario'])[cols].sum().reset_index()
    agg_tot = df_agg.groupby(['Service', 'Year', 'scenario'])[cols].sum().reset_index()
    agg_tot['Nationality'] = 'Total'
    full_agg = pd.concat([agg_nat, agg_tot], ignore_index=True)

    def fmt_rng(s, f, unit=''):
        return f"{f.format(s.median())}{unit} [{f.format(s.min())}, {f.format(s.max())}]"

    header = f"{'Sector':<12} {'Nationality':<15} {'Pop Share % [Min, Max]':>35} {'Demand Share % [Min, Max]':>35} {'M6 Fairness [Min, Max]':>35} {'M5 Sustainability [Min, Max]':>35}"
    print(header)
    print("-" * 200)

    services = full_agg['Service'].unique()
    for s in sorted(services):
        sub = full_agg[full_agg['Service']==s].set_index(['Year', 'Nationality', 'scenario']).sort_index()
        try:
            row_tot = sub.loc[(2050, 'Total')]
        except: continue

        for nat in ['Austria', 'Foreign country', 'Total']:
            if nat == 'Total':
                display_nat = 'TOTAL'
                m6_str = "1.00"
                pop_share_str = "100.0%"
                dem_share_str = "100.0%"
            else:
                display_nat = nat.replace('Foreign country', 'Foreign').replace('Austria', 'Austrian')
                try:
                    row_nat = sub.loc[(2050, nat)]

                    ps = (row_nat['Region_Pop'] / row_tot['Region_Pop']) * 100
                    pop_share_str = fmt_rng(ps, "{:.1f}", "%")

                    ds_ = (row_nat['Demand'] / row_tot['Demand']) * 100
                    dem_share_str = fmt_rng(ds_, "{:.1f}", "%")

                    m6_str = fmt_rng(ds_ / ps, "{:.2f}")
                except:
                    m6_str = "-"
                    pop_share_str = "-"
                    dem_share_str = "-"

            try:
                d25 = sub.loc[(2025, nat), 'Demand']
                d50 = sub.loc[(2050, nat), 'Demand'].reindex(d25.index)
                base = d25.index.str.split('--').str[0]
                w25 = pd.Series(wf_data[(2025, nat)].reindex(base).values, index=d25.index)
                w50 = pd.Series(wf_data[(2050, nat)].reindex(base).values, index=d25.index)

                if (d25 == 0).any() or w25.isna().any() or w50.isna().any(): raise ValueError

                ssr_25 = w25 / d25
                m5 = (w50/d50) / ssr_25 * 100

                m5_str = fmt_rng(m5, "{:.1f}")
            except: m5_str = "-"

            print(f"{s:<12} {display_nat:<15} {pop_share_str:>35} {dem_share_str:>35} {m6_str:>35} {m5_str:>35}")
        print("-" * 200)
    print("="*200 + "\n")

def table_regional_growth_summary(df):
    print("\n" + "="*160)
    print(f"{'Table 5: Regional Service Demand Growth by Sector (2025-2050) [Min, Max]':^160}")
    print("="*160)

    df_reg = df[df['Region_Code'] != 'NAT'].copy()

    df_2050 = df_reg[df_reg['Year'] == 2050].copy()
    pop_df = df_2050[df_2050['Service'] == 'Housing'].copy()

    pop_agg = pop_df.groupby(['NUTS2', 'scenario', 'Nationality'])['Region_Pop'].sum().unstack(fill_value=0)
    pop_agg = pop_agg.rename(columns={'Foreign country': 'For', 'Austria': 'Aut'})
    pop_agg['Share'] = (pop_agg['For'] / (pop_agg['Aut'] + pop_agg['For'])) * 100
    pop_agg = scen_stats(pop_agg.reset_index(), 'NUTS2', ['Share']).set_index('NUTS2')
    pop_agg['Share_Med'] = pop_agg['Share']

    def fmt_share(row):
        mn = min(row['Share_min'], row['Share_max'])
        mx = max(row['Share_min'], row['Share_max'])
        return f"{row['Share_Med']:.1f}% [{mn:.1f}, {mx:.1f}]"

    pop_agg['Foreign_Share_Str'] = pop_agg.apply(fmt_share, axis=1)

    df_growth = df_reg[df_reg['Year'].isin([2025, 2050])].copy()
    serv_agg = df_growth.groupby(['Service', 'NUTS2', 'scenario', 'Year'])['Demand'].sum().unstack()

    base = serv_agg[2025]
    serv_agg['Growth'] = ((serv_agg[2050] - base) / base) * 100
    serv_agg = scen_stats(serv_agg.reset_index(), ['Service', 'NUTS2'], ['Growth']).set_index(['Service', 'NUTS2'])
    serv_agg = serv_agg.rename(columns={'Growth_min': 'Growth_Min', 'Growth_max': 'Growth_Max'})

    def fmt_growth(row):
        return f"{row['Growth']:+.1f}% [{row['Growth_Min']:+.1f}, {row['Growth_Max']:+.1f}]"

    serv_agg['Str'] = serv_agg.apply(fmt_growth, axis=1)
    final_piv = serv_agg['Str'].unstack(level=0)

    final = pop_agg[['Foreign_Share_Str', 'Share_Med']].join(final_piv)
    final = final.rename(index=NUTS2_NAMES)
    final = final.sort_values('Share_Med', ascending=False)

    header = f"{'Region':<15} {'Foreign Pop Share [Range]':>30} {'Education Growth [Range]':>35} {'Housing Growth [Range]':>35} {'Healthcare Growth [Range]':>35}"
    print(header)
    print("-" * 160)

    for region, row in final.iterrows():
        fs = row['Foreign_Share_Str']
        edu = row.get('Education', '-')
        hou = row.get('Housing', '-')
        hlt = row.get('Healthcare', '-')
        print(f"{region:<15} {fs:>30} {edu:>35} {hou:>35} {hlt:>35}")
    print("="*160 + "\n")

def table_regional_contribution_waterfall(df):
    print("\n" + "="*140)
    print(f"{'Table 6: Contribution of Regions to National System Growth (pp) [Min, Max]':^140}")
    print("="*140)

    nat_2050 = df[(df['Year'] == 2050) & (df['Region_Code'] == 'NAT')].copy()
    if nat_2050.empty:
        national_d0 = df[df['Year'] == 2050].groupby(['Service', 'scenario'])['D0'].sum()
    else:
        national_d0 = nat_2050.groupby(['Service', 'scenario'])['D0'].sum()

    df_2050 = df[(df['Year'] == 2050) & (df['Region_Code'] != 'NAT')].copy()

    cols = ['Vol_Abs', 'Beh_Abs', 'Struct_Abs', 'Net_Abs']

    reg_agg = df_2050.groupby(['Service', 'NUTS2', 'scenario'])[cols].sum().reset_index()
    reg_agg = pd.merge(reg_agg, national_d0.rename('D0_Nat').reset_index(), on=['Service', 'scenario'], how='left')
    d0 = reg_agg['D0_Nat'].fillna(1).replace(0, 1.0)
    for c in cols: reg_agg[c] = (reg_agg[c] / d0) * 100
    reg_agg = scen_stats(reg_agg, ['Service', 'NUTS2'], cols)

    results = []
    for idx, row in reg_agg.iterrows():
        s = row['Service']

        def get_pp_str(name):
            med = row[name]
            c_min = row[f'{name}_min']
            c_max = row[f'{name}_max']
            return f"{med:+.1f} [{min(c_min, c_max):+.1f}, {max(c_min, c_max):+.1f}]"

        net_str = get_pp_str('Net_Abs')

        results.append({
            'Sector': s,
            'Region': NUTS2_NAMES.get(row['NUTS2'], row['NUTS2']),
            'Volume': get_pp_str('Vol_Abs'),
            'Behavior': get_pp_str('Beh_Abs'),
            'Structure': get_pp_str('Struct_Abs'),
            'Net': net_str,
            'Net_Val': row['Net_Abs']
        })

    res_df = pd.DataFrame(results)

    print(f"{'Sector':<12} {'Region':<15} {'Volume':>25} {'Behavior':>25} {'Structure':>25} {'Net Impact':>30}")
    print("-" * 140)

    for sector in ['Education', 'Healthcare', 'Housing']:
        subset = res_df[res_df['Sector'] == sector].sort_values('Net_Val', ascending=False)
        print(f"** {sector} **")
        for _, row in subset.iterrows():
            print(f"{'':<12} {row['Region']:<15} {row['Volume']:>25} {row['Behavior']:>25} {row['Structure']:>25} {row['Net']:>30}")
        print("-" * 140)
    print("="*140 + "\n")

def main():
    df, df_pop = load_data()
    if df is not None:
        table_1_summary(df)
        table_2_group_decomp(df)
        table_3_waterfall_text(df)
        table_4_m5_m6(df, df_pop)

        table_regional_growth_summary(df)
        table_regional_contribution_waterfall(df)


if __name__ == "__main__":
    main()
