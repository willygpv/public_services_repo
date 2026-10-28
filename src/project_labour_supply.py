"""Employed persons by citizenship and sector, from the projected working-age population and the 2023 employment rates."""

import os

import config

import pandas as pd


BASE_YEAR = 2025
DEMAND_YEAR = 2050

PROJECTED_PATH = config.PATH_PROJECTION
LABOUR_DIR = r'Data/export_labour_inputs'

TARGET_SECTORS = {
    'P': 'Education',
    'Q': 'Health & social work',
    'F': 'Construction',
}

FOREIGN_TOTAL = 'Foreign total'
AUSTRIAN_TOTAL = 'Austrian total'
AUSTRIAN_SECTOR = 'Austrian {}'

BANDS = ['15-19', '20-24', '25-29', '30-34', '35-39',
         '40-44', '45-49', '50-54', '55-59', '60-64']
BAND_CUTS = [14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64]
ALL_TOTAL = 'All nationalities'

OUT_CSV = r'XLSX/labour_force_contribution_summary.csv'


def convert_age(age_str) -> int:
    if age_str == 'Y_GE100':
        return 100
    elif age_str == 'Y_LT1':
        return 0
    else:
        return int(age_str.lstrip('Y'))


def load_wap(projected_path: str) -> pd.DataFrame:
    df = pd.read_parquet(
        projected_path,
        columns=['TIME_PERIOD', 'age', 'Nationality', 'scenario', 'OBS_VALUE'],
    )
    age_map = {a: convert_age(a) for a in df['age'].unique()}
    df['age_num'] = df['age'].map(age_map)
    df = df[df['age_num'].between(15, 64)]

    df['OBS_VALUE'] = df['OBS_VALUE'].clip(lower=0)

    df['band'] = pd.cut(df['age_num'], bins=BAND_CUTS, labels=BANDS)

    wap = (
        df.groupby(['TIME_PERIOD', 'scenario', 'Nationality', 'band'],
                   as_index=False, observed=True)['OBS_VALUE']
        .sum()
        .rename(columns={'TIME_PERIOD': 'Year', 'OBS_VALUE': 'N_wap'})
    )
    wap = wap[(wap['Year'] >= BASE_YEAR) & (wap['Year'] <= DEMAND_YEAR)]
    return wap


def load_labour_inputs(labour_dir: str) -> dict:
    rates = pd.read_csv(os.path.join(labour_dir, '01_employment_rates_citizenship.csv'))
    duration = pd.read_csv(os.path.join(labour_dir, '02_employment_rate_by_residence_duration.csv'))
    sectors = pd.read_csv(os.path.join(labour_dir, '03a_foreign_employment_by_sector.csv'))
    within = pd.read_csv(os.path.join(labour_dir, '03b_foreign_share_within_sector.csv'))
    age_r = pd.read_csv(os.path.join(labour_dir, '01b_employment_rates_by_age.csv'))
    age_d = pd.read_csv(os.path.join(labour_dir, '02b_employment_rate_by_age_and_residence.csv'))

    def pick(df, **conds):
        mask = pd.Series(True, index=df.index)
        for col, val in conds.items():
            mask &= df[col] == val
        vals = df.loc[mask, 'value']
        assert len(vals) == 1, f"Expected 1 row for {conds}, got {len(vals)}"
        return float(vals.iloc[0])

    e_central = pick(rates, group='Foreign_total', sex='total',
                     indicator='employment_rate')
    e_national = pick(rates, group='National', sex='total',
                      indicator='employment_rate')
    e_low = pick(duration, group='Foreign_total', residence_duration='0-5',
                 sex='total', indicator='employment_rate')
    e_high = pick(duration, group='Foreign_total', residence_duration='>10',
                  sex='total', indicator='employment_rate')

    shares = (
        sectors[sectors['nace_section'].isin(TARGET_SECTORS)]
        .set_index('nace_section')['share_of_foreign_employment']
        .to_dict()
    )
    assert set(shares) == set(TARGET_SECTORS), "Missing NACE sections in 03a"
    assert e_low < e_central < e_high, "Employment-rate band is not ordered"

    sec = within[(within['year'] == 2023) &
                 (within['nace_section'] != 'TOTAL')].copy()
    sec['national_count'] = sec['total_count'] - sec['foreign_count']
    nat_total = sec['national_count'].sum()
    for_total = sec['foreign_count'].sum()
    shares_at = (sec.set_index('nace_section')['national_count'] / nat_total).to_dict()
    shares_at = {k: shares_at[k] for k in TARGET_SECTORS}

    for nace in TARGET_SECTORS:
        obs = float(sec.loc[sec['nace_section'] == nace, 'foreign_share'].iloc[0])
        f = shares[nace] * for_total
        a = shares_at[nace] * nat_total
        assert abs(f / (f + a) - obs) < 5e-3,             f'2023 within-sector share not reproduced for NACE {nace}'

    ar = age_r[age_r['indicator'] == 'employment_rate']
    ad = age_d[age_d['indicator'] == 'employment_rate']
    e_band = {
        'central': dict(zip(ar[ar['group'] == 'Foreign_total']['age_group'],
                            ar[ar['group'] == 'Foreign_total']['value'])),
        'low': dict(zip(ad[ad['residence_duration'] == '0-5']['age_group'],
                        ad[ad['residence_duration'] == '0-5']['value'])),
        'high': dict(zip(ad[ad['residence_duration'] == '>10']['age_group'],
                         ad[ad['residence_duration'] == '>10']['value'])),
    }
    e_band_at = dict(zip(ar[ar['group'] == 'National']['age_group'],
                         ar[ar['group'] == 'National']['value']))
    for name, d in list(e_band.items()) + [('national', e_band_at)]:
        missing = set(BANDS) - set(d)
        assert not missing, f'Age bands missing from {name}: {sorted(missing)}'

    return {'e_central': e_central, 'e_low': e_low, 'e_high': e_high,
            'e_national': e_national, 'shares': shares, 'shares_at': shares_at,
            'nat_total_2023': nat_total, 'for_total_2023': for_total,
            'e_band': e_band, 'e_band_at': e_band_at}


def compute_contribution(wap: pd.DataFrame, labour: dict) -> pd.DataFrame:
    wap_for = wap[wap['Nationality'] == 'Foreign country']
    wap_at = wap[wap['Nationality'] == 'Austria']

    records = []

    def employed_by_band(frame, band_rates):
        f = frame.copy()
        f['employed'] = f['N_wap'] * f['band'].astype(str).map(band_rates)
        assert f['employed'].notna().all(), 'Unmapped age band'
        return f.groupby(['Year', 'scenario'], as_index=False)['employed'].sum()

    for e_name in ('low', 'central', 'high'):
        foreign = employed_by_band(wap_for, labour['e_band'][e_name])
        foreign['e_variant'] = e_name
        foreign['sector'] = FOREIGN_TOTAL
        records.append(foreign)

        for nace, label in TARGET_SECTORS.items():
            sec = foreign.copy()
            sec['sector'] = label
            sec['employed'] = sec['employed'] * labour['shares'][nace]
            records.append(sec)

        austrian = employed_by_band(wap_at, labour['e_band_at'])
        austrian['e_variant'] = e_name
        austrian['sector'] = AUSTRIAN_TOTAL
        records.append(austrian)

        combined = foreign[['Year', 'scenario', 'employed']].merge(
            austrian[['Year', 'scenario', 'employed']],
            on=['Year', 'scenario'], suffixes=('_for', '_at'),
        )
        combined['e_variant'] = e_name
        combined['sector'] = ALL_TOTAL
        combined['employed'] = combined['employed_for'] + combined['employed_at']
        records.append(combined)

        for nace, label in TARGET_SECTORS.items():
            sec = austrian.copy()
            sec['sector'] = AUSTRIAN_SECTOR.format(label)
            sec['employed'] = sec['employed'] * labour['shares_at'][nace]
            records.append(sec)

    out = pd.concat(records, ignore_index=True)
    return out[['Year', 'scenario', 'sector', 'e_variant', 'employed']]


def summarise(contrib: pd.DataFrame) -> pd.DataFrame:
    central = (
        contrib[contrib['e_variant'] == 'central']
        .groupby(['Year', 'sector'])['employed']
        .agg(employed_central_median='median',
             employed_central_min='min',
             employed_central_max='max')
    )
    lower = (
        contrib[contrib['e_variant'] == 'low']
        .groupby(['Year', 'sector'])['employed'].min()
        .rename('employed_lower')
    )
    upper = (
        contrib[contrib['e_variant'] == 'high']
        .groupby(['Year', 'sector'])['employed'].max()
        .rename('employed_upper')
    )
    return pd.concat([central, lower, upper], axis=1).reset_index()


if __name__ == '__main__':
    print('Loading projected working-age population...')
    wap = load_wap(PROJECTED_PATH)
    n_scen = wap['scenario'].nunique()
    print(f'  {n_scen} scenarios, years {wap["Year"].min()}-{wap["Year"].max()}')

    for nat in ['Foreign country', 'Austria']:
        base = (wap[(wap['Year'] == BASE_YEAR) & (wap['Nationality'] == nat)]
                .groupby('scenario')['N_wap'].sum())
        assert base.round(0).nunique() == 1, f'Scenarios diverge in base year ({nat})'
        print(f'  {nat} WAP {BASE_YEAR}: {base.iloc[0]:,.0f} (identical across scenarios)')

    labour = load_labour_inputs(LABOUR_DIR)
    print(f"\nLabour inputs (LFS 2023):")
    print('  employment rate by five-year band (%):')
    print(f"    {'band':>7} {'AT':>6} {'FOR':>6} {'FOR 0-5':>8} {'FOR >10':>8}")
    for b in BANDS:
        print(f"    {b:>7} {100*labour['e_band_at'][b]:6.1f} "
              f"{100*labour['e_band']['central'][b]:6.1f} "
              f"{100*labour['e_band']['low'][b]:8.1f} "
              f"{100*labour['e_band']['high'][b]:8.1f}")
    print(f"  flat 15-64 equivalents: AT {labour['e_national']:.3f}, "
          f"FOR {labour['e_central']:.3f} "
          f"({labour['e_low']:.3f}-{labour['e_high']:.3f})")
    for nace, label in TARGET_SECTORS.items():
        print(f"  s_{nace} ({label}): foreign {labour['shares'][nace]:.4f}   "
              f"Austrian {labour['shares_at'][nace]:.4f}")
    print(f"  2023 employed: national {labour['nat_total_2023']/1e3:,.1f}k   "
          f"foreign {labour['for_total_2023']/1e3:,.1f}k   "
          f"total {(labour['nat_total_2023']+labour['for_total_2023'])/1e3:,.1f}k")

    contrib = compute_contribution(wap, labour)
    summary = summarise(contrib)

    summary.round(0).to_csv(OUT_CSV, index=False)
    print(f'\nSaved summary to: {OUT_CSV}')

    print('\n--- Headline numbers (thousands of employed persons) ---')
    row_order = [AUSTRIAN_TOTAL, FOREIGN_TOTAL, ALL_TOTAL] + list(TARGET_SECTORS.values())
    for year in [2025, 2035, 2050]:
        print(f'\n{year}:')
        for sector in row_order:
            row = summary[(summary['Year'] == year) & (summary['sector'] == sector)].iloc[0]
            print(f"  {sector:<22} median {row['employed_central_median']/1e3:7.1f}k   "
                  f"band [{row['employed_lower']/1e3:7.1f}k - {row['employed_upper']/1e3:7.1f}k]")

    print('\n--- Derived quantities (central employment rates) ---')
    for sector in [AUSTRIAN_TOTAL, FOREIGN_TOTAL, ALL_TOTAL]:
        v25 = summary[(summary['Year'] == 2025) & (summary['sector'] == sector)][
            'employed_central_median'].iloc[0]
        v50 = summary[(summary['Year'] == 2050) & (summary['sector'] == sector)][
            'employed_central_median'].iloc[0]
        print(f'  {sector:<22} 2025-2050 median growth: {100 * (v50 / v25 - 1):+.1f}%')

    for year in [2025, 2050]:
        f = summary[(summary['Year'] == year) & (summary['sector'] == FOREIGN_TOTAL)][
            'employed_central_median'].iloc[0]
        t = summary[(summary['Year'] == year) & (summary['sector'] == ALL_TOTAL)][
            'employed_central_median'].iloc[0]
        print(f'  Foreign share of total employment {year}: {100 * f / t:.1f}%')

    print()
    print('--- Foreign share within each sector workforce (same year) ---')
    for year in [2025, 2035, 2050]:
        print()
        print(f'{year}:')
        for nace, label in TARGET_SECTORS.items():
            fr = summary[(summary['Year'] == year) &
                         (summary['sector'] == label)].iloc[0]
            ar = summary[(summary['Year'] == year) &
                         (summary['sector'] == AUSTRIAN_SECTOR.format(label))].iloc[0]
            mid = fr['employed_central_median'] / (
                fr['employed_central_median'] + ar['employed_central_median'])
            lo = fr['employed_lower'] / (fr['employed_lower'] + ar['employed_upper'])
            hi = fr['employed_upper'] / (fr['employed_upper'] + ar['employed_lower'])
            print(f'  {label:<22} {100*mid:5.1f}%   '
                  f'band [{100*lo:5.1f}% - {100*hi:5.1f}%]')
