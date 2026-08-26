#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Labour Supply — First-Order Projection

Combines the projected working-age (15-64) population from the demographic
model with 2023 LFS employment rates and sectoral shares:

    L_k(t) = N_wap_for(t) * e * s_k

where

    N_wap_for(t)  projected foreign-citizen population aged 15-64 in year t
                  (per scenario, from the decomposed projection)
    e             employment rate of foreign citizens (LFS 2023), applied by
                  five-year age band rather than as one 15-64 average, matching
                  the way the official Austrian labour-force projection combines
                  participation rates with a population projection; reported as
                  a band bounded by the recent-arrival rate (0-5 years of
                  residence) and the settled rate (>10 years)
    s_k           share of employed foreign citizens working in NACE section k
                  (LFS 2023)

The employed Austrian-national labour force is computed analogously (no
residence-duration band) as a comparison: the Austrian working-age population
contracts over the projection horizon, so the foreign labour force offsets a
decline rather than merely adding to a stable base.

Sectors of interest (matching the paper's three service sectors):
    P  Education            -> education demand
    Q  Human health/social  -> healthcare demand
    F  Construction         -> housing supply

The band-specific rates come from 01b/02b, which are extracted from the same
microcensus file as 01/02 and whose band-weighted aggregates reproduce the
15-64 rates those files report.

Pipeline:
  1. Load the projection, extract WAP per year x scenario x nationality x band
  2. Load labour-market inputs (Data/export_labour_inputs/)
  3. Compute employed persons (foreign total/sectors, Austrian, combined)
  4. Export summary table (XLSX/) and figure (PNG/figures_methodology/)
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import config

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

BASE_YEAR = 2025
DEMAND_YEAR = 2050

PATH_PROJECTION = config.PATH_PROJECTION
PATH_LABOUR_DIR = r'Data/export_labour_inputs'

# Outputs
PATH_SUMMARY_CSV = r'XLSX/labour_force_contribution_summary.csv'
FIG_DIR = r'PNG/figures_methodology'
FIG_STEM = 'labour_force_contribution'

# NACE Rev. 2 sections matching the paper's three service sectors
TARGET_SECTORS = {
    'P': 'Education',
    'Q': 'Health & social work',
    'F': 'Construction',
}

# Aggregate row labels
FOREIGN_TOTAL = 'Foreign total'
AUSTRIAN_TOTAL = 'Austrian total'
AUSTRIAN_SECTOR = 'Austrian {}'
ALL_TOTAL = 'All nationalities'

# Five-year bands, matching xbalt5 in the microcensus and the age grouping of
# the official labour-force projection.
BANDS = ['15-19', '20-24', '25-29', '30-34', '35-39',
         '40-44', '45-49', '50-54', '55-59', '60-64']
BAND_CUTS = [14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64]

# Paper palette (Okabe-Ito)
COLOR_AUSTRIAN = '#009E73'
COLOR_FOREIGN = '#CC79A7'

# Sector colours and markers match the regional analysis, so each sector reads
# the same colour across the paper's figures. NACE F is the supply side of
# housing, so it takes the Housing style. Markers double as secondary encoding,
# so the panel never depends on colour alone.
SECTOR_STYLE = {
    'Construction': ('tab:orange', 'X'),           # Housing
    'Education': ('tab:blue', 'o'),                # Education
    'Health & social work': ('tab:green', '^'),    # Healthcare
}


# ==========================================
# 2. LOADERS
# ==========================================

def convert_age(age_str) -> int:
    """Convert Eurostat-style age strings (Y_GE100, Y_LT1, Y25, ...) to int."""
    if age_str == 'Y_GE100':
        return 100
    elif age_str == 'Y_LT1':
        return 0
    else:
        return int(age_str.lstrip('Y'))


def load_wap(projected_path: str) -> pd.DataFrame:
    """
    Load the decomposed projection and return the working-age (15-64)
    population per year x scenario x nationality x five-year band.
    """
    df = pd.read_parquet(
        projected_path,
        columns=['TIME_PERIOD', 'age', 'Nationality', 'scenario', 'OBS_VALUE'],
    )
    age_map = {a: convert_age(a) for a in df['age'].unique()}
    df['age_num'] = df['age'].map(age_map)
    df = df[df['age_num'].between(15, 64)]

    # Same negative-population clip as the sector demand models
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
    """
    Load employment rates and sectoral shares (LFS 2023, own calculations).

    Returns a dict with:
        e_central   employment rate, all foreign citizens 15-64
        e_low       employment rate, foreign citizens 0-5 years of residence
        e_high      employment rate, foreign citizens >10 years of residence
        e_national  employment rate, Austrian citizens 15-64
        e_band      {variant: {band: rate}} for foreign citizens
        e_band_at   {band: rate} for Austrian citizens
        shares      {NACE section: share of foreign employment}
        shares_at   {NACE section: share of Austrian employment}
    """
    rates = pd.read_csv(os.path.join(labour_dir, '01_employment_rates_citizenship.csv'))
    duration = pd.read_csv(os.path.join(labour_dir, '02_employment_rate_by_residence_duration.csv'))
    sectors = pd.read_csv(os.path.join(labour_dir, '03a_foreign_employment_by_sector.csv'))
    within = pd.read_csv(os.path.join(labour_dir, '03b_foreign_share_within_sector.csv'))
    age_rates = pd.read_csv(os.path.join(labour_dir, '01b_employment_rates_by_age.csv'))
    age_duration = pd.read_csv(os.path.join(labour_dir, '02b_employment_rate_by_age_and_residence.csv'))

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

    # Austrian-national sectoral shares, the mirror of s_k. 03b gives the
    # weighted foreign and total headcount per NACE section, so the national
    # side is the difference. TOTAL is an aggregate row, not a section, and
    # would double the denominator if left in.
    sec = within[(within['year'] == 2023) &
                 (within['nace_section'] != 'TOTAL')].copy()
    sec['national_count'] = sec['total_count'] - sec['foreign_count']
    nat_total = sec['national_count'].sum()
    for_total = sec['foreign_count'].sum()
    shares_at = (sec.set_index('nace_section')['national_count'] / nat_total).to_dict()
    shares_at = {k: shares_at[k] for k in TARGET_SECTORS}

    # Self-check: s_k and s_k^AT must reproduce the observed 2023 within-sector
    # foreign shares that 03b reports directly.
    for nace in TARGET_SECTORS:
        obs = float(sec.loc[sec['nace_section'] == nace, 'foreign_share'].iloc[0])
        foreign_k = shares[nace] * for_total
        austrian_k = shares_at[nace] * nat_total
        assert abs(foreign_k / (foreign_k + austrian_k) - obs) < 5e-3, \
            f'2023 within-sector share not reproduced for NACE {nace}'

    # Age-specific rates. The low and high variants are the residence-duration
    # profiles; the Austrian profile has no duration dimension.
    rate_rows = age_rates[age_rates['indicator'] == 'employment_rate']
    duration_rows = age_duration[age_duration['indicator'] == 'employment_rate']
    e_band = {
        'central': dict(zip(rate_rows[rate_rows['group'] == 'Foreign_total']['age_group'],
                            rate_rows[rate_rows['group'] == 'Foreign_total']['value'])),
        'low': dict(zip(duration_rows[duration_rows['residence_duration'] == '0-5']['age_group'],
                        duration_rows[duration_rows['residence_duration'] == '0-5']['value'])),
        'high': dict(zip(duration_rows[duration_rows['residence_duration'] == '>10']['age_group'],
                         duration_rows[duration_rows['residence_duration'] == '>10']['value'])),
    }
    e_band_at = dict(zip(rate_rows[rate_rows['group'] == 'National']['age_group'],
                         rate_rows[rate_rows['group'] == 'National']['value']))
    for name, band_rates in list(e_band.items()) + [('national', e_band_at)]:
        missing = set(BANDS) - set(band_rates)
        assert not missing, f'Age bands missing from {name}: {sorted(missing)}'

    return {'e_central': e_central, 'e_low': e_low, 'e_high': e_high,
            'e_national': e_national, 'shares': shares, 'shares_at': shares_at,
            'nat_total_2023': nat_total, 'for_total_2023': for_total,
            'e_band': e_band, 'e_band_at': e_band_at}


# ==========================================
# 3. CONTRIBUTION CALCULATION
# ==========================================

def compute_contribution(wap: pd.DataFrame, labour: dict) -> pd.DataFrame:
    """
    Compute employed persons per year x scenario x sector, under the three
    foreign employment-rate variants (low / central / high).

    Sectors:
      'Foreign total'      foreign WAP x e_band
      NACE F/P/Q labels    foreign total x s_k
      'Austrian total'     Austrian WAP x e_band_at (identical across variants)
      'Austrian <sector>'  Austrian total x s_k^AT
      'All nationalities'  Austrian total + Foreign total

    Returns a long DataFrame: [Year, scenario, sector, e_variant, employed]
    """
    wap_for = wap[wap['Nationality'] == 'Foreign country']
    wap_at = wap[wap['Nationality'] == 'Austria']

    def employed_by_band(frame, band_rates):
        """Apply a band-specific rate, then collapse the band dimension."""
        applied = frame.copy()
        applied['employed'] = (applied['N_wap']
                               * applied['band'].astype(str).map(band_rates))
        assert applied['employed'].notna().all(), 'Unmapped age band'
        return applied.groupby(['Year', 'scenario'], as_index=False)['employed'].sum()

    records = []
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

        # Austrian nationals: age profile only, no residence-duration band
        austrian = employed_by_band(wap_at, labour['e_band_at'])
        austrian['e_variant'] = e_name
        austrian['sector'] = AUSTRIAN_TOTAL
        records.append(austrian)

        # Combined total employment
        combined = foreign[['Year', 'scenario', 'employed']].merge(
            austrian[['Year', 'scenario', 'employed']],
            on=['Year', 'scenario'], suffixes=('_for', '_at'),
        )
        combined['e_variant'] = e_name
        combined['sector'] = ALL_TOTAL
        combined['employed'] = combined['employed_for'] + combined['employed_at']
        records.append(combined)

        # Austrian nationals by sector, so the foreign share *within* each
        # sector's workforce can be formed for every projection year rather
        # than only for the 2023 base.
        for nace, label in TARGET_SECTORS.items():
            sec = austrian.copy()
            sec['sector'] = AUSTRIAN_SECTOR.format(label)
            sec['employed'] = sec['employed'] * labour['shares_at'][nace]
            records.append(sec)

    out = pd.concat(records, ignore_index=True)
    return out[['Year', 'scenario', 'sector', 'e_variant', 'employed']]


def summarise(contrib: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise across the scenario ensemble:
      central  = median across scenarios at the central employment rate
      lower    = scenario minimum at the recent-arrival rate (outer envelope)
      upper    = scenario maximum at the settled rate (outer envelope)

    For 'Austrian total' the three variants coincide, so the envelope reduces
    to the scenario range.
    """
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


# ==========================================
# 4. FIGURE
# ==========================================

def plot_contribution(summary: pd.DataFrame) -> plt.Figure:
    """
    Two-panel figure:
      (a) employed Austrian vs. foreign nationals (15-64)
      (b) foreign employment in the three service-relevant sectors
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel a: employed persons by citizenship ---
    aut = summary[summary['sector'] == AUSTRIAN_TOTAL].sort_values('Year')
    ax1.fill_between(aut['Year'], aut['employed_lower'], aut['employed_upper'],
                     color=COLOR_AUSTRIAN, alpha=0.25,
                     label='Austrian (scenario range)')
    ax1.plot(aut['Year'], aut['employed_central_median'],
             color=COLOR_AUSTRIAN, lw=2, label='Austrian (median)')

    forn = summary[summary['sector'] == FOREIGN_TOTAL].sort_values('Year')
    ax1.fill_between(forn['Year'], forn['employed_lower'], forn['employed_upper'],
                     color=COLOR_FOREIGN, alpha=0.12,
                     label='Foreign (scenario range × rate band)')
    ax1.fill_between(forn['Year'], forn['employed_central_min'],
                     forn['employed_central_max'],
                     color=COLOR_FOREIGN, alpha=0.25,
                     label='Foreign (scenario range)')
    ax1.plot(forn['Year'], forn['employed_central_median'],
             color=COLOR_FOREIGN, lw=2, label='Foreign (median)')

    ax1.set_title('a  Employed persons (15–64) by citizenship', loc='left')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Employed persons (millions)')
    ax1.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}M'))
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(frameon=False, fontsize=9, loc='center left')

    # --- Panel b: foreign employment by sector ---
    for label, (color, marker) in SECTOR_STYLE.items():
        sec = summary[summary['sector'] == label].sort_values('Year')
        ax2.fill_between(sec['Year'], sec['employed_lower'],
                         sec['employed_upper'], color=color, alpha=0.15)
        ax2.plot(sec['Year'], sec['employed_central_median'],
                 color=color, lw=2, label=label)
        # Thin the markers, keep the line dense
        step = max(1, len(sec) // 12)
        ax2.plot(sec['Year'].iloc[::step],
                 sec['employed_central_median'].iloc[::step],
                 linestyle='none', marker=marker, markersize=6, color=color,
                 markeredgecolor='white', markeredgewidth=0.8)
        # Direct label at the right end (identity not colour-alone)
        ax2.annotate(
            label,
            xy=(sec['Year'].iloc[-1], sec['employed_central_median'].iloc[-1]),
            xytext=(4, 0), textcoords='offset points',
            va='center', fontsize=9, color=color,
        )

    ax2.set_title('b  Foreign employment by service-relevant sector', loc='left')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Employed persons (thousands)')
    ax2.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f'{x * 1e-3:,.0f}k'))
    ax2.set_xlim(ax2.get_xlim()[0], DEMAND_YEAR + 7)  # room for direct labels
    ax2.set_xticks(np.arange(BASE_YEAR, DEMAND_YEAR + 1, 5))
    ax2.grid(True, linestyle='--', alpha=0.4)
    # Legend keyed on the sector MARKER (matching the series and the regional
    # figures), not a bare colour line.
    sector_handles = [
        plt.Line2D([], [], color=color, marker=marker, markersize=8,
                   markeredgecolor='white', markeredgewidth=0.8,
                   linewidth=2.0, label=label)
        for label, (color, marker) in SECTOR_STYLE.items()
    ]
    ax2.legend(handles=sector_handles, frameon=False, fontsize=9,
               loc='upper left')

    fig.tight_layout()
    return fig


def save_figure(fig, stem: str, directory: str = FIG_DIR) -> None:
    """Write the figure as PNG and PDF."""
    os.makedirs(directory, exist_ok=True)
    for path, kwargs in ((os.path.join(directory, f'{stem}.png'), {'dpi': 300}),
                         (os.path.join(directory, f'{stem}.pdf'), {})):
        fig.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved figure to: {path}')


# ==========================================
# 5. REPORTING
# ==========================================

def report_inputs(labour: dict) -> None:
    """Print the LFS 2023 inputs, so the figure can be traced back to them."""
    print('\nLabour inputs (LFS 2023):')
    print('  employment rate by five-year band (%):')
    print(f"    {'band':>7} {'AT':>6} {'FOR':>6} {'FOR 0-5':>8} {'FOR >10':>8}")
    for band in BANDS:
        print(f"    {band:>7} {100 * labour['e_band_at'][band]:6.1f} "
              f"{100 * labour['e_band']['central'][band]:6.1f} "
              f"{100 * labour['e_band']['low'][band]:8.1f} "
              f"{100 * labour['e_band']['high'][band]:8.1f}")
    print(f"  flat 15-64 equivalents: AT {labour['e_national']:.3f}, "
          f"FOR {labour['e_central']:.3f} "
          f"({labour['e_low']:.3f}-{labour['e_high']:.3f})")
    for nace, label in TARGET_SECTORS.items():
        print(f"  s_{nace} ({label}): foreign {labour['shares'][nace]:.4f}   "
              f"Austrian {labour['shares_at'][nace]:.4f}")
    print(f"  2023 employed: national {labour['nat_total_2023'] / 1e3:,.1f}k   "
          f"foreign {labour['for_total_2023'] / 1e3:,.1f}k   "
          f"total {(labour['nat_total_2023'] + labour['for_total_2023']) / 1e3:,.1f}k")


def report_headlines(summary: pd.DataFrame) -> None:
    """Print the quantities the manuscript reports."""
    print('\n--- Headline numbers (thousands of employed persons) ---')
    row_order = ([AUSTRIAN_TOTAL, FOREIGN_TOTAL, ALL_TOTAL]
                 + list(TARGET_SECTORS.values()))
    for year in [2025, 2035, 2050]:
        print(f'\n{year}:')
        for sector in row_order:
            row = summary[(summary['Year'] == year)
                          & (summary['sector'] == sector)].iloc[0]
            print(f"  {sector:<22} median {row['employed_central_median'] / 1e3:7.1f}k   "
                  f"band [{row['employed_lower'] / 1e3:7.1f}k - "
                  f"{row['employed_upper'] / 1e3:7.1f}k]")

    print('\n--- Derived quantities (central employment rates) ---')
    for sector in [AUSTRIAN_TOTAL, FOREIGN_TOTAL, ALL_TOTAL]:
        v25 = summary[(summary['Year'] == 2025)
                      & (summary['sector'] == sector)]['employed_central_median'].iloc[0]
        v50 = summary[(summary['Year'] == 2050)
                      & (summary['sector'] == sector)]['employed_central_median'].iloc[0]
        print(f'  {sector:<22} 2025-2050 median growth: {100 * (v50 / v25 - 1):+.1f}%')

    for year in [2025, 2050]:
        foreign = summary[(summary['Year'] == year)
                          & (summary['sector'] == FOREIGN_TOTAL)]['employed_central_median'].iloc[0]
        total = summary[(summary['Year'] == year)
                        & (summary['sector'] == ALL_TOTAL)]['employed_central_median'].iloc[0]
        print(f'  Foreign share of total employment {year}: {100 * foreign / total:.1f}%')

    # Foreign share *within* each sector's workforce: same year on both sides,
    # unlike the 2023-observed shares. The envelope pairs the extreme foreign
    # value against the opposite Austrian extreme.
    print('\n--- Foreign share within each sector workforce (same year) ---')
    for year in [2025, 2035, 2050]:
        print(f'\n{year}:')
        for label in TARGET_SECTORS.values():
            foreign = summary[(summary['Year'] == year)
                              & (summary['sector'] == label)].iloc[0]
            austrian = summary[(summary['Year'] == year)
                               & (summary['sector'] == AUSTRIAN_SECTOR.format(label))].iloc[0]
            mid = foreign['employed_central_median'] / (
                foreign['employed_central_median'] + austrian['employed_central_median'])
            low = foreign['employed_lower'] / (
                foreign['employed_lower'] + austrian['employed_upper'])
            high = foreign['employed_upper'] / (
                foreign['employed_upper'] + austrian['employed_lower'])
            print(f'  {label:<22} {100 * mid:5.1f}%   '
                  f'band [{100 * low:5.1f}% - {100 * high:5.1f}%]')


# ==========================================
# 6. MAIN
# ==========================================

def main() -> None:
    print('Loading projected working-age population...')
    wap = load_wap(PATH_PROJECTION)
    print(f'  {wap["scenario"].nunique()} scenarios, '
          f'years {wap["Year"].min()}-{wap["Year"].max()}')

    # Base-year consistency: all scenarios must coincide in BASE_YEAR
    for nationality in ['Foreign country', 'Austria']:
        base = (wap[(wap['Year'] == BASE_YEAR)
                    & (wap['Nationality'] == nationality)]
                .groupby('scenario')['N_wap'].sum())
        assert base.round(0).nunique() == 1, \
            f'Scenarios diverge in base year ({nationality})'
        print(f'  {nationality} WAP {BASE_YEAR}: {base.iloc[0]:,.0f} '
              f'(identical across scenarios)')

    labour = load_labour_inputs(PATH_LABOUR_DIR)
    report_inputs(labour)

    contrib = compute_contribution(wap, labour)
    summary = summarise(contrib)

    summary.round(0).to_csv(PATH_SUMMARY_CSV, index=False)
    print(f'\nSaved summary to: {PATH_SUMMARY_CSV}')

    report_headlines(summary)

    print()
    fig = plot_contribution(summary)
    save_figure(fig, FIG_STEM)


if __name__ == '__main__':
    main()
