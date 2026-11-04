"""Sensitivity of demand to the utilisation path: outpatient care, class sizes, household formation.

Three of the four scenarios of Supplementary Table 9: bed-day rates -10% by 2050 (needs the
hospital file), student-to-teacher ratio +10% by 2050, and household formation rate at ages
20-34 -10% by 2050. The fourth, doubled naturalisation, has its own run
(run_sensitivity_naturalisation.sh).

Run from the repository root after run_pipeline.sh. Reads the sector outputs, writes nothing
and prints the results. Every statistic is computed per scenario, then summarised as the
median [min, max] over the 54 scenarios (trend paths for housing and education). The
representation index (RI) is the foreign share of demand divided by the foreign share of the
population.
"""

import os

import pandas as pd
import pyarrow.dataset as ds

import config

Y0, Y1 = 2025, 2050
FOR, AUT = 'Foreign country', 'Austria'

H1_F50 = 0.90        # bed-day rates -10% by 2050
E1_F50 = 1 / 1.10    # student-to-teacher ratio +10% by 2050, so teachers / 1.1
D1_F50 = 0.90        # household formation rate at ages 20-34 -10% by 2050


def base(s):
    return s.astype(str).str.split('--').str[0]


def pop():
    t = ds.dataset(config.PATH_PROJECTION).to_table(
        columns=['TIME_PERIOD', 'Nationality', 'scenario', 'OBS_VALUE'],
        filter=ds.field('TIME_PERIOD').isin([Y0, Y1])).to_pandas()
    t['Nationality'] = t.Nationality.astype(str).str.strip()
    g = t.groupby(['scenario', 'TIME_PERIOD', 'Nationality'])['OBS_VALUE'].sum().unstack('Nationality')
    g.index = g.index.set_names(['base', 'Year'])
    return g


def load(path, ycol, vcol, sfx, extra=()):
    cols = ['scenario', ycol, 'Nationality', vcol, *extra]
    t = ds.dataset(path).to_table(columns=cols, filter=ds.field(ycol).isin([Y0, Y1])).to_pandas()
    t = t.rename(columns={ycol: 'Year', vcol: 'D'})
    for c in ['scenario', 'Nationality']:
        t[c] = t[c].astype(str).str.strip()
    if sfx is not None:
        t = t[t.scenario.str.endswith(sfx)]
    t['base'] = base(t.scenario)
    return t[t.base.str.startswith('BSL-')]


def stats(D, P):
    """D: index (base, Year), columns Austrian and foreign demand. Per-scenario statistics."""
    tot = D.sum(axis=1).unstack('Year')
    share = (D[FOR] / D.sum(axis=1)).unstack('Year')
    out = pd.DataFrame({
        'growth': (tot[Y1] / tot[Y0] - 1) * 100,
        'for_share_2025': share[Y0] * 100,
        'for_share_2050': share[Y1] * 100,
    })
    ps = (P[FOR] / P.sum(axis=1)).unstack('Year')
    out['RI_2050'] = out.for_share_2050 / 100 / ps.loc[out.index, Y1]
    return out


def fmt(s, d=3):
    return f'{s.median():.{d}f} [{s.min():.{d}f}, {s.max():.{d}f}]'


def report(name, o):
    print(f'  {name:<28} n={len(o):>2}  growth {fmt(o.growth)}  for.share 2025 {fmt(o.for_share_2025)}'
          f'  2050 {fmt(o.for_share_2050)}  RI 2050 {fmt(o.RI_2050, 5)}')


P = pop()

if os.path.exists(config.PATH_HEALTHCARE_DEMAND):
    h = load(config.PATH_HEALTHCARE_DEMAND, 'year', 'Bed_Days_Mean', None)
    H = h.groupby(['base', 'Year', 'Nationality'])['D'].sum().unstack('Nationality')
    H1 = H.copy()
    H1.loc[(slice(None), Y1), :] *= H1_F50
    print('HEALTHCARE (bed-days)')
    report('main projection', stats(H, P))
    report('bed-day rates -10%', stats(H1, P))
    # uniform 2050 rate decline that holds total bed-days at the 2025 level
    tot = H.sum(axis=1).unstack('Year')
    print(f'  break-even decline by 2050 (%): {fmt((1 - tot[Y0] / tot[Y1]) * 100)}')
    # behaviour term in percentage points of 2025 demand (volume and ageing use base-year rates)
    d0 = H.xs(Y0, level='Year').sum(axis=1)
    beh = (H1 - H).xs(Y1, level='Year').sum(axis=1) / d0 * 100
    print(f'  behaviour term (pp of 2025 demand): {fmt(beh)}')
else:
    print(f'HEALTHCARE skipped: {config.PATH_HEALTHCARE_DEMAND} not found (see README)')

e = load(config.PATH_EDUCATION_DEMAND, 'Year', 'Demand_teachers', '--trend--BSL')
E = e.groupby(['base', 'Year', 'Nationality'], observed=True)['D'].sum().unstack('Nationality')
E1 = E.copy()
E1.loc[(slice(None), Y1), :] *= E1_F50
print('\nEDUCATION (teachers, trend)')
report('main projection', stats(E, P))
report('student-to-teacher ratio +10%', stats(E1, P))
lvl = (E1 - E).xs(Y1, level='Year').sum(axis=1) / E.xs(Y1, level='Year').sum(axis=1) * 100
print(f'  2050 level against the main projection (%): {fmt(lvl)}')

d = load(config.PATH_HOUSING_DEMAND, 'Year', 'Dwellings', '--trend', extra=('Age', 'Population', 'HHFR'))
print('\nHOUSING (dwellings, trend)')
young = d.Age.between(20, 34)
D = d.groupby(['base', 'Year', 'Nationality'])['D'].sum().unstack('Nationality')
Dy = d[young].groupby(['base', 'Year', 'Nationality'])['D'].sum().unstack('Nationality')
D1 = D.copy()
D1.loc[(slice(None), Y1), :] -= (1 - D1_F50) * Dy.xs(Y1, level='Year', drop_level=False)
report('main projection', stats(D, P))
report('household formation -10%', stats(D1, P))
yshare = (Dy / D).xs(Y1, level='Year') * 100
print(f'  share of 2050 dwellings headed at 20-34: AUT {fmt(yshare[AUT])}  FOR {fmt(yshare[FOR])}')
lvl = (D1 - D).xs(Y1, level='Year').sum(axis=1)
print(f'  2050 dwellings against the main projection: {fmt(lvl, 2)}  ({fmt(lvl / D.xs(Y1, level="Year").sum(axis=1) * 100, 4)}%)')
for nat in (AUT, FOR):
    g = (D1[nat].unstack('Year')[Y1] / D[nat].unstack('Year')[Y0] - 1) * 100
    g0 = (D[nat].unstack('Year')[Y1] / D[nat].unstack('Year')[Y0] - 1) * 100
    print(f'  {nat:<16} growth main projection {fmt(g0)}  scenario {fmt(g)}')
