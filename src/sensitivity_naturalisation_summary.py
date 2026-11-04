"""Summary of the doubled naturalisation scenario of Supplementary Table 9.

Same per-scenario statistics as sensitivity_scenarios.py. run_sensitivity_naturalisation.sh runs
this once in the repository root and once in its copy. Writes nothing. Without the hospital
file the healthcare lines are skipped.
"""

import os

import pandas as pd
import pyarrow.dataset as ds

import config

Y0, Y1 = 2025, 2050
FOR, AUT = 'Foreign country', 'Austria'
print('vintage', config.PROJECTION_VINTAGE, '| total', config.TOTAL_POPULATION_VINTAGE, '| NAT_RATE_MULT', config.NAT_RATE_MULT)


def fmt(s, d=3):
    return f'{s.median():,.{d}f} [{s.min():,.{d}f}, {s.max():,.{d}f}]'


def load(path, ycol, vcol, sfx):
    t = ds.dataset(path).to_table(columns=['scenario', ycol, 'Nationality', vcol],
                                  filter=ds.field(ycol).isin([Y0, Y1])).to_pandas()
    t = t.rename(columns={ycol: 'Year', vcol: 'D'})
    for c in ['scenario', 'Nationality']:
        t[c] = t[c].astype(str).str.strip()
    if sfx:
        t = t[t.scenario.str.endswith(sfx)]
    t['base'] = t.scenario.str.split('--').str[0]
    t = t[t.base.str.startswith('BSL-')]
    return t.groupby(['base', 'Year', 'Nationality'], observed=True)['D'].sum().unstack('Nationality')


p = ds.dataset(config.PATH_PROJECTION).to_table(
    columns=['TIME_PERIOD', 'Nationality', 'scenario', 'OBS_VALUE'],
    filter=ds.field('TIME_PERIOD').isin([Y0, Y1])).to_pandas()
p['Nationality'] = p.Nationality.astype(str).str.strip()
P = p.groupby(['scenario', 'TIME_PERIOD', 'Nationality'])['OBS_VALUE'].sum().unstack('Nationality')
P.index = P.index.set_names(['base', 'Year'])
for y in (Y0, Y1):
    q = P.xs(y, level='Year')
    print(f'population {y}: AUT {fmt(q[AUT] / 1e6, 5)} M  FOR {fmt(q[FOR] / 1e6, 5)} M  '
          f'foreign share {fmt(q[FOR] / q.sum(axis=1) * 100)}%  total {fmt(q.sum(axis=1) / 1e6, 5)} M')

log = pd.read_csv(config.PATH_PROJECTION_LOG, usecols=range(10))
y = log.groupby(['year', 'scenario'])['total_naturalizations'].sum().unstack('scenario')
cum = y.loc[Y0:Y1].sum()
print(f"naturalisations {Y0} {y.loc[Y0].median():,.0f}  {Y1} {y.loc[Y1].median():,.0f}  "
      f"cumulative {cum.median():,.0f} [{cum.min():,.0f}, {cum.max():,.0f}]")

has_hospital = os.path.exists(config.PATH_HEALTHCARE_DEMAND)
ps = (P[FOR] / P.sum(axis=1)).unstack('Year')
for name, path, ycol, vcol, sfx in [
        ('Housing', config.PATH_HOUSING_DEMAND, 'Year', 'Dwellings', '--trend'),
        ('Education', config.PATH_EDUCATION_DEMAND, 'Year', 'Demand_teachers', '--trend--BSL'),
        ('Healthcare', config.PATH_HEALTHCARE_DEMAND, 'year', 'Bed_Days_Mean', None)]:
    if name == 'Healthcare' and not has_hospital:
        print(f'Healthcare skipped: {path} not found (see README)')
        continue
    D = load(path, ycol, vcol, sfx)
    tot = D.sum(axis=1).unstack('Year')
    sh = (D[FOR] / D.sum(axis=1)).unstack('Year')
    ri = sh[Y1] / ps.loc[sh.index, Y1]
    print(f'{name:<10} n={len(tot)} growth {fmt((tot[Y1] / tot[Y0] - 1) * 100)}  '
          f'for.share 2050 {fmt(sh[Y1] * 100)}  RI 2050 {fmt(ri, 5)}')

if has_hospital:
    r = pd.read_csv(config.PATH_REGIONAL_DECOMPOSITION)
    n = r[(r.Region_Code == 'NAT') & (r.Year == Y1)].set_index(['Service', 'Nationality'])
    d0 = n.loc['Healthcare', 'D0'].sum()
    print(f"healthcare: AUT ageing {n.loc[('Healthcare', AUT), 'Struct_Abs'] / d0 * 100:.1f} pp, "
          f"FOR volume {n.loc[('Healthcare', FOR), 'Vol_Abs'] / d0 * 100:.1f} pp, "
          f"ratio {n.loc[('Healthcare', AUT), 'Struct_Abs'] / n.loc[('Healthcare', FOR), 'Vol_Abs']:.2f} "
          f"(highest migration {n.loc[('Healthcare', AUT), 'Struct_Abs'] / n.loc[('Healthcare', FOR), 'Vol_Abs_max']:.2f})")
