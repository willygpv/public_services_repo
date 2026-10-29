"""Crude and age-standardised utilisation ratios of foreign to Austrian nationals in the three sectors."""

import config
import glm_cov

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.api as sm
import statsmodels.formula.api as smf

HOUS_PQ = config.PATH_HOUSING_DEMAND
EDU_PQ = config.PATH_EDUCATION_DEMAND
HEALTH_PQ = config.PATH_HEALTHCARE_DEMAND
HEALTH_RAW = config.PATH_PATIENTS

BSL = 'BSL-BSL2023early__SensCrisisEra__anchorBase-BSL-BSL_ADJ2023-BSL'
BASE_YEAR = 2025
N_DRAWS = 1000
SEED = 42

OUT_CSV = r'XLSX/age_standardised_ratio_summary.csv'

AUT, FOR = 'Austria', 'Foreign country'


def collapse_to_age(df, rate_col):
    d = df.copy()
    d['_num'] = d[rate_col] * d['Population']
    g = (d.groupby(['age_group', 'Nationality'], observed=True, as_index=False)
          .agg(_num=('_num', 'sum'), w=('Population', 'sum')))
    g['rate'] = g['_num'] / g['w']
    return g.rename(columns={'age_group': 'age'})[['age', 'Nationality', 'rate', 'w']]


def age_profile(cell, min_demand_share=0.005):
    d = cell.copy()
    d['demand'] = d['rate'] * d['w']
    share = d.groupby('age')['demand'].sum() / d['demand'].sum()

    p = d.pivot(index='age', columns='Nationality', values='rate')
    w = d.pivot(index='age', columns='Nationality', values='w')
    keep = (share >= min_demand_share).reindex(p.index, fill_value=False)

    out = pd.DataFrame({'age': p.index[keep], 'ratio': (p[FOR] / p[AUT])[keep]})
    out['w'] = (w[AUT] + w[FOR])[keep].values
    return out.dropna().reset_index(drop=True)


def crude_and_asr(cell):
    if cell.groupby(['age', 'Nationality']).size().max() > 1:
        raise ValueError('crude_and_asr expects one row per (age, Nationality); '
                         'collapse the grid with collapse_to_age() first')

    std_total = cell.groupby('age', as_index=False)['w'].sum()
    std_aut = (cell[cell['Nationality'] == AUT]
               .groupby('age', as_index=False)['w'].sum())

    out = {}
    for label, std in [('total', std_total), ('austrian', std_aut)]:
        for nat in (AUT, FOR):
            s = cell[cell['Nationality'] == nat]
            m = s.merge(std, on='age', suffixes=('_own', '_std'))
            out[(label, nat, 'asr')] = ((m['rate'] * m['w_std']).sum()
                                        / m['w_std'].sum())
            out[(label, nat, 'crude')] = ((m['rate'] * m['w_own']).sum()
                                          / m['w_own'].sum())

    crude = out[('total', FOR, 'crude')] / out[('total', AUT, 'crude')]
    asr_t = out[('total', FOR, 'asr')] / out[('total', AUT, 'asr')]
    asr_a = out[('austrian', FOR, 'asr')] / out[('austrian', AUT, 'asr')]
    return crude, asr_t, asr_a


def report(sector, crude, asr_t, asr_a):
    lg = np.log(crude)
    print(f'\n--- {sector} ---')
    print(f'  crude foreign/Austrian rate    : {crude:.4f}')
    print(f'  ASR (total std)   [BEHAVIOURAL]: {asr_t:.4f}')
    print(f'  ASR (Austrian std, robustness) : {asr_a:.4f}')
    print(f'  age-composition factor         : {crude / asr_t:.4f}')
    print(f'  log split: age {np.log(crude / asr_t) / lg:.1%} '
          f'| behaviour {np.log(asr_t) / lg:.1%}')
    return dict(sector=sector, crude=crude, asr_total_std=asr_t,
                asr_austrian_std=asr_a, age_factor=crude / asr_t)


def housing():
    t = ds.dataset(HOUS_PQ, format='parquet').to_table(
        columns=['Age', 'Nationality', 'Population', 'Dwellings', 'Year',
                 'scenario'],
        filter=((ds.field('Year') == BASE_YEAR)
                & (ds.field('scenario') == f'{BSL}--statusquo')),
    ).to_pandas()
    cell = (t.groupby(['Age', 'Nationality'], as_index=False)
             [['Population', 'Dwellings']].sum()
             .rename(columns={'Age': 'age', 'Population': 'w'}))
    cell['rate'] = cell['Dwellings'] / cell['w']
    res = report('housing (dwellings per capita)', *crude_and_asr(cell))
    return res, age_profile(cell)


def education():
    t = ds.dataset(EDU_PQ, format='parquet').to_table(
        columns=['Age', 'Nationality', 'Population', 'Students',
                 'Demand_teachers', 'School_Type', 'Year', 'scenario'],
        filter=((ds.field('Year') == BASE_YEAR)
                & (ds.field('scenario') == f'{BSL}--statusquo--BSL')),
    ).to_pandas()

    pop = (t.groupby(['Age', 'Nationality', 'School_Type'], as_index=False)
            ['Population'].sum()
            .groupby(['Age', 'Nationality'], as_index=False)['Population'].max())
    tch = t.groupby(['Age', 'Nationality'], as_index=False)['Demand_teachers'].sum()
    cell = (pop.merge(tch, on=['Age', 'Nationality'])
               .rename(columns={'Age': 'age', 'Population': 'w'}))
    cell['rate'] = cell['Demand_teachers'] / cell['w']
    res = report('education (FTE teachers per capita)', *crude_and_asr(cell))

    stu = t.groupby(['Age', 'Nationality'], as_index=False)['Students'].sum()
    enr = pop.merge(stu, on=['Age', 'Nationality'])
    enr['enrol'] = enr['Students'] / enr['Population']
    e = enr.pivot(index='Age', columns='Nationality', values='enrol')
    e['ratio'] = e[FOR] / e[AUT]

    tps = stu.merge(tch, on=['Age', 'Nationality'])
    tps['tps'] = tps['Demand_teachers'] / tps['Students']
    p = tps.pivot(index='Age', columns='Nationality', values='tps')
    p['ratio'] = p[FOR] / p[AUT]

    print('\n  education mechanism, foreign/Austrian ratio by age:')
    print('    age   enrolment   teachers-per-enrolled-student')
    for a in range(12, 19):
        print(f'    {a:3d}   {e.loc[a, "ratio"]:9.3f}   {p.loc[a, "ratio"]:9.3f}')
    print('  -> shortfall is post-compulsory PARTICIPATION, not tracking:')
    print('     enrolment at parity to 14, teachers-per-enrolled-student >= 1.')

    mix = (t[t['Age'].between(16, 18)]
           .groupby(['Nationality', 'School_Type'], observed=True,
                    as_index=False)['Students'].sum())
    mix['pct'] = mix.groupby('Nationality')['Students'].transform(
        lambda x: 100 * x / x.sum())
    print('\n  school-type mix, ages 16-18 (% of that group\'s students):')
    print(mix.pivot(index='School_Type', columns='Nationality',
                    values='pct').round(1).to_string())
    return res, age_profile(cell)


def healthcare_from_parquet():
    t = ds.dataset(HEALTH_PQ, format='parquet').to_table(
        columns=['age_group', 'Nationality', 'Population', 'Bed_Days_Mean',
                 'year', 'scenario'],
        filter=((ds.field('year') == BASE_YEAR) & (ds.field('scenario') == BSL)),
    ).to_pandas()
    cell = (t.groupby(['age_group', 'Nationality'], as_index=False)
             [['Population', 'Bed_Days_Mean']].sum()
             .rename(columns={'age_group': 'age', 'Population': 'w'}))
    cell['rate'] = cell['Bed_Days_Mean'] * 365.0 / cell['w']
    res = report('healthcare (bed-days per capita, from parquet)',
                 *crude_and_asr(cell))
    prof = age_profile(cell)
    prof['age'] = (prof['age'] - 1) * 5 + 2.5
    return res, prof


def healthcare_refit():
    df = pd.read_csv(HEALTH_RAW).rename(
        columns={'geo_2': 'NUTS2', 'ag_id': 'age_group'})
    group_cols = ['age_group', 'sex', 'Nationality', 'NUTS2']
    agg = (df.groupby(group_cols, observed=True)
             .agg(num_days=('num_days', 'sum'),
                  Population=('num_days', 'count'))
             .reset_index())
    assert agg['Population'].sum() == len(df), 'denominator is not the full population'

    formula = f'num_days ~ {config.HEALTHCARE_RHS}'
    model = smf.glm(formula=formula, data=agg,
                    offset=np.log(agg['Population']),
                    family=sm.families.Poisson()).fit()
    agg['rate'] = model.predict(agg, offset=np.zeros(len(agg)))
    cell = collapse_to_age(agg, 'rate')
    res = report('healthcare (bed-days per capita, exact GLM refit)',
                 *crude_and_asr(cell))

    agg['obs'] = agg['num_days'] / agg['Population']
    _, obs_asr, _ = crude_and_asr(collapse_to_age(agg, 'obs'))
    print(f'  observed (model-free) ASR      : {obs_asr:.4f}  '
          f'(calibration check)')

    import patsy
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(model.params.values,
                                    glm_cov.cov_hc3(model).values, N_DRAWS)
    X = patsy.dmatrix(formula.split('~')[1], agg, return_type='dataframe').values
    ratios = []
    for b in draws:
        c = collapse_to_age(agg.assign(rate=np.exp(X @ b)), 'rate')
        std = c.groupby('age', as_index=False)['w'].sum()
        vals = {}
        for nat in (AUT, FOR):
            m = c[c['Nationality'] == nat].merge(std, on='age',
                                                 suffixes=('_own', '_std'))
            vals[nat] = (m['rate'] * m['w_std']).sum() / m['w_std'].sum()
        ratios.append(vals[FOR] / vals[AUT])
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    print(f'  MC 95% CI on ASR               : [{lo:.4f}, {hi:.4f}]')
    res['asr_ci_low'], res['asr_ci_high'] = lo, hi
    res['asr_observed'] = obs_asr
    return res


def main():
    (h_res, h_prof) = housing()
    (e_res, e_prof) = education()
    (c_res, c_prof) = healthcare_from_parquet()
    r_res = healthcare_refit()

    rows = [h_res, e_res, c_res, r_res]
    summary = pd.DataFrame(rows)
    summary['sector_short'] = ['Housing', 'Education', 'Healthcare',
                               'Healthcare (refit)']

    print('\n' + '=' * 84)
    print('AGE-STANDARDISED FOREIGN/AUSTRIAN UTILISATION RATIO')
    print('=' * 84)
    print(summary.round(4).to_string(index=False))

    summary.to_csv(OUT_CSV, index=False)
    print(f'\nSaved summary to: {OUT_CSV}')


if __name__ == '__main__':
    main()
