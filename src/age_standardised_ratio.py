#!/usr/bin/env python
# coding: utf-8
"""
Age-Standardised Utilisation Ratio — foreign vs Austrian
=========================================================
Separates a difference in per-capita utilisation between citizenship groups
from a difference in their age composition.

The representation index (RI = demand share / population share) cannot make
that separation, because a deviation from parity mixes two things: a difference
in AGE COMPOSITION between the groups, and a difference in per-capita
utilisation between members of the two groups AT THE SAME AGE. Only the second
is behavioural. This script separates them by direct age standardisation:

    d_bar_n = sum_a d_{a,n} * P_std_a / sum_a P_std_a
    ASR     = d_bar_For / d_bar_Aut

Because both groups are weighted by the SAME age structure, the ASR is
invariant to each group's age composition and to its size, so the ageing and
volume effects of the decomposition drop out by construction. ASR = 1.0 means a
foreign and an Austrian national of the same age generate the same demand.

    crude ratio = age-composition factor x ASR      (multiplicative)

Per-capita rates come from each sector model's own base-year (--statusquo)
output, so this re-uses the published coefficients rather than re-estimating
anything. Healthcare is additionally refit from the patient microdata to obtain
an exact exp(X.beta) and a Monte Carlo CI on the ratio.

  Sector      Rate                         Base year   Source
  housing     dwellings per capita         2025        dwelling_demand parquet
  education   FTE teachers per capita      2025        teacher_projections parquet
  healthcare  bed-days per capita          2019        healthcare_demand parquet + GLM refit

SOURCE DATA IS NOT REDISTRIBUTED. The patient-level bed-day file behind the
healthcare arm is licensed hospital microdata and is not part of this repository,
so healthcare_refit() reports what is missing and stops rather than failing on a
missing path. The other sectors read the sector models' own outputs.

CAUTION: XLSX/patient_cohorts_2019.csv has Population = PATIENT count (1.31M),
NOT the resident population. `patient_no` is NaN for the ~7.55M non-hospitalised
rows, so a count of it silently drops them. Like the healthcare model, this
script counts `num_days` rows in patients_full_factorial_2019.csv.gz (8.86M =
full resident population, zeros included). Using the former as a denominator
yields bed-days per *patient* (~7/yr) instead of per capita (~1.2/yr).

Pipeline:
  1. Housing    -> crude + ASR from the dwelling_demand parquet (2025, statusquo)
  2. Education  -> crude + ASR from the teacher_projections parquet (2025, statusquo);
                   Population is duplicated across School_Type, so take it once per cell
  3. Healthcare -> crude + ASR from the healthcare_demand parquet, and again from an
                   exact GLM refit with a 1000-draw MC interval on the ratio
  4. Education mechanism: enrolment rate and teachers-per-enrolled-student by age,
                   to show the shortfall is participation, not tracking
  5. Export summary table (XLSX/) and figure (PNG/figures_methodology/)
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import pyarrow.dataset as ds
import statsmodels.api as sm
import statsmodels.formula.api as smf

import config

# ── Inputs ────────────────────────────────────────────────────────────

PATH_HOUSING = config.PATH_HOUSING_DEMAND
PATH_EDUCATION = config.PATH_EDUCATION_DEMAND
PATH_HEALTHCARE = config.PATH_HEALTHCARE_DEMAND
PATH_HEALTHCARE_RAW = r'XLSX/patients_full_factorial_2019.csv.gz'

# Baseline demographic scenario. The per-capita coefficients carry no year or
# scenario term, so the ASR is scenario-invariant; the scenario only fixes the
# standard population.
BASELINE_SCENARIO = 'BSL-BSL2023early__SensCrisisEra__anchorBase-BSL-BSL_ADJ2023-BSL'
BASE_YEAR = 2025
N_DRAWS = 1000
SEED = 42

# Outputs
PATH_SUMMARY_CSV = r'XLSX/age_standardised_ratio_summary.csv'
FIG_DIR = r'PNG/figures_methodology'
FIG_STEM = 'age_standardised_ratio'

AUT, FOR = 'Austria', 'Foreign country'

# Paper palette (Okabe-Ito). Nationality grammar: the ratio is
# foreign-relative-to-Austrian, so the marks are pink and the parity line --
# where a foreign national uses the service at the Austrian rate -- is green.
COLOR_AUSTRIAN = '#009E73'
COLOR_FOREIGN = '#CC79A7'

# Sector colours match the regional analysis, so each sector reads the same
# colour across the paper's figures. Markers double as secondary encoding so
# the panel never depends on colour alone.
SECTOR_STYLE = {
    'Housing': ('tab:orange', 'X'),
    'Education': ('tab:blue', 'o'),
    'Healthcare': ('tab:green', '^'),
}


# ── Helpers ───────────────────────────────────────────────────────────

def collapse_to_age(df, rate_col):
    """
    Collapse a finer grid (age x sex x nationality x region) to one row per
    (age, Nationality), taking the POPULATION-WEIGHTED mean rate.

    crude_and_asr() merges the standard population on `age` alone, so it must be
    handed exactly one row per (age, Nationality). Passing the uncollapsed grid
    silently gives every sub-cell within an age the whole age's weight, i.e. an
    unweighted mean over sex and region.
    """
    d = df.copy()
    d['_num'] = d[rate_col] * d['Population']
    grouped = (d.groupby(['age_group', 'Nationality'], observed=True, as_index=False)
                .agg(_num=('_num', 'sum'), w=('Population', 'sum')))
    grouped['rate'] = grouped['_num'] / grouped['w']
    return grouped.rename(columns={'age_group': 'age'})[['age', 'Nationality', 'rate', 'w']]


def age_profile(cell, min_demand_share=0.005):
    """
    Foreign/Austrian rate ratio at each age, for the mechanism panel.

    Ages where a service is barely used give an unstable ratio on a negligible
    base: no dwellings below ~15, no teachers above ~21. Left in, they dominate
    the axis while carrying almost no demand (housing at 15 reads 2.10 on 0.17%
    of dwellings; education's adult tail runs to age 90 on <0.5% of teachers).

    We therefore keep only ages holding at least `min_demand_share` of that
    sector's base-year demand. The rule is uniform across sectors and stated in
    the caption; it trims housing to 22+, education to 6-21, and leaves
    healthcare untouched.
    """
    d = cell.copy()
    d['demand'] = d['rate'] * d['w']
    share = d.groupby('age')['demand'].sum() / d['demand'].sum()

    rates = d.pivot(index='age', columns='Nationality', values='rate')
    weights = d.pivot(index='age', columns='Nationality', values='w')
    keep = (share >= min_demand_share).reindex(rates.index, fill_value=False)

    out = pd.DataFrame({'age': rates.index[keep],
                        'ratio': (rates[FOR] / rates[AUT])[keep]})
    out['w'] = (weights[AUT] + weights[FOR])[keep].values
    return out.dropna().reset_index(drop=True)


def crude_and_asr(cell):
    """
    cell: DataFrame with EXACTLY ONE ROW per (age, Nationality), columns
          [age, Nationality, rate, w]; rate = per-capita utilisation,
          w = that cell's own population. See collapse_to_age().

    Returns (crude, asr_total, asr_austrian) where
      crude        = foreign/Austrian rate, each on its OWN age structure
      asr_total    = standardised to the TOTAL resident age structure
      asr_austrian = standardised to the AUSTRIAN age structure (robustness)
    """
    if cell.groupby(['age', 'Nationality']).size().max() > 1:
        raise ValueError('crude_and_asr expects one row per (age, Nationality); '
                         'collapse the grid with collapse_to_age() first')

    std_total = cell.groupby('age', as_index=False)['w'].sum()
    std_aut = (cell[cell['Nationality'] == AUT]
               .groupby('age', as_index=False)['w'].sum())

    out = {}
    for label, std in [('total', std_total), ('austrian', std_aut)]:
        for nat in (AUT, FOR):
            merged = (cell[cell['Nationality'] == nat]
                      .merge(std, on='age', suffixes=('_own', '_std')))
            out[(label, nat, 'asr')] = ((merged['rate'] * merged['w_std']).sum()
                                        / merged['w_std'].sum())
            out[(label, nat, 'crude')] = ((merged['rate'] * merged['w_own']).sum()
                                          / merged['w_own'].sum())

    crude = out[('total', FOR, 'crude')] / out[('total', AUT, 'crude')]
    asr_total = out[('total', FOR, 'asr')] / out[('total', AUT, 'asr')]
    asr_austrian = out[('austrian', FOR, 'asr')] / out[('austrian', AUT, 'asr')]
    return crude, asr_total, asr_austrian


def report(sector, crude, asr_total, asr_austrian):
    """Print one sector's decomposition and return it as a summary row."""
    log_crude = np.log(crude)
    print(f'\n--- {sector} ---')
    print(f'  crude foreign/Austrian rate    : {crude:.4f}')
    print(f'  ASR (total std)   [BEHAVIOURAL]: {asr_total:.4f}')
    print(f'  ASR (Austrian std, robustness) : {asr_austrian:.4f}')
    print(f'  age-composition factor         : {crude / asr_total:.4f}')
    print(f'  log split: age {np.log(crude / asr_total) / log_crude:.1%} '
          f'| behaviour {np.log(asr_total) / log_crude:.1%}')
    return dict(sector=sector, crude=crude, asr_total_std=asr_total,
                asr_austrian_std=asr_austrian, age_factor=crude / asr_total)


# ── 1. Housing ────────────────────────────────────────────────────────

def housing():
    """Dwellings per capita, base year, status-quo scenario."""
    table = ds.dataset(PATH_HOUSING, format='parquet').to_table(
        columns=['Age', 'Nationality', 'Population', 'Dwellings', 'Year',
                 'scenario'],
        filter=((ds.field('Year') == BASE_YEAR)
                & (ds.field('scenario') == f'{BASELINE_SCENARIO}--statusquo')),
    ).to_pandas()
    cell = (table.groupby(['Age', 'Nationality'], as_index=False)
                 [['Population', 'Dwellings']].sum()
                 .rename(columns={'Age': 'age', 'Population': 'w'}))
    cell['rate'] = cell['Dwellings'] / cell['w']
    res = report('housing (dwellings per capita)', *crude_and_asr(cell))
    return res, age_profile(cell)


# ── 2. Education ──────────────────────────────────────────────────────

def education():
    """FTE teachers per capita, plus the participation-vs-tracking mechanism."""
    table = ds.dataset(PATH_EDUCATION, format='parquet').to_table(
        columns=['Age', 'Nationality', 'Population', 'Students',
                 'Demand_teachers', 'School_Type', 'Year', 'scenario'],
        filter=((ds.field('Year') == BASE_YEAR)
                & (ds.field('scenario') == f'{BASELINE_SCENARIO}--statusquo--BSL')),
    ).to_pandas()

    # Population repeats across School_Type -> take it once per (Age, Nationality)
    pop = (table.groupby(['Age', 'Nationality', 'School_Type'], as_index=False)
                ['Population'].sum()
                .groupby(['Age', 'Nationality'], as_index=False)['Population'].max())
    teachers = table.groupby(['Age', 'Nationality'], as_index=False)['Demand_teachers'].sum()
    cell = (pop.merge(teachers, on=['Age', 'Nationality'])
               .rename(columns={'Age': 'age', 'Population': 'w'}))
    cell['rate'] = cell['Demand_teachers'] / cell['w']
    res = report('education (FTE teachers per capita)', *crude_and_asr(cell))

    # Mechanism: is the shortfall participation, or tracking into cheaper types?
    students = table.groupby(['Age', 'Nationality'], as_index=False)['Students'].sum()
    enrolled = pop.merge(students, on=['Age', 'Nationality'])
    enrolled['enrol'] = enrolled['Students'] / enrolled['Population']
    enrol_ratio = enrolled.pivot(index='Age', columns='Nationality', values='enrol')
    enrol_ratio['ratio'] = enrol_ratio[FOR] / enrol_ratio[AUT]

    per_student = students.merge(teachers, on=['Age', 'Nationality'])
    per_student['tps'] = per_student['Demand_teachers'] / per_student['Students']
    tps_ratio = per_student.pivot(index='Age', columns='Nationality', values='tps')
    tps_ratio['ratio'] = tps_ratio[FOR] / tps_ratio[AUT]

    print('\n  education mechanism, foreign/Austrian ratio by age:')
    print('    age   enrolment   teachers-per-enrolled-student')
    for age in range(12, 19):
        print(f'    {age:3d}   {enrol_ratio.loc[age, "ratio"]:9.3f}   '
              f'{tps_ratio.loc[age, "ratio"]:9.3f}')
    print('  -> shortfall is post-compulsory PARTICIPATION, not tracking:')
    print('     enrolment at parity to 14, teachers-per-enrolled-student >= 1.')

    mix = (table[table['Age'].between(16, 18)]
           .groupby(['Nationality', 'School_Type'], observed=True,
                    as_index=False)['Students'].sum())
    mix['pct'] = mix.groupby('Nationality')['Students'].transform(
        lambda x: 100 * x / x.sum())
    print("\n  school-type mix, ages 16-18 (% of that group's students):")
    print(mix.pivot(index='School_Type', columns='Nationality',
                    values='pct').round(1).to_string())
    return res, age_profile(cell)


# ── 3. Healthcare ─────────────────────────────────────────────────────

def healthcare_from_parquet():
    """Recover lambda_hat = Bed_Days_Mean * 365 / Population off the projection."""
    table = ds.dataset(PATH_HEALTHCARE, format='parquet').to_table(
        columns=['age_group', 'Nationality', 'Population', 'Bed_Days_Mean',
                 'year', 'scenario'],
        filter=((ds.field('year') == BASE_YEAR)
                & (ds.field('scenario') == BASELINE_SCENARIO)),
    ).to_pandas()
    cell = (table.groupby(['age_group', 'Nationality'], as_index=False)
                 [['Population', 'Bed_Days_Mean']].sum()
                 .rename(columns={'age_group': 'age', 'Population': 'w'}))
    cell['rate'] = cell['Bed_Days_Mean'] * 365.0 / cell['w']
    res = report('healthcare (bed-days per capita, from parquet)',
                 *crude_and_asr(cell))
    # age_group g covers ages [(g-1)*5, g*5-1]; plot at the midpoint so the
    # 5-year healthcare grid shares an x-axis with the single-year sectors
    profile = age_profile(cell)
    profile['age'] = (profile['age'] - 1) * 5 + 2.5
    return res, profile


def healthcare_refit():
    """
    Exact refit of the published Poisson GLM, to get exp(X.beta) rather than the
    MC mean E[exp(X.beta_s)] (a small Jensen gap that mostly cancels in a ratio),
    plus a Monte Carlo interval on the ratio itself.

    Cohort construction mirrors model_healthcare_demand.py: Population = count of
    num_days rows = full resident population (zeros included).
    """
    if not os.path.exists(PATH_HEALTHCARE_RAW):
        sys.exit(f"Patient bed-day file not found: {PATH_HEALTHCARE_RAW}. "
                 "It is licensed hospital microdata and is not redistributed "
                 "with this repository; see the module docstring.")

    df = pd.read_csv(PATH_HEALTHCARE_RAW).rename(
        columns={'geo_2': 'NUTS2', 'ag_id': 'age_group'})
    group_cols = ['age_group', 'sex', 'Nationality', 'NUTS2']
    agg = (df.groupby(group_cols, observed=True)
             .agg(num_days=('num_days', 'sum'),
                  Population=('num_days', 'count'))
             .reset_index())
    assert agg['Population'].sum() == len(df), 'denominator is not the full population'

    formula = ('num_days ~ C(age_group) + C(sex) + C(Nationality) + C(NUTS2) '
               '+ C(age_group):C(Nationality) + C(age_group):C(sex)')
    model = smf.glm(formula=formula, data=agg,
                    offset=np.log(agg['Population']),
                    family=sm.families.Poisson()).fit(cov_type='HC3')
    agg['rate'] = model.predict(agg, offset=np.zeros(len(agg)))
    cell = collapse_to_age(agg, 'rate')
    res = report('healthcare (bed-days per capita, exact GLM refit)',
                 *crude_and_asr(cell))

    # observed, model-free -- a calibration check on the GLM
    agg['obs'] = agg['num_days'] / agg['Population']
    _, obs_asr, _ = crude_and_asr(collapse_to_age(agg, 'obs'))
    print(f'  observed (model-free) ASR      : {obs_asr:.4f}  (calibration check)')

    # MC interval on the ratio, same draws and seed as the published model
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(model.params.values,
                                    model.cov_params().values, N_DRAWS)
    design = patsy.dmatrix(formula.split('~')[1], agg,
                           return_type='dataframe').values
    ratios = []
    for beta in draws:
        drawn = collapse_to_age(agg.assign(rate=np.exp(design @ beta)), 'rate')
        std = drawn.groupby('age', as_index=False)['w'].sum()
        vals = {}
        for nat in (AUT, FOR):
            merged = (drawn[drawn['Nationality'] == nat]
                      .merge(std, on='age', suffixes=('_own', '_std')))
            vals[nat] = (merged['rate'] * merged['w_std']).sum() / merged['w_std'].sum()
        ratios.append(vals[FOR] / vals[AUT])
    low, high = np.percentile(ratios, [2.5, 97.5])
    print(f'  MC 95% CI on ASR               : [{low:.4f}, {high:.4f}]')
    res['asr_ci_low'], res['asr_ci_high'] = low, high
    res['asr_observed'] = obs_asr
    return res


# ── Figure ────────────────────────────────────────────────────────────

def plot_asr(summary, profiles, ci):
    """
    Two panels:
      (a) crude -> age-standardised ratio per sector (a dumbbell, so the
          segment length IS the age-composition effect)
      (b) the underlying foreign/Austrian rate ratio at each age

    Both panels share one meaning for the y=1 / x=1 line: the Austrian rate.
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Panel a: crude -> age-standardised ----
    order = ['Housing', 'Education', 'Healthcare']
    ypos = {s: i for i, s in enumerate(order[::-1])}

    ax1.axvline(1.0, color=COLOR_AUSTRIAN, linestyle='--', linewidth=1.5,
                alpha=0.8, zorder=1)
    ax1.text(1.0, 2.95, ' Austrian rate', color=COLOR_AUSTRIAN,
             fontsize=9, va='top', ha='left')

    for sector in order:
        y = ypos[sector]
        crude = summary.loc[summary['sector_short'] == sector, 'crude'].iloc[0]
        asr = summary.loc[summary['sector_short'] == sector, 'asr_total_std'].iloc[0]

        ax1.plot([crude, asr], [y, y], color=COLOR_FOREIGN, linewidth=2.5,
                 alpha=0.45, zorder=2, solid_capstyle='round')
        ax1.scatter([crude], [y], s=95, facecolor='white',
                    edgecolor=COLOR_FOREIGN, linewidth=2.0, zorder=4)
        ax1.scatter([asr], [y], s=95, color=COLOR_FOREIGN,
                    edgecolor='white', linewidth=1.0, zorder=5)

        # direct labels, pushed to the outside of each end of the dumbbell
        left = min(crude, asr)
        ax1.annotate(f'{crude:.2f}', (crude, y),
                     xytext=(-9 if crude == left else 9, 0),
                     textcoords='offset points', fontsize=9, color='#555555',
                     va='center', ha='right' if crude == left else 'left')
        ax1.annotate(f'{asr:.2f}', (asr, y),
                     xytext=(-9 if asr == left else 9, 0),
                     textcoords='offset points', fontsize=9,
                     fontweight='bold', color='#333333',
                     va='center', ha='right' if asr == left else 'left')

    # Healthcare CI on the standardised estimate. It is narrower than the marker,
    # so caps are drawn on top -- otherwise the interval is invisible and the
    # reader cannot tell it was estimated at all.
    y_health = ypos['Healthcare']
    ax1.errorbar([np.mean(ci)], [y_health],
                 xerr=[[np.mean(ci) - ci[0]], [ci[1] - np.mean(ci)]],
                 fmt='none', ecolor='#8C4A6E', elinewidth=1.4, capsize=4,
                 capthick=1.4, zorder=6)

    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order[::-1])
    ax1.set_xlim(0.25, 1.30)
    ax1.set_ylim(-0.55, 3.05)
    ax1.set_xlabel('Foreign / Austrian per-capita utilisation rate')
    ax1.set_title('a  Removing age composition', loc='left', fontweight='bold')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker='o', linestyle='none', markersize=9,
                   markerfacecolor='white', markeredgecolor=COLOR_FOREIGN,
                   markeredgewidth=2.0, label='Crude'),
        plt.Line2D([], [], marker='o', linestyle='none', markersize=9,
                   color=COLOR_FOREIGN, markeredgecolor='white',
                   label='Age-standardised'),
    ]
    ax1.legend(handles=handles, frameon=False, fontsize=9, loc='upper left',
               ncol=2, columnspacing=1.2, handletextpad=0.4)

    # ---- Panel b: rate ratio by age ----
    ax2.axhline(1.0, color=COLOR_AUSTRIAN, linestyle='--', linewidth=1.5,
                alpha=0.8, zorder=1)

    for sector, profile in profiles.items():
        color, marker = SECTOR_STYLE[sector]
        step = max(1, len(profile) // 12)  # thin the markers, keep the line dense
        ax2.plot(profile['age'], profile['ratio'], color=color, linewidth=2.0,
                 zorder=3, label=sector)
        ax2.plot(profile['age'][::step], profile['ratio'][::step], linestyle='none',
                 marker=marker, markersize=6, color=color,
                 markeredgecolor='white', markeredgewidth=0.8, zorder=4)

    # Austria's compulsory schooling ends at 15; this is where the education
    # ratio breaks from parity, so mark it rather than leave the drop unexplained
    ax2.axvline(15, color='#999999', linestyle=':', linewidth=1.2, zorder=1)
    ax2.annotate('compulsory\nschooling ends', xy=(15, 0.47),
                 xytext=(18, 0.47), fontsize=8, color='#666666',
                 va='bottom', ha='left')

    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Foreign / Austrian rate')
    ax2.set_title('b  Rate ratio at each age', loc='left', fontweight='bold')
    ax2.set_xlim(0, 95)
    ax2.set_ylim(0.42, 1.28)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    # Legend keyed on the sector MARKER (matching the series and the regional
    # figures), not a bare colour line.
    sector_handles = [
        plt.Line2D([], [], color=SECTOR_STYLE[s][0], marker=SECTOR_STYLE[s][1],
                   markersize=8, markeredgecolor='white', markeredgewidth=0.8,
                   linewidth=2.0, label=s)
        for s in profiles
    ]
    ax2.legend(handles=sector_handles, frameon=False, fontsize=9,
               loc='upper right', ncol=3, columnspacing=1.2, handlelength=1.6)

    fig.tight_layout()
    return fig


def save_figure(fig, stem: str, directory: str = FIG_DIR) -> None:
    """Write the figure as PNG and PDF."""
    os.makedirs(directory, exist_ok=True)
    for path, kwargs in ((os.path.join(directory, f'{stem}.png'), {'dpi': 300}),
                         (os.path.join(directory, f'{stem}.pdf'), {})):
        fig.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved figure to: {path}')


# ── Main ──────────────────────────────────────────────────────────────

def check_inputs() -> None:
    """Report every missing input at once, rather than failing on the first.

    The sector parquets are rebuilt by running the sector models; the patient
    bed-day file is licensed microdata that is not redistributed, so on a clean
    checkout this reports what is needed instead of raising FileNotFoundError
    from inside a reader.
    """
    required = [
        (PATH_HOUSING, 'housing demand',
         'run src/model_housing_demand.py to produce it'),
        (PATH_EDUCATION, 'education demand',
         'run src/model_education_demand.py to produce it'),
        (PATH_HEALTHCARE, 'healthcare demand',
         'run src/model_healthcare_demand.py to produce it'),
        (PATH_HEALTHCARE_RAW, 'patient bed-days',
         'licensed hospital microdata, not redistributed with this repository'),
    ]
    missing = [r for r in required if not os.path.exists(r[0])]
    if missing:
        print('Cannot run: the following inputs are not present.')
        for path, what, why in missing:
            print()
            print(f'  {what}')
            print(f'    {path}')
            print(f'    {why}')
        sys.exit(1)


def main():
    check_inputs()
    housing_res, housing_prof = housing()
    education_res, education_prof = education()
    healthcare_res, healthcare_prof = healthcare_from_parquet()
    refit_res = healthcare_refit()

    summary = pd.DataFrame([housing_res, education_res, healthcare_res, refit_res])
    summary['sector_short'] = ['Housing', 'Education', 'Healthcare',
                               'Healthcare (refit)']

    print('\n' + '=' * 84)
    print('AGE-STANDARDISED FOREIGN/AUSTRIAN UTILISATION RATIO')
    print('=' * 84)
    print(summary.round(4).to_string(index=False))
    print("""
Values reported in the paper (rounded):  housing 1.01 | education 0.95
                                         healthcare 0.62 (CI 0.60-0.64)
Crude ratios:                            0.89 | 1.08 | 0.36

The split between age composition and behaviour is MULTIPLICATIVE:
crude = age factor x ASR = 0.58 x 0.62 = 0.36 for healthcare (on the log scale,
age 53% / behaviour 47%). Restating it as a share of the additive shortfall from
parity would give a different, and misleading, ~60/40.

Cross-check against the published representation index (foreign population share
20.2% in 2025): RI = r / (0.202*r + 0.798) for a crude ratio r. That gives
healthcare 0.417 (published 0.412), education 1.06, housing 0.91.""")

    summary.to_csv(PATH_SUMMARY_CSV, index=False)
    print(f'\nSaved summary to: {PATH_SUMMARY_CSV}')

    # Figure. Panel (a) uses the parquet-derived healthcare row so all three
    # sectors come from the published projection outputs on the same footing;
    # the CI comes from the refit, which is where the MC draws live.
    fig = plot_asr(summary,
                   {'Housing': housing_prof, 'Education': education_prof,
                    'Healthcare': healthcare_prof},
                   ci=(refit_res['asr_ci_low'], refit_res['asr_ci_high']))
    save_figure(fig, FIG_STEM)


if __name__ == '__main__':
    main()
