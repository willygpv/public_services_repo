#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
from datetime import datetime
import os

# --- Configuration ---
INIT_YEAR = 2022
END_YEAR = 2026

# Input Paths
PATH_POP_TOTALS = r'XLSX/AT_population_totals_2022.csv.gz'
PATH_MORTALITY = r'Data/Eurostat/mortality-AT_nuts3_age_sex_2019-2100.csv'
PATH_FERTILITY = r'Data/Eurostat/fertility_adjusted-AT_nuts3_age_2019-2050.csv'
PATH_SEX_RATIO = r'Data/Eurostat/sexratio-AT_nuts3_sex_2019-2050.csv'
PATH_MIGRATION = r'Data/Eurostat/netmigration_EP2019-2023-AT_nuts3_age_sex_2022-2050_dynamicresidual.csv'

# Output Paths (Directories must exist or be handled by the user's environment)
PATH_OUT_XLSX = r'XLSX'
PATH_OUT_PQT = r'PQT'


def load_data():
    """Loads all required datasets."""
    print("Loading data...")
    
    # Initial Population
    df_init_pop = pd.read_csv(PATH_POP_TOTALS)
    
    # Mortality
    df_m = pd.read_csv(PATH_MORTALITY)
    
    # Fertility
    df_f = pd.read_csv(PATH_FERTILITY)
    
    # Sex Ratio
    df_sr = pd.read_csv(PATH_SEX_RATIO)
    
    # Net Migration
    df_mig = pd.read_csv(PATH_MIGRATION)
    
    return df_init_pop, df_m, df_f, df_sr, df_mig


def project_population(
    df_init_pop,
    df_mig,
    df_m,
    df_f,
    df_sr,
    start_year=2023,
    end_year=2035,
    regions=None
):
    """
    Projects population across all combinations of scenarios present in the input dataframes.
    Returns a dataframe with columns [year, geo, sex, age, Pop, scenario],
    and a process_log dict keyed by scenario combination label.
    """
    print(f"Running projection from {start_year} to {end_year}...")
    
    # Get unique scenario values
    pop_scenarios = df_init_pop['projection'].unique()
    mig_scenarios = df_mig['projection'].unique()
    mort_scenarios = df_m['projection'].unique()
    fert_scenarios = df_f['projection'].unique()

    final_proj_list = []
    process_log_all = {}

    # Iterate over all scenario combinations
    for pop_s, mig_s, mort_s, fert_s in itertools.product(
        pop_scenarios, mig_scenarios, mort_scenarios, fert_scenarios
    ):
        combo_label = f"{pop_s}-{mig_s}-{mort_s}-{fert_s}"

        # Filter dataframes for current scenario combination
        df_pop = df_init_pop[df_init_pop['projection'] == pop_s].copy()
        df_mig_s = df_mig[df_mig['projection'] == mig_s].copy()
        df_mort_s = df_m[df_m['projection'] == mort_s].copy()
        df_fert_s = df_f[df_f['projection'] == fert_s].copy()
        df_sr_s = df_sr.copy()  # assuming sex ratio not scenario-specific

        # Filter by regions if provided
        if regions is not None:
            df_pop = df_pop[df_pop['geo'].isin(regions)]
            df_mig_s = df_mig_s[df_mig_s['geo'].isin(regions)]
            df_mort_s = df_mort_s[df_mort_s['geo'].isin(regions)]
            df_fert_s = df_fert_s[df_fert_s['geo'].isin(regions)]
            df_sr_s = df_sr_s[df_sr_s['geo'].isin(regions)]

        # Build age integer mapping except for fertility
        ages = set(df_pop['age']).union(df_mort_s['age']).union(df_mig_s['age'])
        age_map = {
            a: (0 if a == 'Y_LT1' else 100 if a == 'Y_GE100' else int(a.replace('Y','')))
            for a in ages
        }
        for df_ in (df_pop, df_mig_s, df_mort_s):
            df_['age_int'] = df_['age'].map(age_map)

        # Pivot for fast lookup
        mort = df_mort_s.set_index(['geo','sex','age_int','TIME_PERIOD'])['OBS_VALUE']
        mig  = df_mig_s.set_index(['geo','sex','age_int','TIME_PERIOD'])['OBS_VALUE']
        fert = df_fert_s.set_index(['geo','age','TIME_PERIOD'])['OBS_VALUE']
        sr   = df_sr_s.set_index(['geo','TIME_PERIOD'])['Male to Female Ratio']

        # Initialize population at start_year
        pop0 = (
            df_pop[df_pop['TIME_PERIOD'] == start_year]
            .query("sex in ['M','F']")
            .groupby(['geo','sex','age','age_int'])['OBS_VALUE']
            .sum()
            .reset_index()
            .rename(columns={'OBS_VALUE':'Pop'})
        )
        pop0['TIME_PERIOD'] = start_year

        pop_dict = {start_year: pop0}
        process_log = {}

        # Simulation loop
        for year in range(start_year, end_year):
            pop = pop_dict[year].copy().sort_values(['geo','sex','age_int']).reset_index(drop=True)

            # Mortality
            idx = pd.MultiIndex.from_frame(pop[['geo','sex','age_int','TIME_PERIOD']])
            q = mort.reindex(idx).fillna(0).values
            pop['survivors'] = np.where(pop['age_int'] != 0, pop['Pop'] * (1 - q), pop['Pop'])
            deaths1 = (pop['Pop'] * q).sum()

            # Migration
            idx = pd.MultiIndex.from_frame(pop[['geo','sex','age_int','TIME_PERIOD']])
            net_mig = mig.reindex(idx).fillna(0).values
            pop['pop_after_mig'] = pop['survivors'] + net_mig
            immigrants = net_mig[net_mig > 0].sum()
            emmigrants = (-net_mig[net_mig < 0]).sum()

            # Fertility
            pop['fert_age'] = pop['age_int'].apply(lambda ai: 'Y_LT15' if ai < 15 else 'Y_GE50' if ai >= 50 else f'Y{ai}')
            females = pop[pop['sex'] == 'F']
            female_pop = females.groupby(['geo', 'fert_age'], as_index=False)['pop_after_mig'].sum().rename(
                columns={'fert_age': 'age', 'pop_after_mig': 'female_pop'})
            fert_rates = df_fert_s.loc[df_fert_s['TIME_PERIOD'] == year, ['geo', 'age', 'OBS_VALUE']].rename(
                columns={'OBS_VALUE': 'fert'})
            female_pop = female_pop.merge(fert_rates, on=['geo', 'age'], how='left')
            female_pop['fert'] = female_pop['fert'].fillna(0)
            female_pop['births'] = female_pop['female_pop'] * female_pop['fert']
            births_by_geo = female_pop.groupby('geo', as_index=False)['births'].sum().rename(
                columns={'births': 'total_births'})
            sr_year = df_sr_s.loc[df_sr_s['TIME_PERIOD'] == year, ['geo', 'Male to Female Ratio']].rename(
                columns={'Male to Female Ratio': 'sr'})
            births_by_geo = births_by_geo.merge(sr_year, on='geo', how='left')
            births_by_geo['sr'] = births_by_geo['sr'].fillna(1.05)
            births_by_geo['female_births'] = births_by_geo['total_births'] / (1 + births_by_geo['sr'])
            births_by_geo['male_births'] = births_by_geo['total_births'] - births_by_geo['female_births']

            # Births
            births_df = pd.concat([
                births_by_geo.assign(
                    sex='M', age='Y_LT1', age_int=age_map['Y_LT1'], TIME_PERIOD=year + 1,
                    Pop=lambda df: df['male_births']
                )[['geo', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']],
                births_by_geo.assign(
                    sex='F', age='Y_LT1', age_int=age_map['Y_LT1'], TIME_PERIOD=year + 1,
                    Pop=lambda df: df['female_births']
                )[['geo', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]
            ], ignore_index=True)

            # Age-0 mortality
            age0 = pop[pop['age_int'] == 0].copy()
            q0 = mort.reindex(pd.MultiIndex.from_frame(age0[['geo', 'sex', 'age_int', 'TIME_PERIOD']])).fillna(0).values
            age0['surv0'] = age0['pop_after_mig'] * (1 - q0)
            age1_df = age0.groupby(['geo', 'sex'])['surv0'].sum().reset_index().assign(
                age='Y1', age_int=1, TIME_PERIOD=year + 1, Pop=lambda d: d['surv0']
            )[['geo', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]

            # Aging
            others = pop[pop['age_int'] != 0].copy()
            others['age_int'] += 1
            others['age'] = np.where(
                others['age_int'] >= 100, 'Y_GE100', 'Y' + others['age_int'].astype(int).astype(str)
            )
            others['Pop'] = others['pop_after_mig']
            others['TIME_PERIOD'] = year + 1
            others = others[['geo', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]

            # Combine
            pop_next = pd.concat([others, age1_df, births_df], ignore_index=True)
            pop_next = (
                pop_next
                .groupby(['geo', 'sex', 'age', 'TIME_PERIOD'], as_index=False)['Pop']
                .sum()
            )
            pop_next['age_int'] = pop_next['age'].map(age_map)

            # Compute negatives before clipping
            is_neg = pop_next['Pop'] < 0
            n_neg = int(is_neg.sum())
            total_neg = float((-pop_next.loc[is_neg, 'Pop']).sum())

            # now clip the negatives away
            pop_next['Pop'] = pop_next['Pop'].clip(lower=0)

            # Log
            deaths0 = (age0['pop_after_mig'] * q0).sum()
            process_log[year] = {
                'total_births': float(female_pop['births'].sum()),
                'total_immigrants': float(immigrants),
                'total_emmigrants': float(emmigrants),
                'total_deaths': float(deaths1 + deaths0),
                'n_negative_entries_corrected': n_neg,
                'total_negative_pop_corrected': total_neg
            }

            pop_dict[year + 1] = pop_next

        # Compile final projection for this combination
        all_years = []
        for y in range(start_year, end_year + 1):
            dfy = pop_dict[y].copy()
            dfy['year'] = y
            all_years.append(dfy)
        final_proj_combo = pd.concat(all_years, ignore_index=True)
        final_proj_combo = final_proj_combo[['year', 'geo', 'sex', 'age', 'Pop']]
        final_proj_combo['Pop'] = final_proj_combo['Pop'].round().astype(int)
        final_proj_combo['scenario'] = combo_label

        final_proj_list.append(final_proj_combo)
        process_log_all[combo_label] = process_log

    final_proj_all = pd.concat(final_proj_list, ignore_index=True)
    return final_proj_all, process_log_all


def plot_cumulative_negative(df_log):
    """Generates the cumulative negative population correction plot."""
    # 1) Sort and compute cumulative sum
    df = df_log.sort_values(['scenario', 'year']).copy()
    df['cumulative_negative'] = df.groupby('scenario')['total_negative_pop_corrected'].cumsum()

    # 2) Plot preparation
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df['scenario'].unique())))

    # Create readable scenario labels
    df['scenario_label'] = df['scenario'].apply(lambda x: x.split('__')[0])

    # Plot lines
    for idx, (scen, grp) in enumerate(df.groupby('scenario')):
        label = grp['scenario_label'].iloc[0]
        ax.plot(grp['year'], grp['cumulative_negative'],
                label=label, linewidth=1.5, color=colors[idx])

    # 3) Axis formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative clipped-to-zero population (millions)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Negative Population Correction by Scenario', fontsize=14, fontweight='bold')

    # Format y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}' if x >= 1e6 else f'{x/1e3:.0f}k'))
    ax.grid(True, linestyle='--', alpha=0.7)

    # 4) Legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if len(df['scenario_label'].unique()) > 15:
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                           fontsize=7, frameon=True, ncol=max(1, len(df['scenario_label'].unique()) // 20),
                           title='Scenarios', title_fontsize=9)
    else:
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                           fontsize=9, frameon=True, ncol=1,
                           title='Scenarios', title_fontsize=10)

    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('lightgray')

    # 5) Styling
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 6) Annotations
    top_scenarios = df.groupby('scenario')['cumulative_negative'].last().nlargest(3)
    for scen in top_scenarios.index:
        last_point = df[df['scenario'] == scen].iloc[-1]
        ax.annotate(f"{scen.split('__')[0]}",
                    xy=(last_point['year'], last_point['cumulative_negative']),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=8, ha='left', va='center')

    return fig


def plot_total_population(projections):
    """Generates the total population over time plot."""
    eurostat_df = projections.copy()
    eurostat_df = eurostat_df.rename(columns={'year': 'TIME_PERIOD', 'Pop': 'OBS_VALUE'})
    eurostat_df = eurostat_df[(eurostat_df['TIME_PERIOD'] >= 2023) & (eurostat_df['TIME_PERIOD'] <= 2050)]

    # Aggregate total population by year and scenario
    total_pop = eurostat_df.groupby(['TIME_PERIOD', 'scenario'])['OBS_VALUE'].sum().unstack()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    total_pop.plot(ax=ax, colormap='tab20')

    ax.set_xlabel('Year')
    ax.set_ylabel('Total population')
    ax.set_title('Total population over time by scenario')
    ax.legend(title='Scenario')
    plt.tight_layout()
    
    return fig


def main():
    # 1. Load Data
    df_init_pop, df_m, df_f, df_sr, df_mig = load_data()

    # 2. Run Projections
    projections, process_log = project_population(
        df_init_pop,
        df_mig,
        df_m,
        df_f,
        df_sr,
        start_year=INIT_YEAR,
        end_year=END_YEAR,
        regions=None
    )

    # 3. Process Logs
    today = datetime.now().strftime("%Y%m%d")
    
    records = []
    for scenario, yearly_log in process_log.items():
        for year, metrics in yearly_log.items():
            row = {'scenario': scenario, 'year': year}
            row.update(metrics)
            records.append(row)

    df_log = pd.DataFrame(records)
    
    # Save Log CSV
    log_filename = os.path.join(PATH_OUT_XLSX, f"log_projected_AT_EU_{INIT_YEAR}_{END_YEAR}_{today}-fert.csv")
    print(f"Saving log to: {log_filename}")
    df_log.to_csv(log_filename, index=False)

    # 4. Post-process Projections (Scenario Adjustment)
    # Add naturalisations suffix
    projections['scenario'] = projections['scenario'] + '-BSL'
    
    # Optional: round float columns to 4 decimal places (if any remain)
    projections = projections.round(4)

    # Save Projections Parquet
    pqt_filename = os.path.join(PATH_OUT_PQT, f"projected_AT_EU_{INIT_YEAR}_{END_YEAR}_{today}-fert.parquet")
    print(f"Saving projections to: {pqt_filename}")
    projections.to_parquet(pqt_filename, index=False)
    
    print(f"Total rows in projection: {len(projections)}")

    # 5. Visualizations
    print("Generating plots...")
    
    # Plot 1: Cumulative Negative Population
    fig1 = plot_cumulative_negative(df_log)
    # plt.show() # Uncomment to display interactively
    
    # Plot 2: Total Population
    fig2 = plot_total_population(projections)
    # plt.show() # Uncomment to display interactively

    print("Done.")


if __name__ == "__main__":
    main()