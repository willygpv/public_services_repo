"""Cohort-component projection of the total population by NUTS 3 region, sex and single age, 2025-2051."""

import pandas as pd
import numpy as np
import itertools
import os

import config

PATH_MORTALITY = r'Data/Eurostat/mortality-AT_nuts3_age_sex_2019-2100.csv'
PATH_FERTILITY = r'Data/Eurostat/fertility_adjusted-AT_nuts3_age_2019-2050.csv'
PATH_SEX_RATIO = r'Data/Eurostat/sexratio-AT_nuts3_sex_2019-2050.csv'

PATH_OUT_XLSX = r'XLSX'

INIT_YEAR = 2025
END_YEAR = 2051
PATH_POP_TOTALS = config.PATH_POPULATION_2025
PATH_MIGRATION = r'XLSX/netmigration_EP2019-2023-AT_nuts3_age_sex_2022-2050_dynamicresidual.csv'


def load_data():
    print("Loading data...")

    df_init_pop = pd.read_csv(PATH_POP_TOTALS)

    df_m = pd.read_csv(PATH_MORTALITY)

    df_f = pd.read_csv(PATH_FERTILITY)

    df_sr = pd.read_csv(PATH_SEX_RATIO)

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
    print(f"Running projection from {start_year} to {end_year}...")

    pop_scenarios = df_init_pop['projection'].unique()
    mig_scenarios = df_mig['projection'].unique()
    mort_scenarios = df_m['projection'].unique()
    fert_scenarios = df_f['projection'].unique()

    final_proj_list = []
    process_log_all = {}

    for pop_s, mig_s, mort_s, fert_s in itertools.product(
        pop_scenarios, mig_scenarios, mort_scenarios, fert_scenarios
    ):
        combo_label = f"{pop_s}-{mig_s}-{mort_s}-{fert_s}"

        df_pop = df_init_pop[df_init_pop['projection'] == pop_s].copy()
        df_mig_s = df_mig[df_mig['projection'] == mig_s].copy()
        df_mort_s = df_m[df_m['projection'] == mort_s].copy()
        df_fert_s = df_f[df_f['projection'] == fert_s].copy()
        df_sr_s = df_sr.copy()

        if regions is not None:
            df_pop = df_pop[df_pop['geo'].isin(regions)]
            df_mig_s = df_mig_s[df_mig_s['geo'].isin(regions)]
            df_mort_s = df_mort_s[df_mort_s['geo'].isin(regions)]
            df_fert_s = df_fert_s[df_fert_s['geo'].isin(regions)]
            df_sr_s = df_sr_s[df_sr_s['geo'].isin(regions)]

        ages = set(df_pop['age']).union(df_mort_s['age']).union(df_mig_s['age'])
        age_map = {
            a: (0 if a == 'Y_LT1' else 100 if a == 'Y_GE100' else int(a.replace('Y','')))
            for a in ages
        }
        for df_ in (df_pop, df_mig_s, df_mort_s):
            df_['age_int'] = df_['age'].map(age_map)

        mort = df_mort_s.set_index(['geo','sex','age_int','TIME_PERIOD'])['OBS_VALUE']
        mig  = df_mig_s.set_index(['geo','sex','age_int','TIME_PERIOD'])['OBS_VALUE']
        fert = df_fert_s.set_index(['geo','age','TIME_PERIOD'])['OBS_VALUE']
        sr   = df_sr_s.set_index(['geo','TIME_PERIOD'])['Male to Female Ratio']

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

        for year in range(start_year, end_year):
            pop = pop_dict[year].copy().sort_values(['geo','sex','age_int']).reset_index(drop=True)

            idx_names = ['geo', 'sex', 'age_int', 'TIME_PERIOD']
            idx = pd.MultiIndex.from_arrays(
                [pop['geo'], pop['sex'], (pop['age_int'] + 1).clip(upper=100), pop['TIME_PERIOD']],
                names=idx_names)
            q = mort.reindex(idx).fillna(0).values
            pop['survivors'] = pop['Pop'] * (1 - q)
            deaths1 = (pop['Pop'] * q).sum()

            idx = pd.MultiIndex.from_arrays(
                [pop['geo'], pop['sex'], np.where(pop['age_int'] < 100, pop['age_int'] + 1, -1),
                 pop['TIME_PERIOD']],
                names=idx_names)
            net_mig = mig.reindex(idx).fillna(0).values
            pop['pop_after_mig'] = pop['survivors'] + net_mig
            immigrants = net_mig[net_mig > 0].sum()
            emmigrants = (-net_mig[net_mig < 0]).sum()

            pop['fert_age'] = pop['age_int'].apply(lambda ai: 'Y_LT15' if ai == 14 else 'Y_GE50' if ai == 50 else f'Y{ai}')
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

            idx0 = pd.MultiIndex.from_arrays(
                [births_df['geo'], births_df['sex'], np.zeros(len(births_df), dtype=int),
                 np.full(len(births_df), year)],
                names=idx_names)
            q0 = mort.reindex(idx0).fillna(0).values
            deaths0 = (births_df['Pop'] * q0).sum()
            mig0 = mig.reindex(idx0).fillna(0).values
            births_df['Pop'] = births_df['Pop'] * (1 - q0) + mig0
            immigrants += mig0[mig0 > 0].sum()
            emmigrants += (-mig0[mig0 < 0]).sum()

            others = pop.copy()
            others['age_int'] += 1
            others['age'] = np.where(
                others['age_int'] >= 100, 'Y_GE100', 'Y' + others['age_int'].astype(int).astype(str)
            )
            others['Pop'] = others['pop_after_mig']
            others['TIME_PERIOD'] = year + 1
            others = others[['geo', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]

            pop_next = pd.concat([others, births_df], ignore_index=True)
            pop_next = (
                pop_next
                .groupby(['geo', 'sex', 'age', 'TIME_PERIOD'], as_index=False)['Pop']
                .sum()
            )
            pop_next['age_int'] = pop_next['age'].map(age_map)

            is_neg = pop_next['Pop'] < 0
            n_neg = int(is_neg.sum())
            total_neg = float((-pop_next.loc[is_neg, 'Pop']).sum())

            pop_next['Pop'] = pop_next['Pop'].clip(lower=0)

            process_log[year] = {
                'total_births': float(female_pop['births'].sum()),
                'total_immigrants': float(immigrants),
                'total_emmigrants': float(emmigrants),
                'total_deaths': float(deaths1 + deaths0),
                'n_negative_entries_corrected': n_neg,
                'total_negative_pop_corrected': total_neg
            }

            pop_dict[year + 1] = pop_next

        all_years = []
        for y in range(start_year, end_year + 1):
            dfy = pop_dict[y].copy()
            dfy['year'] = y
            all_years.append(dfy)
        final_proj_combo = pd.concat(all_years, ignore_index=True)
        final_proj_combo = final_proj_combo[['year', 'geo', 'sex', 'age', 'Pop']]
        final_proj_combo['Pop'] = final_proj_combo['Pop'].round().astype('int64')
        final_proj_combo['scenario'] = combo_label

        final_proj_list.append(final_proj_combo)
        process_log_all[combo_label] = process_log

    final_proj_all = pd.concat(final_proj_list, ignore_index=True)
    return final_proj_all, process_log_all


def main():
    df_init_pop, df_m, df_f, df_sr, df_mig = load_data()

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

    tag = f"{INIT_YEAR}_{END_YEAR}_{config.TOTAL_POPULATION_VINTAGE}"

    records = []
    for scenario, yearly_log in process_log.items():
        for year, metrics in yearly_log.items():
            row = {'scenario': scenario, 'year': year}
            row.update(metrics)
            records.append(row)

    df_log = pd.DataFrame(records)

    log_filename = os.path.join(PATH_OUT_XLSX, f"log_projected_AT_EU_{tag}.csv")
    print(f"Saving log to: {log_filename}")
    df_log.to_csv(log_filename, index=False)

    projections['scenario'] = projections['scenario'] + '-BSL'

    projections = projections.round(4)

    pqt_filename = config.PATH_TOTAL_POPULATION
    print(f"Saving projections to: {pqt_filename}")
    projections.to_parquet(pqt_filename, index=False)

    print(f"Total rows in projection: {len(projections)}")

    adjfert = projections[projections['scenario'].str.contains('BSL_ADJ2023')]
    print(f"Saving adjusted-fertility subset to: {config.PATH_TOTAL_POPULATION_ADJFERT}")
    adjfert.to_parquet(config.PATH_TOTAL_POPULATION_ADJFERT, index=False)
    print(f"Total rows in adjusted-fertility subset: {len(adjfert)}")

    print("Done.")


if __name__ == "__main__":
    main()
