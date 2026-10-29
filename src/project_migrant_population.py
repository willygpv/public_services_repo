"""Cohort-component projection of the foreign-national population, 2025-2051, with naturalisation and the nationality of births, for every migration scenario."""

import pandas as pd
import numpy as np
import itertools
import warnings

import config

warnings.filterwarnings('ignore')

INIT_YEAR = 2025
END_YEAR = 2051

PATH_TOT_POP = config.PATH_TOTAL_POPULATION
PATH_INIT_POP = config.PATH_FOREIGN_POPULATION_2025
PATH_MORTALITY = r'Data/Eurostat/mortality-AT_nuts3_age_sex_2019-2100.csv'
PATH_FERTILITY = r'Data/Eurostat/fertility_adjusted-AT_nuts3_age_2019-2050-foreigners_adjusted.csv'
PATH_SEX_RATIO = r'Data/Eurostat/sexratio-AT_nuts3_sex_2019-2050.csv'
PATH_MIGRATION = config.PATH_MIGRATION_SCENARIOS
PATH_NATURALIZATION = config.PATH_NATURALISATION

PATH_OUT_XLSX = r'XLSX/'
PATH_OUT_PQT = r'PQT/'


def load_and_prep_data():
    print("Loading data...")

    tot_df = pd.read_parquet(PATH_TOT_POP)

    init_pop = pd.read_csv(PATH_INIT_POP)
    df_init_pop = init_pop[init_pop['TIME_PERIOD'] == INIT_YEAR].copy()

    df_m = pd.read_csv(PATH_MORTALITY)

    df_f_eu = pd.read_csv(PATH_FERTILITY)
    df_f_eu = df_f_eu[df_f_eu['projection'] == 'BSL_ADJ2023']
    fert_keys = ['projection', 'geo', 'age', 'TIME_PERIOD']
    df_f_eu = df_f_eu[fert_keys + ['OBS_VALUE']].drop_duplicates()
    assert not df_f_eu.duplicated(fert_keys).any(), "conflicting fertility rates"

    df_sr = pd.read_csv(PATH_SEX_RATIO)

    df_mig = pd.read_parquet(PATH_MIGRATION)

    df_nat = pd.read_csv(PATH_NATURALIZATION)

    print("Data loading complete.")
    return df_init_pop, df_mig, df_m, df_f_eu, df_sr, df_nat, tot_df


def project_population(
    df_init_pop,
    df_mig,
    df_m,
    df_f_eu,
    df_sr,
    df_nat,
    tot_df,
    start_year=2023,
    end_year=2035,
    homophily=0.61
):

    all_ages = set(df_init_pop['age']).union(df_m['age']).union(df_mig['age']).union(
        df_nat['age']).union(df_f_eu['age']).union(tot_df['age'])

    age_map = {}
    for a in all_ages:
        if a == 'Y_LT1':
            age_map[a] = 0
        elif a == 'Y_GE100':
            age_map[a] = 100
        elif a == 'Y_LT15':
            age_map[a] = -1
        elif a == 'Y_GE50':
            age_map[a] = 50
        else:
            try:
                age_map[a] = int(a.replace('Y', ''))
            except (ValueError, TypeError):
                age_map[a] = -99

    df_init_pop['age_int'] = df_init_pop['age'].map(age_map)
    df_mig['age_int'] = df_mig['age'].map(age_map)
    df_m['age_int'] = df_m['age'].map(age_map)
    df_f_eu['age_int'] = df_f_eu['age'].map(age_map)
    df_nat['age_int'] = df_nat['age'].map(age_map)
    tot_df['age_int'] = tot_df['age'].map(age_map)

    sr_full = df_sr.set_index(['geo', 'TIME_PERIOD'])['Male to Female Ratio']

    pop_scenarios = df_init_pop['projection'].unique()
    mig_scenarios = df_mig['projection'].unique()
    mort_scenarios = df_m['projection'].unique()
    fert_scenarios = df_f_eu['projection'].unique()
    nat_scenarios = df_nat['projection'].unique()

    final_proj_list = []
    process_log_all = {}

    for pop_s, mig_s, mort_s, fert_s, nat_s in itertools.product(
        pop_scenarios, mig_scenarios, mort_scenarios, fert_scenarios, nat_scenarios
    ):
        combo_label = f"{pop_s}-{mig_s}-{mort_s}-{fert_s}-{nat_s}"
        print(f"Processing Scenario: {combo_label}")

        df_pop = df_init_pop[df_init_pop['projection'] == pop_s].copy()
        df_mig_s = df_mig[df_mig['projection'] == mig_s].copy()
        df_mort_s = df_m[df_m['projection'] == mort_s].copy()
        df_fert_s = df_f_eu[df_f_eu['projection'] == fert_s].copy()
        df_nat_s = df_nat[df_nat['projection'] == nat_s].copy()

        sr_s = sr_full

        mig_base = mig_s.split('__')[0]

        matching_tot_scenarios = [s for s in tot_df['scenario'].unique()
                                  if f'-{mig_base}-' in s]
        if not matching_tot_scenarios:
            print(f"Warning: No matching total population scenario for {mig_base}")
            continue

        tot_scenario = matching_tot_scenarios[0]
        df_tot_s = tot_df[tot_df['scenario'] == tot_scenario].copy()

        mort = df_mort_s.set_index(['geo', 'sex', 'age_int', 'TIME_PERIOD'])['OBS_VALUE']
        mig = df_mig_s.set_index(['geo', 'sex', 'age_int', 'TIME_PERIOD'])['net_nonA_by_age']
        nat = df_nat_s.set_index(['geo', 'sex', 'age_int'])['nat_rate']
        sr = sr_s
        fert = df_fert_s.set_index(['geo', 'age_int', 'TIME_PERIOD'])['OBS_VALUE']

        pop0 = (
            df_pop[df_pop['TIME_PERIOD'] == start_year]
            .query("sex in ['M','F']")
            .groupby(['geo', 'sex', 'age', 'age_int'])['OBS_VALUE']
            .sum()
            .reset_index()
            .rename(columns={'OBS_VALUE': 'Pop'})
        )
        pop0['TIME_PERIOD'] = start_year

        pop_dict = {start_year: pop0}
        process_log = {}

        for year in range(start_year, end_year):
            pop = pop_dict[year].copy().sort_values(['geo', 'sex', 'age_int']).reset_index(drop=True)

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
            contrib = net_mig
            immigrants = contrib[contrib > 0].sum()
            emmigrants = (-contrib[contrib < 0]).sum()

            nat_idx = pd.MultiIndex.from_frame(pop[['geo', 'sex', 'age_int']])
            nat_rates = nat.reindex(nat_idx).fillna(0).values
            pop['naturalizations'] = pop['pop_after_mig'] * nat_rates
            pop['pop_after_nat'] = pop['pop_after_mig'] - pop['naturalizations']
            total_naturalizations = pop['naturalizations'].sum()

            df_tot_year = df_tot_s[df_tot_s['year'] == year].copy()

            marriageable_age_min = 20
            marriageable_age_max = 65

            pop_males_marriageable = pop[
                (pop['sex'] == 'M') &
                (pop['age_int'].between(marriageable_age_min, marriageable_age_max))
            ]

            df_tot_year_males_marriageable = df_tot_year[
                (df_tot_year['sex'] == 'M') &
                (df_tot_year['age_int'].between(marriageable_age_min, marriageable_age_max))
            ]

            total_males_marriageable_by_geo = df_tot_year_males_marriageable.groupby('geo')['Pop'].sum()

            foreign_males_total_by_geo = pop_males_marriageable.groupby('geo')['pop_after_nat'].sum()

            austrian_males_by_geo = (total_males_marriageable_by_geo - foreign_males_total_by_geo).clip(lower=0).fillna(0)

            austrian_prop_geo_series = (austrian_males_by_geo / total_males_marriageable_by_geo).replace([float('inf'), -float('inf')], 0).fillna(0)
            austrian_male_props = austrian_prop_geo_series.to_dict()

            total_austrian_marriageable = austrian_males_by_geo.sum()
            total_marriageable_males = total_males_marriageable_by_geo.sum()

            mixing_rates = austrian_prop_geo_series * (1 - homophily)

            overall_austrian_male_prop = (
                total_austrian_marriageable / total_marriageable_males
                if total_marriageable_males > 0 else 0.0
            )

            females = pop[pop['sex'] == 'F'].copy()

            females['fert_age_group'] = np.where(
                females['age_int'] == 14, 'Y_LT15',
                np.where(females['age_int'] == 50, 'Y_GE50',
                         'Y' + females['age_int'].astype(str))
            )

            female_pop_by_fert_age = (
                females.groupby(['geo', 'fert_age_group'])['pop_after_nat']
                .sum()
                .reset_index()
            )

            female_pop_by_fert_age['fert_age_int'] = female_pop_by_fert_age['fert_age_group'].map(age_map)

            fert_year = df_fert_s[df_fert_s['TIME_PERIOD'] == year][['geo', 'age_int', 'OBS_VALUE']].copy()
            fert_year.rename(columns={'age_int': 'fert_age_int', 'OBS_VALUE': 'fert_rate'}, inplace=True)

            female_pop_by_fert_age = female_pop_by_fert_age.merge(
                fert_year,
                on=['geo', 'fert_age_int'],
                how='left'
            )
            female_pop_by_fert_age['fert_rate'] = female_pop_by_fert_age['fert_rate'].fillna(0)

            female_pop_by_fert_age['births'] = (
                female_pop_by_fert_age['pop_after_nat'] *
                female_pop_by_fert_age['fert_rate']
            )

            mixing_df = mixing_rates.rename('mix_rate').rename_axis('geo').reset_index()

            female_pop_by_fert_age = female_pop_by_fert_age.merge(
                mixing_df,
                on=['geo'],
                how='left'
            )
            female_pop_by_fert_age['mix_rate'] = female_pop_by_fert_age['mix_rate'].fillna(0)

            female_pop_by_fert_age['births_foreign'] = (
                female_pop_by_fert_age['births'] * (1 - female_pop_by_fert_age['mix_rate'])
            )
            female_pop_by_fert_age['births_austrian'] = (
                female_pop_by_fert_age['births'] * female_pop_by_fert_age['mix_rate']
            )

            female_pop = (
                female_pop_by_fert_age
                .groupby('geo')
                .agg({
                    'births_foreign': 'sum',
                    'births_austrian': 'sum',
                    'births': 'sum'
                })
                .rename(columns={'births': 'births_total'})
            )

            sr_idx = list(zip(female_pop.index.get_level_values(0), [year] * len(female_pop)))
            female_pop['sr'] = sr.reindex(sr_idx).fillna(1.05).values

            female_pop['female_births'] = female_pop['births_foreign'] / (1 + female_pop['sr'])
            female_pop['male_births'] = female_pop['births_foreign'] - female_pop['female_births']

            births_df = pd.concat([
                female_pop['male_births'].reset_index().assign(
                    sex='M', age='Y_LT1', age_int=0, TIME_PERIOD=year + 1,
                    Pop=lambda d: d['male_births']
                )[['geo', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']],
                female_pop['female_births'].reset_index().assign(
                    sex='F', age='Y_LT1', age_int=0, TIME_PERIOD=year + 1,
                    Pop=lambda d: d['female_births']
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
            others['Pop'] = others['pop_after_nat']
            others['TIME_PERIOD'] = year + 1
            others = others[['geo', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]

            pop_next = pd.concat([others, births_df], ignore_index=True)
            pop_next = (
                pop_next
                .groupby(['geo', 'sex', 'age', 'TIME_PERIOD'], as_index=False)['Pop']
                .sum()
            )
            pop_next['age_int'] = pop_next['age'].map(age_map)

            sigma = 5
            negatives = pop_next[pop_next['Pop'] < 0].copy()
            positives = pop_next[pop_next['Pop'] > 0].copy()

            if not negatives.empty and not positives.empty:
                key_cols = ['geo', 'sex']

                neg_data = negatives[key_cols + ['age_int', 'Pop']].rename(
                    columns={'age_int': 'age_neg', 'Pop': 'deficit'}
                )
                neg_data['deficit'] = -neg_data['deficit']

                pos_data = positives[key_cols + ['age_int', 'Pop']].reset_index().rename(
                    columns={'age_int': 'age_pos', 'index': 'pos_index'}
                )

                merged = pd.merge(neg_data, pos_data, on=key_cols)

                if not merged.empty:
                    age_diff = merged['age_pos'] - merged['age_neg']
                    weights = np.exp(-0.5 * (age_diff / sigma)**2)
                    merged['weighted_capacity'] = merged['Pop'] * weights

                    neg_group_cols = key_cols + ['age_neg']
                    total_capacity = merged.groupby(neg_group_cols)['weighted_capacity'].transform('sum')

                    merged['redistribution'] = (
                        (merged['weighted_capacity'] / total_capacity).fillna(0) * merged['deficit']
                    )

                    total_to_subtract = merged.groupby('pos_index')['redistribution'].sum()

                    positives['Pop'] = positives['Pop'].sub(total_to_subtract, fill_value=0)

                pop_next.update(positives)
                pop_next.loc[negatives.index, 'Pop'] = 0
            elif not negatives.empty:
                 pop_next.loc[negatives.index, 'Pop'] = 0

            is_neg = pop_next['Pop'] < 0
            n_neg = int(is_neg.sum())
            total_neg = float((-pop_next.loc[is_neg, 'Pop']).sum())

            pop_next['Pop'] = pop_next['Pop'].clip(lower=0)

            process_log[year] = {
                'total_births': float(female_pop['births_total'].sum()),
                'total_foreign_births': float(female_pop['births_foreign'].sum()),
                'total_austrian_births': float(female_pop['births_austrian'].sum()),
                'total_immigrants': float(immigrants),
                'total_emmigrants': float(emmigrants),
                'total_deaths': float(deaths1 + deaths0),
                'total_naturalizations': float(total_naturalizations),
                'n_negative_entries_corrected': n_neg,
                'total_negative_pop_corrected': total_neg,
                'overall_austrian_male_marriage_prop': float(overall_austrian_male_prop),
                'austrian_male_marriage_prop_by_geo': {geo: float(prop) for geo, prop in austrian_male_props.items()},
                'total_austrian_marriageable_males': int(total_austrian_marriageable),
                'total_marriageable_males': int(total_marriageable_males)
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


def frange(start, stop, step):
    return [round(start + i * step, 2) for i in range(int((stop - start) / step) + 1)]

def main():
    df_init_pop, df_mig, df_m, df_f_eu, df_sr, df_nat, tot_df = load_and_prep_data()

    homophily_values = [0.61]

    for h in homophily_values:
        print(f"\n--- Running projection for Homophily: {h} ---")

        projections, process_log = project_population(
            df_init_pop,
            df_mig,
            df_m,
            df_f_eu,
            df_sr,
            df_nat,
            tot_df,
            start_year=INIT_YEAR,
            end_year=END_YEAR,
            homophily=h
        )

        h_code = f"h{int(h * 100)}"

        records = []
        for scenario, yearly_log in process_log.items():
            for year, metrics in yearly_log.items():
                row = {'scenario': scenario, 'year': year}
                row.update(metrics)
                records.append(row)
        df_log = pd.DataFrame(records)

        assert h_code == 'h61', "config.MIGRANT_STEM is the h = 0.61 run"
        log_filename = config.PATH_PROJECTION_LOG
        pqt_filename = config.PATH_MIGRANT_PROJECTION

        print(f"Saving log to {log_filename}")
        df_log.to_csv(log_filename, index=False)

        print(f"Saving projections to {pqt_filename}")
        projections = projections.round(4)
        projections.to_parquet(pqt_filename, index=False)

    print("\nAll projections completed successfully.")

if __name__ == "__main__":
    main()
