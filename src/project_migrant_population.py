#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Migrant Forecast EU 2025-2051 (Differtility Model)
Refactored from Jupyter Notebook
"""

import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import warnings

# Suppress warnings for cleaner output (optional, based on notebook behavior)
warnings.filterwarnings('ignore')

# ==========================================
# Configuration & Constants
# ==========================================
INIT_YEAR = 2025
END_YEAR = 2051

# File Paths
PATH_TOT_POP = r'PQT/projected_AT_EU_2025_2051_20251204.parquet'
PATH_INIT_POP = r'XLSX/redistributed_ig_pop_2025.csv'
PATH_MORTALITY = r'Data/Eurostat/mortality-AT_nuts3_age_sex_2019-2100.csv'
PATH_FERTILITY = r'Data/Eurostat/fertility_adjusted-AT_nuts3_age_2019-2050-foreigners_adjusted.csv'
PATH_SEX_RATIO = r'Data/Eurostat/sexratio-AT_nuts3_sex_2019-2050.csv'
PATH_MIGRATION = r'PQT/dynres_net_migration_SensPreCrisis_SensCrisisEra_SensPostCovid_scenarios_20251203_123931.parquet'
PATH_NATURALIZATION = r'XLSX/redistributed_nat_2024.csv'

# Output Paths
PATH_OUT_XLSX = r'XLSX/'
PATH_OUT_PQT = r'PQT/'


# ==========================================
# Data Loading & Preprocessing
# ==========================================
def load_and_prep_data():
    print("Loading data...")

    # 1. Total Population
    tot_df = pd.read_parquet(PATH_TOT_POP)
    
    # 2. Initial Population
    ig_df = pd.read_csv(PATH_INIT_POP)
    # Filter only pop of initial year
    init_pop = ig_df[ig_df['TIME_PERIOD'] == INIT_YEAR]
    # Remove stateless
    df_init_pop = init_pop[init_pop['IncomeGroup'] != 'STATELESS'].copy()

    # 3. Mortality Rates
    df_m = pd.read_csv(PATH_MORTALITY)

    # 4. Fertility Rates
    df_f_eu = pd.read_csv(PATH_FERTILITY)
    # Correcting presumed syntax error in original script: '=' to '=='
    df_f_eu = df_f_eu[df_f_eu['projection'] == 'BSL_ADJ2023']

    # 5. Sex Ratio
    df_sr = pd.read_csv(PATH_SEX_RATIO)

    # 6. Net Migration
    df_mig = pd.read_parquet(PATH_MIGRATION)

    # 7. Naturalizations
    df_nat = pd.read_csv(PATH_NATURALIZATION)

    print("Data loading complete.")
    return df_init_pop, df_mig, df_m, df_f_eu, df_sr, df_nat, tot_df


# ==========================================
# Core Projection Logic
# ==========================================
def project_population(
    df_init_pop,
    df_mig,
    df_m,
    df_f_eu,  # Only Age-specific fertility rate data now
    df_sr,
    df_nat,
    tot_df,   # New parameter: total population data
    start_year=2023,
    end_year=2035,
    homophily_by_income=None
):
    """
    Projects population across all combinations of scenarios present in the input dataframes.
    Includes nationality assignment for births based on mixed partnerships and age-weighted 
    redistribution for negative population artifacts.
    """
    
    if homophily_by_income is None:
        homophily_by_income = {
            'High income': 0.5,
            'Lower middle income': 0.6,
            'Upper middle income': 0.6,
            'Low income': 0.7
        }

    # === OPTIMIZATION 1: Pre-calculate age_map and age_int columns ===
    # Build age integer mapping once, outside all loops.
    all_ages = set(df_init_pop['age']).union(df_m['age']).union(df_mig['age']).union(
        df_nat['age']).union(df_f_eu['age']).union(tot_df['age'])

    age_map = {}
    for a in all_ages:
        if a == 'Y_LT1':
            age_map[a] = 0
        elif a == 'Y_GE100':
            age_map[a] = 100
        elif a == 'Y_LT15':  # Special case for fertility data
            age_map[a] = -1  # Use -1 as identifier for under 15
        elif a == 'Y_GE50':  # Special case for fertility data
            age_map[a] = 50  # Use 50 as identifier for 50 and above
        else:
            try:
                # Handle standard 'Y25' format
                age_map[a] = int(a.replace('Y', ''))
            except (ValueError, TypeError):
                # Fallback for any other unexpected values
                age_map[a] = -99 

    # Apply mapping to all input dataframes ONCE.
    df_init_pop['age_int'] = df_init_pop['age'].map(age_map)
    df_mig['age_int'] = df_mig['age'].map(age_map)
    df_m['age_int'] = df_m['age'].map(age_map)
    df_f_eu['age_int'] = df_f_eu['age'].map(age_map)
    df_nat['age_int'] = df_nat['age'].map(age_map)
    tot_df['age_int'] = tot_df['age'].map(age_map)
    # =================================================================

    # === OPTIMIZATION 2: Pre-pivot Sex Ratio data ===
    # Create the pivoted Series once, outside the loop.
    sr_full = df_sr.set_index(['geo', 'TIME_PERIOD'])['Male to Female Ratio']
    # ================================================

    # Get unique scenario values
    pop_scenarios = df_init_pop['projection'].unique()
    mig_scenarios = df_mig['projection'].unique()
    mort_scenarios = df_m['projection'].unique()
    fert_scenarios = df_f_eu['projection'].unique()
    nat_scenarios = df_nat['projection'].unique()

    final_proj_list = []
    process_log_all = {}

    # Iterate over all scenario combinations
    for pop_s, mig_s, mort_s, fert_s, nat_s in itertools.product(
        pop_scenarios, mig_scenarios, mort_scenarios, fert_scenarios, nat_scenarios
    ):
        combo_label = f"{pop_s}-{mig_s}-{mort_s}-{fert_s}-{nat_s}"
        print(f"Processing Scenario: {combo_label}")

        # Filter dataframes for current scenario combination
        df_pop = df_init_pop[df_init_pop['projection'] == pop_s].copy()
        df_mig_s = df_mig[df_mig['projection'] == mig_s].copy()
        df_mort_s = df_m[df_m['projection'] == mort_s].copy()
        df_fert_s = df_f_eu[df_f_eu['projection'] == fert_s].copy()
        df_nat_s = df_nat[df_nat['projection'] == nat_s].copy()

        # === Use pre-pivoted SR data ===
        sr_s = sr_full
        # ===============================

        # Extract migration base scenario to match with tot_df
        mig_base = mig_s.split('__')[0]

        # Find matching tot_df scenario
        matching_tot_scenarios = [s for s in tot_df['scenario'].unique()
                                  if f'-{mig_base}-' in s]
        if not matching_tot_scenarios:
            print(f"Warning: No matching total population scenario for {mig_base}")
            continue

        tot_scenario = matching_tot_scenarios[0]
        df_tot_s = tot_df[tot_df['scenario'] == tot_scenario].copy()

        # Pivot for fast lookup
        mort = df_mort_s.set_index(['geo', 'sex', 'age_int', 'TIME_PERIOD'])['OBS_VALUE']
        mig = df_mig_s.set_index(['geo', 'sex', 'age_int', 'TIME_PERIOD'])['net_nonA_by_age']
        nat = df_nat_s.set_index(['geo', 'sex', 'age_int', 'IncomeGroup'])['nat_rate_by_income']
        sr = sr_s  # Use the (potentially filtered) sr_s Series
        fert = df_fert_s.set_index(['geo', 'age_int', 'IncomeGroup', 'TIME_PERIOD'])['OBS_VALUE']

        # Initialize population at start_year
        pop0 = (
            df_pop[df_pop['TIME_PERIOD'] == start_year]
            .query("sex in ['M','F']")
            .groupby(['geo', 'IncomeGroup', 'sex', 'age', 'age_int'])['OBS_VALUE']
            .sum()
            .reset_index()
            .rename(columns={'OBS_VALUE': 'Pop'})
        )
        pop0['TIME_PERIOD'] = start_year

        pop_dict = {start_year: pop0}
        process_log = {}

        # Simulation loop
        for year in range(start_year, end_year):
            pop = pop_dict[year].copy().sort_values(['geo', 'IncomeGroup', 'sex', 'age_int']).reset_index(drop=True)

            # Mortality
            idx = pd.MultiIndex.from_frame(pop[['geo', 'sex', 'age_int', 'TIME_PERIOD']])
            q = mort.reindex(idx).fillna(0).values
            pop['survivors'] = np.where(pop['age_int'] != 0, pop['Pop'] * (1 - q), pop['Pop'])
            deaths1 = (pop['Pop'] * q).sum()

            # Migration
            surv_by_ig = pop.groupby(['geo', 'IncomeGroup'])['survivors'].sum()
            surv_geo = pop.groupby('geo')['survivors'].sum()
            pop = pop.merge(surv_by_ig.rename('surv_ig').reset_index(), on=['geo', 'IncomeGroup'])
            pop = pop.merge(surv_geo.rename('surv_geo').reset_index(), on='geo')
            pop['share_IG'] = np.where(pop['surv_geo'] > 0, pop['surv_ig'] / pop['surv_geo'], 0)
            net_mig = mig.reindex(idx).fillna(0).values
            pop['pop_after_mig'] = pop['survivors'] + net_mig * pop['share_IG']
            contrib = net_mig * pop['share_IG']
            immigrants = contrib[contrib > 0].sum()
            emmigrants = (-contrib[contrib < 0]).sum()

            # Naturalization (after migration, before fertility/aging)
            nat_idx = pd.MultiIndex.from_frame(pop[['geo', 'sex', 'age_int', 'IncomeGroup']])
            nat_rates = nat.reindex(nat_idx).fillna(0).values
            pop['naturalizations'] = pop['pop_after_mig'] * nat_rates
            pop['pop_after_nat'] = pop['pop_after_mig'] - pop['naturalizations']
            total_naturalizations = pop['naturalizations'].sum()

            # Calculate Austrian population for fertility mixing
            # Get total population for current year
            df_tot_year = df_tot_s[df_tot_s['year'] == year].copy()

            # Filter for marriageable males (20-60) once for both dataframes
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

            # Calculate foreign males by income group and geography
            foreign_males_by_ig_geo = pop_males_marriageable.groupby(['geo', 'IncomeGroup'])['pop_after_nat'].sum()

            # Calculate total marriageable males by geography
            total_males_marriageable_by_geo = df_tot_year_males_marriageable.groupby('geo')['Pop'].sum()

            # Calculate foreign males total by geography
            foreign_males_total_by_geo = pop_males_marriageable.groupby('geo')['pop_after_nat'].sum()

            # Calculate Austrian males by geography (vectorized)
            austrian_males_by_geo = (total_males_marriageable_by_geo - foreign_males_total_by_geo).clip(lower=0).fillna(0)

            # Calculate proportion of Austrian males ready to marry in each geography
            austrian_prop_geo_series = (austrian_males_by_geo / total_males_marriageable_by_geo).replace([float('inf'), -float('inf')], 0).fillna(0)
            austrian_male_props = austrian_prop_geo_series.to_dict()

            # Accumulate for overall proportion calculation
            total_austrian_marriageable = austrian_males_by_geo.sum()
            total_marriageable_males = total_males_marriageable_by_geo.sum()

            # === OPTIMIZATION 3: Vectorized Mixing Rate Calculation ===
            # Convert the groupby Series to a DataFrame for easier merging
            mix_calc_df = foreign_males_by_ig_geo.reset_index()

            # Map geo-level and income-level data
            mix_calc_df['total_males_geo'] = mix_calc_df['geo'].map(total_males_marriageable_by_geo).fillna(0)
            mix_calc_df['austrian_males_geo'] = mix_calc_df['geo'].map(austrian_males_by_geo).fillna(0)
            mix_calc_df['homophily'] = mix_calc_df['IncomeGroup'].map(homophily_by_income).fillna(0.6) # Default 0.6

            # Calculate austrian share, handling division by zero
            mix_calc_df['austrian_share'] = (
                mix_calc_df['austrian_males_geo'] / mix_calc_df['total_males_geo']
            ).replace([np.inf, -np.inf], 0).fillna(0)

            # Calculate the final mixing rate
            mix_calc_df['mix_rate'] = mix_calc_df['austrian_share'] * (1 - mix_calc_df['homophily'])

            # Convert back to the dictionary format expected by the fertility section
            mixing_rates = mix_calc_df.set_index(['geo', 'IncomeGroup'])['mix_rate'].to_dict()
            # ==========================================================

            # Calculate overall proportion across all geographies
            overall_austrian_male_prop = (
                total_austrian_marriageable / total_marriageable_males
                if total_marriageable_males > 0 else 0.0
            )

            # Fertility calculation (ASFR method only) - VECTORIZED VERSION
            females = pop[pop['sex'] == 'F'].copy()

            # Vectorized age mapping to fertility groups
            females['fert_age_group'] = np.where(
                females['age_int'] < 15, 'Y_LT15',
                np.where(females['age_int'] >= 50, 'Y_GE50',
                         'Y' + females['age_int'].astype(str))
            )

            # Sum population by geo, IncomeGroup, and fertility age group
            female_pop_by_fert_age = (
                females.groupby(['geo', 'IncomeGroup', 'fert_age_group'])['pop_after_nat']
                .sum()
                .reset_index()
            )

            # Map fertility age groups to age_int for merging
            female_pop_by_fert_age['fert_age_int'] = female_pop_by_fert_age['fert_age_group'].map(age_map)

            # Create fertility lookup dataframe for the current year
            fert_year = df_fert_s[df_fert_s['TIME_PERIOD'] == year][['geo', 'age_int', 'IncomeGroup', 'OBS_VALUE']].copy()
            fert_year.rename(columns={'age_int': 'fert_age_int', 'OBS_VALUE': 'fert_rate'}, inplace=True)

            # Merge fertility rates
            female_pop_by_fert_age = female_pop_by_fert_age.merge(
                fert_year,
                on=['geo', 'IncomeGroup', 'fert_age_int'],
                how='left'
            )
            female_pop_by_fert_age['fert_rate'] = female_pop_by_fert_age['fert_rate'].fillna(0)

            # Calculate births
            female_pop_by_fert_age['births'] = (
                female_pop_by_fert_age['pop_after_nat'] *
                female_pop_by_fert_age['fert_rate']
            )

            # Vectorized mixing rate application
            # Convert mixing_rates dict to DataFrame for merging
            mixing_df = pd.DataFrame(
                [(geo, ig, rate) for (geo, ig), rate in mixing_rates.items()],
                columns=['geo', 'IncomeGroup', 'mix_rate']
            )

            female_pop_by_fert_age = female_pop_by_fert_age.merge(
                mixing_df,
                on=['geo', 'IncomeGroup'],
                how='left'
            )
            female_pop_by_fert_age['mix_rate'] = female_pop_by_fert_age['mix_rate'].fillna(0)

            # Calculate foreign and Austrian births
            female_pop_by_fert_age['births_foreign'] = (
                female_pop_by_fert_age['births'] * (1 - female_pop_by_fert_age['mix_rate'])
            )
            female_pop_by_fert_age['births_austrian'] = (
                female_pop_by_fert_age['births'] * female_pop_by_fert_age['mix_rate']
            )

            # Aggregate by geo and IncomeGroup
            female_pop = (
                female_pop_by_fert_age
                .groupby(['geo', 'IncomeGroup'])
                .agg({
                    'births_foreign': 'sum',
                    'births_austrian': 'sum',
                    'births': 'sum'
                })
                .rename(columns={'births': 'births_total'})
            )

            # Sex ratio calculation
            sr_idx = list(zip(female_pop.index.get_level_values(0), [year] * len(female_pop)))
            female_pop['sr'] = sr.reindex(sr_idx).fillna(1.05).values

            # Calculate sex-specific foreign births only
            female_pop['female_births'] = female_pop['births_foreign'] / (1 + female_pop['sr'])
            female_pop['male_births'] = female_pop['births_foreign'] - female_pop['female_births']

            # Births (only foreign births are added to the population)
            births_df = pd.concat([
                female_pop['male_births'].reset_index().assign(
                    sex='M', age='Y_LT1', age_int=0, TIME_PERIOD=year + 1,
                    Pop=lambda d: d['male_births']
                )[['geo', 'IncomeGroup', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']],
                female_pop['female_births'].reset_index().assign(
                    sex='F', age='Y_LT1', age_int=0, TIME_PERIOD=year + 1,
                    Pop=lambda d: d['female_births']
                )[['geo', 'IncomeGroup', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]
            ], ignore_index=True)

            # Age-0 mortality
            age0 = pop[pop['age_int'] == 0].copy()
            q0 = mort.reindex(pd.MultiIndex.from_frame(age0[['geo', 'sex', 'age_int', 'TIME_PERIOD']])).fillna(0).values
            age0['surv0'] = age0['pop_after_nat'] * (1 - q0)
            age1_df = age0.groupby(['geo', 'IncomeGroup', 'sex'])['surv0'].sum().reset_index().assign(
                age='Y1', age_int=1, TIME_PERIOD=year + 1, Pop=lambda d: d['surv0']
            )[['geo', 'IncomeGroup', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]

            # Aging
            others = pop[pop['age_int'] != 0].copy()
            others['age_int'] += 1
            others['age'] = np.where(
                others['age_int'] >= 100, 'Y_GE100', 'Y' + others['age_int'].astype(int).astype(str)
            )
            others['Pop'] = others['pop_after_nat']
            others['TIME_PERIOD'] = year + 1
            others = others[['geo', 'IncomeGroup', 'sex', 'age', 'age_int', 'TIME_PERIOD', 'Pop']]

            # Combine
            pop_next = pd.concat([others, age1_df, births_df], ignore_index=True)
            pop_next = (
                pop_next
                .groupby(['geo', 'IncomeGroup', 'sex', 'age', 'TIME_PERIOD'], as_index=False)['Pop']
                .sum()
            )
            pop_next['age_int'] = pop_next['age'].map(age_map)

            # === OPTIMIZATION 4: Vectorized Redistribution ===
            sigma = 5
            negatives = pop_next[pop_next['Pop'] < 0].copy()
            positives = pop_next[pop_next['Pop'] > 0].copy()

            if not negatives.empty and not positives.empty:
                key_cols = ['geo', 'IncomeGroup', 'sex']

                # Prepare negatives data
                neg_data = negatives[key_cols + ['age_int', 'Pop']].rename(
                    columns={'age_int': 'age_neg', 'Pop': 'deficit'}
                )
                neg_data['deficit'] = -neg_data['deficit']

                # Prepare positives data (with original index for later subtraction)
                pos_data = positives[key_cols + ['age_int', 'Pop']].reset_index().rename(
                    columns={'age_int': 'age_pos', 'index': 'pos_index'}
                )

                # Merge to find all pos-neg pairs in the same group
                merged = pd.merge(neg_data, pos_data, on=key_cols)

                if not merged.empty:
                    # Calculate weights and capacity
                    age_diff = merged['age_pos'] - merged['age_neg']
                    weights = np.exp(-0.5 * (age_diff / sigma)**2)
                    merged['weighted_capacity'] = merged['Pop'] * weights

                    # Calculate total capacity per negative group
                    neg_group_cols = key_cols + ['age_neg']
                    total_capacity = merged.groupby(neg_group_cols)['weighted_capacity'].transform('sum')

                    # Calculate redistribution amount for each positive "donor"
                    merged['redistribution'] = (
                        (merged['weighted_capacity'] / total_capacity).fillna(0) * merged['deficit']
                    )

                    # Sum all subtractions for each positive index
                    total_to_subtract = merged.groupby('pos_index')['redistribution'].sum()

                    # Apply subtractions to positives
                    positives['Pop'] = positives['Pop'].sub(total_to_subtract, fill_value=0)

                # Update pop_next with new positive values and zero-out negatives
                pop_next.update(positives)
                pop_next.loc[negatives.index, 'Pop'] = 0
            elif not negatives.empty:
                 # No positives to redistribute to, just zero out negatives
                 pop_next.loc[negatives.index, 'Pop'] = 0
            # ======================================================

            # Compute negatives before clipping (for logging)
            is_neg = pop_next['Pop'] < 0
            n_neg = int(is_neg.sum())
            total_neg = float((-pop_next.loc[is_neg, 'Pop']).sum())

            # Clip any remaining micro-negatives (from float errors)
            pop_next['Pop'] = pop_next['Pop'].clip(lower=0)

            # Log
            deaths0 = (age0['pop_after_nat'] * q0).sum()
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

        # Compile final projection for this combination
        all_years = []
        for y in range(start_year, end_year + 1):
            dfy = pop_dict[y].copy()
            dfy['year'] = y
            all_years.append(dfy)
        final_proj_combo = pd.concat(all_years, ignore_index=True)
        final_proj_combo = final_proj_combo[['year', 'geo', 'IncomeGroup', 'sex', 'age', 'Pop']]
        final_proj_combo['Pop'] = final_proj_combo['Pop'].round().astype(int)
        final_proj_combo['scenario'] = combo_label

        final_proj_list.append(final_proj_combo)
        process_log_all[combo_label] = process_log

    final_proj_all = pd.concat(final_proj_list, ignore_index=True)
    return final_proj_all, process_log_all


# ==========================================
# Main Execution
# ==========================================
def frange(start, stop, step):
    return [round(start + i * step, 2) for i in range(int((stop - start) / step) + 1)]

def main():
    # 1. Load Data
    df_init_pop, df_mig, df_m, df_f_eu, df_sr, df_nat, tot_df = load_and_prep_data()

    # 2. Define Parameters
    today = datetime.now().strftime("%Y%m%d")
    homophily_values = [0.61]  # As set in the original script

    # 3. Execution Loop
    for h in homophily_values:
        print(f"\n--- Running projection for Homophily: {h} ---")
        
        # Define homophily for all income groups
        homophily_by_income = {
            'High income': h,
            'Lower middle income': h,
            'Upper middle income': h,
            'Low income': h
        }

        # Run the population projection
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
            homophily_by_income=homophily_by_income
        )

        # Create short identifier from homophily value
        h_code = f"h{int(h * 100)}"  # e.g., h61

        # Convert process_log to DataFrame
        records = []
        for scenario, yearly_log in process_log.items():
            for year, metrics in yearly_log.items():
                row = {'scenario': scenario, 'year': year}
                row.update(metrics)
                records.append(row)
        df_log = pd.DataFrame(records)

        # 4. Save Outputs
        log_filename = f"{PATH_OUT_XLSX}log_negtest_projected_mig_EU_{INIT_YEAR}_{END_YEAR}_{h_code}_{today}_adjfert.csv"
        pqt_filename = f"{PATH_OUT_PQT}negtest_projected_mig_EU_{INIT_YEAR}_{END_YEAR}_{h_code}_{today}_adjfert.parquet"

        print(f"Saving log to {log_filename}")
        df_log.to_csv(log_filename, index=False)

        print(f"Saving projections to {pqt_filename}")
        projections = projections.round(4)
        projections.to_parquet(pqt_filename, index=False)

    print("\nAll projections completed successfully.")

if __name__ == "__main__":
    main()