#!/usr/bin/env python
# coding: utf-8

"""
Education Demand Projection Script (Binomial + Empirical + Teacher Demand)
Refactored from Jupyter Notebook: education_demand-EU_scenarios_20251124 - bi-emp.py
Date: 2026-02-04
"""

import os
import gc
import re
import time
import logging
import warnings
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from patsy import dmatrix
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

BASE_YEAR = 2023
DEMAND_YEAR = 2050
VALIDATION_YEARS = [2023]
YEARS_HISTORIC = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Inputs
PATH_INPUT_RATIO = r'XLSX/df_ratio_nuts3_cob_shift_mod-type.csv'
PATH_INPUT_TEACHERS = r'XLSX/df_tpers_st_nuts3_mod-type.csv'
PATH_GEOJSON = r'GEOJSON/NUTS_RG_03M_2024_3035.geojson'

# Projection Input Path (UPDATED from new file)
PATH_PROJECTION_PQT = r'PQT/negtest_projected_simEU_negtest_projected_mig_EU_2025_2051_h61_20251204_adjfert_adjasfr.parquet'
PATH_EXTRAPOLATION_2025 = r'Data/Statcube/population-2025_nuts3_agegroups_nat.csv'

# Outputs
PATH_OUTPUT_DIR = r'PQT'
PATH_SUMMARY_CSV = r'XLSX/education_summary.csv'
TEMP_DIR_TEACHERS = "temp_teacher_projection"

# ==========================================
# 2. DATA PREPARATION
# ==========================================

def load_geometries():
    print("--- Loading Geometries ---")
    try:
        eu_gdf = gpd.read_file(PATH_GEOJSON)
        at_gdf = eu_gdf[(eu_gdf['CNTR_CODE'] == 'AT') & (eu_gdf['LEVL_CODE'] == 3)].copy()
        return at_gdf[['NUTS_ID', 'geometry']].rename(columns={'NUTS_ID': 'NUTS3'})
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        return None

def load_historic_data():
    print("--- Loading Historic Data ---")
    if not os.path.exists(PATH_INPUT_RATIO) or not os.path.exists(PATH_INPUT_TEACHERS):
        raise FileNotFoundError("Input files from previous step not found.")
        
    df_ratio = pd.read_csv(PATH_INPUT_RATIO)
    df_teachers = pd.read_csv(PATH_INPUT_TEACHERS)
    df_teachers['scenario'] = 'BSL'
    return df_ratio, df_teachers

def prepare_population_projections():
    print("--- Preparing Population Projections ---")
    # 1. Main Projection
    projected_all = pd.read_parquet(PATH_PROJECTION_PQT)
    projected_all.rename(columns={'sex': 'Sex', 'age': 'Age', 'OBS_VALUE': 'Population', 
                                  'geo': 'NUTS3', 'TIME_PERIOD': 'Year'}, inplace=True)

    def convert_age(x):
        if x == 'Y_GE100': return 100
        elif x == 'Y_LT1': return 0
        return int(str(x).lstrip('Y'))

    projected_all['Age'] = projected_all['Age'].apply(convert_age)
    projected_all['Age'] = projected_all['Age'].apply(lambda x: 5 if x <= 5 else (50 if x >= 50 else x))
    projected_all['Population'] = projected_all['Population'].clip(lower=0)
    
    projected_all = projected_all.groupby(['Year', 'NUTS3', 'Age', 'Nationality', 'scenario'], as_index=False)['Population'].sum()
    projected_all = projected_all[projected_all['Year'] <= DEMAND_YEAR]

    # 2. Extrapolation
    extra_df = pd.read_csv(PATH_EXTRAPOLATION_2025, skiprows=8, encoding='latin-1')
    if 'Values' in extra_df.columns:
        symbol_idx = extra_df[extra_df['Values'] == "Symbol"].index
        if not symbol_idx.empty: extra_df = extra_df.iloc[:symbol_idx[0]]

    extra_df['Number'] = pd.to_numeric(extra_df['Number'].replace('-', '0'), errors='coerce').fillna(0)
    extra_df.columns = ['Values', 'Year', 'NUTS3', 'Age', 'Nationality', 'Population', 'Annotations', 'Values_drop']
    extra_df = extra_df[extra_df['NUTS3'] != 'Not classifiable <0>']
    extra_df['Year'] = pd.to_numeric(extra_df['Year'], errors='coerce')
    extra_df['NUTS3'] = extra_df['NUTS3'].str.extract(r'<(.*?)>')
    extra_df = extra_df[extra_df['Age'] != 'Not applicable']

    def map_age_bin(age_str):
        match = re.search(r"\d+", str(age_str))
        num = int(match.group()) if match else None
        if "under" in str(age_str): return 5
        elif "plus" in str(age_str): return 50
        elif "to" in str(age_str): return 50 if num >= 50 else num
        elif num is not None: return 5 if num <= 5 else (50 if num >= 50 else num)
        return np.nan

    extra_df["Age"] = extra_df["Age"].apply(map_age_bin)
    extra_df = extra_df.groupby(["Year", "NUTS3", "Nationality", "Age"], as_index=False)["Population"].sum()
    extra_df['scenario'] = 'extrapolated'
    extra_df['Year'] = 2024

    return pd.concat([projected_all, extra_df], ignore_index=True)

# ==========================================
# 3. ENROLLMENT MODEL
# ==========================================

def define_age_groups(age_series):
    bins = [4, 5, 9, 13, 17, 51] 
    labels = ['5', '6-9', '10-13', '14-17', '18+']
    return pd.cut(age_series, bins=bins, labels=labels, right=True)

def apply_projection(df_in, pred_results, school_cols):
    population = df_in['Population'].values
    df_out = df_in.copy()
    total_ratio = (pred_results['Base_Rate'] + pred_results['Surplus_Component'])
    total_students = population * total_ratio
    
    school_distributions = pred_results[school_cols]
    for col in school_cols:
        valid_props = school_distributions[col].fillna(0)
        df_out[col] = total_students * valid_props
    return df_out

class EnrollmentModelEmpirical:
    def __init__(self, school_type_cols):
        self.school_type_cols = school_type_cols
        self.stage1_formula = {
            "sq": "C(Nationality) + C(Age) + C(NUTS3) + C(Age):C(Nationality)",
            "trend": "C(Nationality) + C(Age) + C(NUTS3) + C(Age):C(Nationality) + Year_std"
        }
        self.final_model_sq_base = None
        self.final_model_trend_base = None
        self.final_design_info_sq_base = None
        self.final_design_info_trend_base = None
        self.surplus_lookup_table = None
        self.empirical_distributions = None
        self.age_stats = None
        self.year_stats = None

    def _prepare_wide_data(self, df_historical, is_full_fit=True):
        df_wide = df_historical.pivot_table(
            index=['Year', 'NUTS3', 'Age', 'Nationality'],
            columns='School_Type', values='Students', fill_value=0
        ).reset_index()
        
        df_pop = df_historical[['Year', 'NUTS3', 'Age', 'Nationality', 'Population']].drop_duplicates()
        df_wide = pd.merge(df_wide, df_pop, on=['Year', 'NUTS3', 'Age', 'Nationality'])

        df_wide['Total_Students'] = df_wide[self.school_type_cols].sum(axis=1)
        df_wide['Total_Students_Int'] = df_wide['Total_Students'].round().astype(int).clip(lower=1)
        df_wide['Population_Int'] = df_wide['Population'].round().astype(int)
        df_wide['Enrolled_Count'] = np.minimum(df_wide['Total_Students'].round().astype(int), df_wide['Population_Int'])
        df_wide['Total_Enrollment_Ratio'] = df_wide['Total_Students'] / df_wide['Population'].clip(lower=1)
        df_wide['Surplus_Rate'] = np.maximum(0, df_wide['Total_Enrollment_Ratio'] - 1.0)

        if self.age_stats is None:
            self.age_stats = {'mean': df_wide['Age'].mean(), 'std': df_wide['Age'].std()}
        if self.year_stats is None:
            self.year_stats = {'mean': df_wide['Year'].mean(), 'std': df_wide['Year'].std()}

        df_wide['Age_std'] = (df_wide['Age'] - self.age_stats['mean']) / self.age_stats['std']
        df_wide['Year_std'] = (df_wide['Year'] - self.year_stats['mean']) / self.year_stats['std']
        df_wide['const'] = 1.0
        return df_wide

    def _fit_binomial_glm(self, df_agg, formula):
        y_successes = df_agg['Enrolled_Count'].values
        y_trials = df_agg['Population_Int'].values
        y_failures = np.maximum(0, y_trials - y_successes)
        y_binomial = np.column_stack([y_successes, y_failures])
        try:
            X_dm = dmatrix(formula, df_agg, return_type='dataframe')
            model = sm.GLM(y_binomial, X_dm, family=sm.families.Binomial())
            result = model.fit()
            return result, result.aic, X_dm.design_info
        except:
            return None, np.inf, None

    def _build_surplus_lookup(self, df_wide):
        return df_wide.groupby(['Age', 'NUTS3', 'Nationality'])['Surplus_Rate'].mean().reset_index()

    def _build_empirical_distributions(self, df_wide):
        cohort_cols = ['NUTS3', 'Age', 'Nationality']
        df_sums = df_wide.groupby(cohort_cols)[self.school_type_cols + ['Total_Students']].sum().reset_index()
        empirical_dists = df_sums.copy()
        total_students_sum = empirical_dists['Total_Students'].clip(lower=1)
        for school in self.school_type_cols:
            empirical_dists[school] = empirical_dists[school] / total_students_sum
        return empirical_dists[cohort_cols + self.school_type_cols]

    def _get_empirical_distribution(self, df_projection):
        cohort_cols = ['NUTS3', 'Age', 'Nationality']
        df_with_dist = df_projection.merge(self.empirical_distributions, on=cohort_cols, how='left')
        return df_with_dist[self.school_type_cols]

    def _predict_stage1_base(self, model, design_info, df_projection, scenario_type):
        df_proj = df_projection.copy()
        df_proj['Age_std'] = (df_proj['Age'] - self.age_stats['mean']) / self.age_stats['std']
        df_proj['Year_std'] = 0.0 if scenario_type == 'statusquo' else (df_proj['Year'] - self.year_stats['mean']) / self.year_stats['std']
        df_proj['const'] = 1.0
        try:
            X_proj = dmatrix(design_info, df_proj, return_type='dataframe')
            return pd.DataFrame({'Base_Rate': model.predict(X_proj)})
        except:
            return None

    def run_validation(self, df_historical, validation_years):
        val_year = validation_years[0]
        df_train = df_historical[df_historical['Year'] < val_year]
        self.age_stats = None
        self.year_stats = None
        df_train_wide = self._prepare_wide_data(df_train, is_full_fit=False)
        self.empirical_distributions = self._build_empirical_distributions(df_train_wide)
        model, _, _ = self._fit_binomial_glm(df_train_wide, self.stage1_formula['sq'])
        return model is not None

    def fit_final_models(self, df_historical):
        df_wide = self._prepare_wide_data(df_historical, is_full_fit=True)
        self.final_model_sq_base, _, self.final_design_info_sq_base = self._fit_binomial_glm(df_wide, self.stage1_formula['sq'])
        self.final_model_trend_base, _, self.final_design_info_trend_base = self._fit_binomial_glm(df_wide, self.stage1_formula['trend'])
        self.surplus_lookup_table = self._build_surplus_lookup(df_wide)
        self.empirical_distributions = self._build_empirical_distributions(df_wide)
        return self.final_model_sq_base is not None

    def project(self, df_projection, model_scenario_type, final_scenario_name):
        model = self.final_model_sq_base if model_scenario_type == 'statusquo' else self.final_model_trend_base
        design = self.final_design_info_sq_base if model_scenario_type == 'statusquo' else self.final_design_info_trend_base
        
        cols = ['Year', 'NUTS3', 'Age', 'Nationality', 'Population']
        df_proj_unique = df_projection[cols].drop_duplicates().reset_index(drop=True)
        
        pred_base = self._predict_stage1_base(model, design, df_proj_unique, model_scenario_type)
        if pred_base is None: return None
        
        df_surplus = df_proj_unique.merge(self.surplus_lookup_table, on=['Age', 'NUTS3', 'Nationality'], how='left')
        pred_results = pd.DataFrame({
            'Base_Rate': pred_base['Base_Rate'],
            'Surplus_Component': df_surplus['Surplus_Rate'].fillna(0.0)
        })
        
        emp_dist = self._get_empirical_distribution(df_proj_unique).reset_index(drop=True)
        pred_results = pd.concat([pred_results, emp_dist], axis=1)
        df_projected = apply_projection(df_proj_unique, pred_results, self.school_type_cols)
        
        # NOTE: Using 'Projected_Students' as value name to match original
        df_long = df_projected.melt(id_vars=cols, value_vars=self.school_type_cols, 
                                    var_name='School_Type', value_name='Projected_Students')
        df_long['scenario'] = final_scenario_name
        return df_long

# ==========================================
# 4. RUNNER FUNCTIONS
# ==========================================

def run_empirical_model(df_historical, df_projection, validation_years=[2023]):
    def optimize_memory(df):
        for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype('category')
        for col in df.select_dtypes(include=['float64']).columns: df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns: df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    school_cols = df_historical['School_Type'].unique().tolist()
    model = EnrollmentModelEmpirical(school_cols)
    if not model.run_validation(df_historical, validation_years): return None, None
    if not model.fit_final_models(df_historical): return None, None

    print("\n=== GENERATING PROJECTIONS ===")
    all_projections = []
    input_scenarios = df_projection['scenario'].unique()
    
    for i, input_scen in enumerate(input_scenarios):
        if i % 5 == 0: gc.collect()
        df_subset = df_projection[df_projection['scenario'] == input_scen].copy()
        
        df_sq = model.project(df_subset, 'statusquo', f"{input_scen}--statusquo") # Matches new file naming
        if df_sq is not None: all_projections.append(optimize_memory(df_sq))
        
        df_trend = model.project(df_subset, 'trend', f"{input_scen}--trend") # Matches new file naming
        if df_trend is not None: all_projections.append(optimize_memory(df_trend))

    if not all_projections: return None, None
    
    df_final_projections = pd.concat(all_projections, ignore_index=True)
    del all_projections
    gc.collect()
    
    # Historic Data Setup (Renaming to match projection column 'Projected_Students')
    df_hist = df_historical.copy()
    df_hist['scenario'] = 'historic'
    df_hist = df_hist.rename(columns={'Students': 'Projected_Students'})
    df_hist = optimize_memory(df_hist)
    
    # Combine
    df_final = pd.concat([df_hist, df_final_projections], ignore_index=True)
    df_final = optimize_memory(df_final)
    
    # Filter columns to drop intermediate calculations (Removing 'Enrollment_Ratio' capital R)
    final_columns = ['Year', 'NUTS3', 'Age', 'Nationality', 'scenario',
                     'School_Type', 'Population', 'Projected_Students']
    existing_cols = [c for c in final_columns if c in df_final.columns]
    df_final = df_final[existing_cols]

    # Sort
    df_final = df_final.sort_values(by=['Year', 'NUTS3', 'Age', 'Nationality'])

    # Calc Enrollment Ratio (lowercase 'r')
    print("Calculating final enrollment ratios...")
    group_cols = ['Year', 'NUTS3', 'Age', 'Nationality', 'scenario']
    df_agg = df_final.groupby(group_cols, observed=True).agg(
        Total_Projected_Students=('Projected_Students', 'sum'),
        Population=('Population', 'first')
    ).reset_index()
    
    df_agg['Enrollment_ratio'] = np.where(
        df_agg['Population'] > 0,
        df_agg['Total_Projected_Students'] / df_agg['Population'],
        0.0
    ).astype('float32')
    
    df_final = df_final.merge(df_agg[group_cols + ['Enrollment_ratio']], on=group_cols, how='left')
    
    # Final Rename back to 'Students' as per external script usage
    df_final = df_final.rename(columns={'Projected_Students': 'Students'})
    
    return df_final, model

def prepare_teacher_projection_hyper_efficient(df_stu, df_tp, base_year, temp_dir):
    print("\n=== CALCULATING TEACHER DEMAND ===")
    Path(temp_dir).mkdir(exist_ok=True, parents=True)
    for f in Path(temp_dir).glob("*.parquet"): f.unlink()

    TEACHER_DEMO_COLS = ['NUTS3', 'School_Type']
    STUDENT_DEMO_COLS = ['NUTS3', 'School_Type', 'Age', 'Nationality']

    def optimize_df(df):
        for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype('category')
        for col in df.select_dtypes(include=['int64']).columns: df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float64']).columns: df[col] = df[col].astype('float32')
        return df

    df_stu = optimize_df(df_stu)
    df_tp = optimize_df(df_tp)

    # Base Ratios
    base_tp_scen = df_tp[df_tp['Year'] == base_year]['scenario'].unique()[0]
    base_tp_data = df_tp[(df_tp['Year'] == base_year) & (df_tp['scenario'] == base_tp_scen)].copy()
    base_ratios = base_tp_data[TEACHER_DEMO_COLS + ['Student-to-teacher']].rename(columns={'Student-to-teacher': 'Base_StT_Ratio'})

    # Historical Lookup
    hist_years = df_tp['Year'].unique()
    hist_lookup = df_tp[df_tp['Year'].isin(hist_years)].groupby(['Year'] + TEACHER_DEMO_COLS, observed=True)['Teachers'].sum().reset_index()
    hist_lookup.rename(columns={'Teachers': 'Actual_Teachers_Hist'}, inplace=True)

    scenarios = df_stu['scenario'].unique()
    result_files = []

    for i, scen in enumerate(scenarios):
        if i % 10 == 0: 
            print(f"Teacher calculation: Scenario {i+1}/{len(scenarios)}...", end="\r")
            gc.collect()

        df_chunk = df_stu[df_stu['scenario'] == scen].copy()
        
        # 1. Future Years (Ratio Based)
        df_chunk = df_chunk.merge(base_ratios, on=TEACHER_DEMO_COLS, how='left')
        st = df_chunk['Students'].values
        ratio = df_chunk['Base_StT_Ratio'].values
        teachers_calc = np.zeros_like(st, dtype=np.float32)
        mask = (ratio > 0) & (~np.isnan(ratio))
        teachers_calc[mask] = st[mask] / ratio[mask]
        df_chunk['Teachers_fraction'] = teachers_calc

        # 2. Historical Years (Actual Based)
        hist_mask = df_chunk['Year'].isin(hist_years)
        if hist_mask.any():
            df_hist_part = df_chunk[hist_mask].copy().merge(hist_lookup, on=['Year'] + TEACHER_DEMO_COLS, how='left')
            grp_sum = df_hist_part.groupby(['Year'] + TEACHER_DEMO_COLS, observed=True)['Students'].transform('sum')
            
            actual_teachers = df_hist_part['Actual_Teachers_Hist'].fillna(0).values
            my_students = df_hist_part['Students'].values
            total_students = grp_sum.values
            
            corrected = np.zeros_like(my_students, dtype=np.float32)
            valid = total_students > 0
            corrected[valid] = (my_students[valid] / total_students[valid]) * actual_teachers[valid]
            df_chunk.loc[hist_mask, 'Teachers_fraction'] = corrected

        # 3. Demand Calculation
        base_mask = df_chunk['Year'] == base_year
        base_lookup = df_chunk.loc[base_mask, STUDENT_DEMO_COLS + ['Teachers_fraction']].rename(columns={'Teachers_fraction': 'Base_Year_Ref'})
        df_chunk = df_chunk.merge(base_lookup, on=STUDENT_DEMO_COLS, how='left')
        df_chunk['Demand_teachers'] = df_chunk['Teachers_fraction'] - df_chunk['Base_Year_Ref'].fillna(0)

        # 4. Finalize
        df_chunk.drop(columns=['Base_StT_Ratio', 'Base_Year_Ref', 'Actual_Teachers_Hist'], errors='ignore', inplace=True)
        final_scen_name = f"{scen}--{base_tp_scen}"
        df_chunk['scenario'] = final_scen_name
        df_chunk['scenario'] = df_chunk['scenario'].astype('category')
        
        # MATCHING ORIGINAL: Explicitly add scenario_tpers
        df_chunk['scenario_tpers'] = base_tp_scen
        df_chunk['scenario_tpers'] = df_chunk['scenario_tpers'].astype('category')

        out_file = Path(temp_dir) / f"t_scen_{i}.parquet"
        df_chunk.to_parquet(out_file, index=False)
        result_files.append(out_file)

    print("\nCombining teacher projection files...")
    dfs = [pd.read_parquet(f) for f in result_files]
    df_final = pd.concat(dfs, ignore_index=True)
    return df_final

def aggregate_summary_statistics(df_teach_projections):
    """Generates the education_summary.csv file matching the notebook logic."""
    print("\n--- Generating Summary Statistics ---")
    cohort_cols_final = ['Year', 'NUTS3', 'Age', 'Nationality']
    group_cols_sum = cohort_cols_final + ['scenario']

    df_clean = df_teach_projections.copy()
    df_clean['Teachers_fraction_clean'] = pd.to_numeric(df_clean['Teachers_fraction'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Teachers_fraction_clean'])

    # Sum Teachers_fraction across School_Types
    df_summed = (
        df_clean
        .groupby(group_cols_sum, observed=True)['Teachers_fraction_clean']
        .sum()
        .rename('Teachers_fraction_sum')
        .reset_index()
    )

    # Calculate Median, Max, Min across scenarios
    df_agg_teach = (
        df_summed
        .groupby(cohort_cols_final, observed=True)['Teachers_fraction_sum']
        .agg(
            Teachers_fraction_median=('median'),
            Teachers_fraction_max=('max'),
            Teachers_fraction_min=('min')
        )
        .reset_index()
    )
    
    os.makedirs(os.path.dirname(PATH_SUMMARY_CSV), exist_ok=True)
    df_agg_teach.to_csv(PATH_SUMMARY_CSV, index=False)
    print(f"Summary saved to {PATH_SUMMARY_CSV}")

# ==========================================
# 5. MAIN
# ==========================================

def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    try:
        df_ratio, df_teachers = load_historic_data()
        df_projection = prepare_population_projections()
        
        df_stu_projections, model = run_empirical_model(df_ratio, df_projection, validation_years=VALIDATION_YEARS)
        if df_stu_projections is None:
            print("Projection failed.")
            return

        df_teach_projections = prepare_teacher_projection_hyper_efficient(
            df_stu_projections, df_teachers, BASE_YEAR, TEMP_DIR_TEACHERS
        )
        if df_teach_projections is None:
            print("Teacher projection failed.")
            return

        # Generate summary CSV
        aggregate_summary_statistics(df_teach_projections)

        # Save main parquet output
        filename_no_ext = os.path.splitext(os.path.basename(PATH_PROJECTION_PQT))[0]
        output_path = os.path.join(PATH_OUTPUT_DIR, f'teacher_projections_{filename_no_ext}_NEWTEST-type.parquet')
        
        print(f"Saving main projection to {output_path}...")
        df_teach_projections.to_parquet(output_path, index=False, compression='gzip')
        print("Done.")
        
    except Exception as e:
        print(f"Critical Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()