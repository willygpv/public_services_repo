#!/usr/bin/env python
"""
Demographic Projection: Multi-Scenario Sensitivity Analysis

Generates NUTS-3 regional migration projections that reconcile pre-crisis regional
trends (EUROPOP2019) with updated post-crisis national aggregates (EUROPOP2023),
using a hybrid top-down disaggregation approach.

Outputs:
  - Parquet file: full scenario projections (PQT/)
  - CSV file: collapsed scenario averages (XLSX/)
"""

import re
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_and_clean_statcube_csv(filepath, geo_col_index=2, header_keyword="Number",
                                encoding="latin-1"):
    """
    Load and clean a STATcube CSV file by dynamically finding the header row.

    Args:
        filepath: Path to the CSV file.
        geo_col_index: Column index containing NUTS3 codes in angle brackets.
        header_keyword: Keyword uniquely identifying the header row.
        encoding: File encoding.
    """
    skiprows_num = 0
    with open(filepath, "r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if header_keyword in line:
                skiprows_num = i
                break
        else:
            raise ValueError(
                f"Header keyword '{header_keyword}' not found in file: {filepath}"
            )

    df = pd.read_csv(filepath, skiprows=skiprows_num, encoding=encoding)

    # Drop footer rows
    try:
        symbol_index = df[df[df.columns[0]] == "Symbol"].index[0]
        df = df.iloc[:symbol_index]
    except IndexError:
        pass

    df["Time section"] = df["Time section"].astype(float)
    df["Number"] = df["Number"].replace("-", "0").astype(float)
    df.iloc[:, geo_col_index] = df.iloc[:, geo_col_index].str.extract(r"<(.*?)>")
    df["Sex"] = df["Sex"].replace({"male": "M", "female": "F"})
    df = df.drop(
        columns=["Unnamed: 8", "Unnamed: 7", "Unnamed: 6", "Annotations"],
        errors="ignore",
    )
    return df


def load_and_clean_historical_csv(filepath, skiprows=5, encoding="latin-1"):
    """Load and clean a historical STATcube CSV (fixed-format variant)."""
    df = pd.read_csv(filepath, skiprows=skiprows, encoding=encoding)
    symbol_index = df[df["Time section"] == "Symbol"].index[0]
    df = df.iloc[:symbol_index]

    df["Time section"] = df["Time section"].astype(float)
    df["Number"] = df["Number"].replace("-", "0").astype(float)
    df.iloc[:, 1] = df.iloc[:, 1].str.extract(r"<(.*?)>")
    df["Sex"] = df["Sex"].replace({"male": "M", "female": "F"})
    df = df.drop(
        columns=["Unnamed: 7", "Unnamed: 6", "Annotations", "Values"], errors="ignore"
    )
    return df


def load_internal_migration(file_template, years, ages, places):
    """Load internal migration CSVs for multiple years/ages/places."""
    int_mig_dest = pd.DataFrame()
    int_mig_orig = pd.DataFrame()

    for year in years:
        for age in ages:
            for place in places:
                file_path = file_template.format(year=year, age=age, place=place)
                df_cleaned = load_and_clean_statcube_csv(file_path, geo_col_index=2)
                if place == "destination":
                    int_mig_dest = pd.concat(
                        [int_mig_dest, df_cleaned], ignore_index=True
                    )
                else:
                    int_mig_orig = pd.concat(
                        [int_mig_orig, df_cleaned], ignore_index=True
                    )

    return int_mig_dest, int_mig_orig


def load_external_migration(file_template, years, ages):
    """Load international migration CSVs for multiple years/ages."""
    ext_mig = pd.DataFrame()
    for year in years:
        for age in ages:
            file_path = file_template.format(year=year, age=age)
            df_cleaned = load_and_clean_statcube_csv(file_path, geo_col_index=1)
            ext_mig = pd.concat([ext_mig, df_cleaned], ignore_index=True)
    return ext_mig


def load_ukrainian_internal_migration(file_template, places):
    """Load Ukrainian internal migration CSVs."""
    int_mig_dest = pd.DataFrame()
    int_mig_orig = pd.DataFrame()

    for place in places:
        file_path = file_template.format(place=place)
        df_cleaned = load_and_clean_statcube_csv(file_path, geo_col_index=2)
        if place == "destination":
            int_mig_dest = pd.concat([int_mig_dest, df_cleaned], ignore_index=True)
        else:
            int_mig_orig = pd.concat([int_mig_orig, df_cleaned], ignore_index=True)

    return int_mig_dest, int_mig_orig


def load_ukrainian_external_migration(filepath):
    """Load Ukrainian international migration CSV."""
    df_cleaned = load_and_clean_statcube_csv(filepath, geo_col_index=2)
    return df_cleaned


# =============================================================================
# 2. AGE PARSING & FORMATTING
# =============================================================================

def parse_age_string_to_int(age_str):
    """Convert raw age string to integer age (e.g. '59 years old' -> 59)."""
    if pd.isna(age_str):
        return None
    if "under 1 year" in age_str:
        return 0
    match = re.search(r"(\d+)", age_str)
    return int(match.group(1)) if match else None


def get_eurostat_age_formats(age_int):
    """Convert integer age to Eurostat format (e.g. 1 -> ('Y1', 1))."""
    if age_int is None:
        return None, None
    age_num = int(age_int)
    if age_num == 0:
        return "Y_LT1", 0
    if age_num >= 100:
        return "Y_GE100", 100
    return f"Y{age_num}", age_num


def parse_5yr_age_group(age_str):
    """Parse 5-year age group string to numeric lower bound."""
    if pd.isna(age_str):
        return None
    s = str(age_str).lower().strip()
    if "up to 4" in s or "under" in s:
        return 0
    match = re.search(r"(\d+)", s)
    return int(match.group(1)) if match else None


def get_eurostat_age_label(age_int):
    """Generate Eurostat age label (e.g. 'Y0', 'Y_GE100')."""
    if age_int >= 100:
        return "Y_GE100"
    return f"Y{age_int}"


# =============================================================================
# 3. MIGRATION PROCESSING (SINGLE-YEAR AGE DATA)
# =============================================================================

def process_migration_component(df_raw, geo_col_name, value_col_name="Number"):
    """
    Clean and standardize a raw migration DataFrame with single-year ages.
    Creates Eurostat-compatible 'age' and 'age_num' columns.
    """
    df_proc = df_raw.copy()

    rename_map = {
        "Time section": "TIME_PERIOD",
        geo_col_name: "geo",
        "Age in single years": "age_raw_str",
        "Sex": "sex",
        "Nationality, pol. breakdown (level +3)": "nationality_raw",
        value_col_name: "OBS_VALUE",
    }

    cols_to_use = [col for col in rename_map.keys() if col in df_proc.columns]

    if "Values" in df_proc.columns:
        df_proc = df_proc[df_proc["Values"].str.contains("migration", na=False)]

    df_proc = df_proc[cols_to_use].rename(columns=rename_map)
    df_proc["TIME_PERIOD"] = df_proc["TIME_PERIOD"].astype(float).astype(int)
    df_proc["sex"] = df_proc["sex"].str[0].str.upper()
    df_proc["nationality"] = df_proc["nationality_raw"].map(
        {"Austria": "Austrian", "Foreign country": "Foreign"}
    )
    df_proc["OBS_VALUE"] = df_proc["OBS_VALUE"].fillna(0)

    df_proc["age_int"] = df_proc["age_raw_str"].apply(parse_age_string_to_int)
    age_formats = df_proc["age_int"].apply(get_eurostat_age_formats)
    df_proc["age"] = age_formats.str[0]
    df_proc["age_num"] = age_formats.str[1]

    grouping_cols = ["TIME_PERIOD", "geo", "sex", "nationality", "age", "age_num"]
    df_proc = df_proc.dropna(subset=grouping_cols)
    df_agg = df_proc.groupby(grouping_cols, as_index=False)["OBS_VALUE"].sum()
    df_agg["OBS_VALUE"] = df_agg["OBS_VALUE"].astype(int)

    return df_agg


def calculate_total_net_migration_single_year(ext_mig_raw, int_mig_dest_raw,
                                               int_mig_orig_raw):
    """
    Calculate total net migration from single-year age components.
    Total Net = Net International + (Internal In - Internal Out)
    """
    print("  Calculating total net migration (single-year ages)...")

    ext_clean = process_migration_component(ext_mig_raw, geo_col_name="NUTS 3-unit")
    dest_clean = process_migration_component(
        int_mig_dest_raw, geo_col_name="NUTS 3-unit of place of destination"
    )
    orig_clean = process_migration_component(
        int_mig_orig_raw, geo_col_name="NUTS 3-unit of place of origin"
    )

    merge_cols = ["TIME_PERIOD", "geo", "sex", "nationality", "age", "age_num"]

    net_internal = pd.merge(
        dest_clean.rename(columns={"OBS_VALUE": "in_flow"}),
        orig_clean.rename(columns={"OBS_VALUE": "out_flow"}),
        on=merge_cols,
        how="outer",
    )
    net_internal[["in_flow", "out_flow"]] = net_internal[
        ["in_flow", "out_flow"]
    ].fillna(0)
    net_internal["OBS_VALUE"] = net_internal["in_flow"] - net_internal["out_flow"]
    net_internal_clean = net_internal[merge_cols + ["OBS_VALUE"]]

    total_net = pd.merge(
        ext_clean.rename(columns={"OBS_VALUE": "external_net"}),
        net_internal_clean.rename(columns={"OBS_VALUE": "internal_net"}),
        on=merge_cols,
        how="outer",
    )
    total_net[["external_net", "internal_net"]] = total_net[
        ["external_net", "internal_net"]
    ].fillna(0)
    total_net["OBS_VALUE"] = (
        total_net["external_net"] + total_net["internal_net"]
    ).astype(int)

    df_final = total_net[merge_cols + ["OBS_VALUE"]]
    print(f"    -> {len(df_final)} records created.")
    return df_final


# =============================================================================
# 4. MIGRATION PROCESSING (5-YEAR AGE DATA, SPLINE EXPANSION)
# =============================================================================

def spline_expand_group(group_df, value_col="OBS_VALUE"):
    """Apply PCHIP interpolation to expand 5-year bins into 1-year bins."""
    group_df = group_df.sort_values("age_lower")
    ages_lower = group_df["age_lower"].values
    counts = group_df[value_col].values

    if len(counts) == 0:
        return pd.DataFrame()

    next_ages = np.append(ages_lower[1:], ages_lower[-1] + 5)
    x_knots = np.concatenate(([0], next_ages))
    y_cum = np.concatenate(([0], np.cumsum(counts)))

    try:
        interpolator = PchipInterpolator(x_knots, y_cum)
    except Exception:
        return pd.DataFrame()

    x_target = np.arange(0, 101)
    y_target_cum = interpolator(x_target)
    single_year_counts = np.diff(y_target_cum)

    result_df = pd.DataFrame(
        {"age_num": x_target[:-1], "OBS_VALUE": single_year_counts}
    )

    if x_knots[-1] >= 100:
        total_dist = result_df["OBS_VALUE"].sum()
        original_sum = counts.sum()
        diff = original_sum - total_dist
        row_100 = pd.DataFrame({"age_num": [100], "OBS_VALUE": [diff]})
        result_df = pd.concat([result_df, row_100], ignore_index=True)

    return result_df


def process_migration_component_splined(df_raw, geo_col_name,
                                         value_col_name="Number"):
    """Clean raw 5-year data and spline it into 1-year data."""
    df_proc = df_raw.copy()

    rename_map = {
        "Time section": "TIME_PERIOD",
        geo_col_name: "geo",
        "Age in 5 years groups": "age_raw_str",
        "Age in 5-years groups": "age_raw_str",
        "Sex": "sex",
        "Nationality (aggregation by political breakdown)": "nationality_raw",
        "Nationality (aggregated by political breakdown)": "nationality_raw",
        value_col_name: "OBS_VALUE",
    }

    cols_to_use = [col for col in rename_map.keys() if col in df_proc.columns]
    df_proc = df_proc[cols_to_use].rename(columns=rename_map)

    df_proc["TIME_PERIOD"] = (
        pd.to_numeric(df_proc["TIME_PERIOD"], errors="coerce").fillna(0).astype(int)
    )
    df_proc["sex"] = df_proc["sex"].str[0].str.upper()

    nat_map = {"Ukraine": "Ukrainian", "Austria": "Austrian", "Foreign country": "Foreign"}
    df_proc["nationality"] = df_proc["nationality_raw"].apply(
        lambda x: nat_map.get(x, x)
    )
    df_proc["OBS_VALUE"] = pd.to_numeric(df_proc["OBS_VALUE"], errors="coerce").fillna(0)
    df_proc["age_lower"] = df_proc["age_raw_str"].apply(parse_5yr_age_group)
    df_proc = df_proc.dropna(subset=["age_lower"])

    group_cols = ["TIME_PERIOD", "geo", "sex", "nationality"]

    expanded_df = (
        df_proc.groupby(group_cols)
        .apply(spline_expand_group, include_groups=False)
        .reset_index()
    )

    expanded_df["age_num"] = expanded_df["age_num"].astype(int)
    expanded_df["age"] = expanded_df["age_num"].apply(get_eurostat_age_label)
    expanded_df["OBS_VALUE"] = expanded_df["OBS_VALUE"].round(0).astype(int)

    return expanded_df


def calculate_total_net_migration_splined(ext_mig_raw, int_mig_dest_raw,
                                           int_mig_orig_raw):
    """
    Calculate total net migration from 5-year age components (spline-expanded).
    """
    print("  Calculating total net migration (splined 5-year -> 1-year)...")

    ext_clean = process_migration_component_splined(
        ext_mig_raw, geo_col_name="NUTS 3 unit"
    )
    dest_clean = process_migration_component_splined(
        int_mig_dest_raw, geo_col_name="NUTS 3-unit of place of destination"
    )
    orig_clean = process_migration_component_splined(
        int_mig_orig_raw, geo_col_name="NUTS 3-unit of place of origin"
    )

    merge_cols = ["TIME_PERIOD", "geo", "sex", "nationality", "age", "age_num"]

    net_internal = pd.merge(
        dest_clean.rename(columns={"OBS_VALUE": "in_flow"}),
        orig_clean.rename(columns={"OBS_VALUE": "out_flow"}),
        on=merge_cols,
        how="outer",
    ).fillna(0)
    net_internal["internal_net"] = net_internal["in_flow"] - net_internal["out_flow"]

    total_net = pd.merge(
        ext_clean.rename(columns={"OBS_VALUE": "external_net"}),
        net_internal,
        on=merge_cols,
        how="outer",
    ).fillna(0)
    total_net["OBS_VALUE"] = (
        total_net["external_net"] + total_net["internal_net"]
    ).astype(int)

    df_final = total_net[merge_cols + ["OBS_VALUE"]].sort_values(
        ["TIME_PERIOD", "geo", "sex", "age_num"]
    )
    print(f"    -> {len(df_final)} single-year records created.")
    return df_final


# =============================================================================
# 5. SENSITIVITY ESTIMATION
# =============================================================================

def estimate_sensitivity_theilsen(df_gross, start_year, end_year=2024):
    """Estimate marginal propensity to migrate (Beta) using Theil-Sen."""
    df = df_gross[
        (df_gross["geo"].str.len() == 5)
        & (df_gross["year"] >= start_year)
        & (df_gross["year"] <= end_year)
    ].copy()

    df_pivot = df.pivot_table(
        index=["year", "geo", "nationality"],
        columns="flow",
        values="value",
        aggfunc="sum",
    ).reset_index()

    if "in" not in df_pivot.columns:
        df_pivot["in"] = 0
    if "out" not in df_pivot.columns:
        df_pivot["out"] = 0
    df_pivot["net"] = df_pivot["in"] - df_pivot["out"]
    df_pivot["nationality"] = df_pivot["nationality"].replace(
        {"Austria": "Austrian", "Foreign country": "Foreign"}
    )

    df_wide = (
        df_pivot.pivot_table(index=["year", "geo"], columns="nationality", values="net")
        .reset_index()
        .fillna(0)
    )
    if "Austrian" not in df_wide.columns:
        df_wide["Austrian"] = 0
    if "Foreign" not in df_wide.columns:
        df_wide["Foreign"] = 0
    df_wide["Total"] = df_wide["Austrian"] + df_wide["Foreign"]

    sensitivity_map = {}
    for geo in df_wide["geo"].unique():
        df_geo = df_wide[df_wide["geo"] == geo]
        if len(df_geo) >= 3:
            try:
                slope = stats.theilslopes(
                    df_geo["Austrian"].values, df_geo["Total"].values, 0.95
                )[0]
            except Exception:
                slope = 0.0
        else:
            slope = 0.0
        sensitivity_map[geo] = slope

    return sensitivity_map


# =============================================================================
# 6. SHARED HELPER FUNCTIONS FOR PROJECTION
# =============================================================================

def add_age_num_column(df):
    """Add 'age_num' column derived from Eurostat 'age' strings."""
    if "age_num" in df.columns:
        return df

    def get_age_num(age_str):
        if age_str == "Y_LT1":
            return 0
        if age_str == "Y_GE100":
            return 100
        try:
            return int(age_str.replace("Y", ""))
        except Exception:
            return -1

    df_copy = df.copy()
    df_copy["age_num"] = df_copy["age"].astype(str).apply(get_age_num)
    return df_copy


def add_5_year_age_group(df):
    """Add 'age_group_5' column from 'age_num'."""
    if "age_num" not in df.columns:
        df = add_age_num_column(df)
    bins = [
        -1, 0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84,
        89, 94, 99, 101,
    ]
    labels = [
        "Y_LT1", "Y1-4", "Y5-9", "Y10-14", "Y15-19", "Y20-24", "Y25-29",
        "Y30-34", "Y35-39", "Y40-44", "Y45-49", "Y50-54", "Y55-59", "Y60-64",
        "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y85-89", "Y90-94", "Y95-99",
        "Y_GE100",
    ]
    df_copy = df.copy()
    df_copy["age_group_5"] = pd.cut(
        df_copy["age_num"], bins=bins, labels=labels, right=True
    ).astype(str)
    return df_copy


def smooth_by_age(df, value_col, window=5, groupby_cols=None):
    """Apply rolling mean smoothing over age within groups."""
    if groupby_cols is None:
        groupby_cols = []
    if "age_num" not in df.columns:
        df = add_age_num_column(df)
    df_copy = df.copy().sort_values(groupby_cols + ["age_num"])
    df_copy[value_col] = df_copy.groupby(groupby_cols)[value_col].transform(
        lambda x: x.rolling(window=window, center=True, min_periods=1).mean()
    )
    return df_copy


def add_time_weights(df, weights_dict):
    """Add temporal weights column based on TIME_PERIOD."""
    df_copy = df.copy()
    df_copy["weight"] = df_copy["TIME_PERIOD"].map(weights_dict).fillna(1)
    return df_copy


# =============================================================================
# 7. CORE PROJECTION LOGIC
# =============================================================================

def create_composite_shock_key(df_total_hist, df_ukr_hist, weights_dict):
    """Create a blended spatial key (Ukrainian vs. Regular Foreign)."""
    df_u = add_time_weights(add_5_year_age_group(df_ukr_hist.copy()), weights_dict)

    merge_cols = ["geo", "TIME_PERIOD", "sex", "age_group_5"]
    df_u_agg = (
        df_u.groupby(merge_cols + ["weight"])["OBS_VALUE"]
        .sum()
        .reset_index(name="Net_Ukr")
    )

    df_t = add_5_year_age_group(
        df_total_hist[df_total_hist["nationality"] == "Foreign"].copy()
    )
    df_t_agg = df_t.groupby(merge_cols)["OBS_VALUE"].sum().reset_index(name="Net_Total")

    df_merge = pd.merge(df_t_agg, df_u_agg, on=merge_cols, how="left")
    df_merge["Net_Ukr"] = df_merge["Net_Ukr"].fillna(0)
    df_merge["weight"] = df_merge["weight"].fillna(
        df_merge["TIME_PERIOD"].map(weights_dict).fillna(1)
    )
    df_merge["Net_Regular"] = df_merge["Net_Total"] - df_merge["Net_Ukr"]
    df_merge["Inflow_Ukr"] = df_merge["Net_Ukr"].clip(lower=0) * df_merge["weight"]
    df_merge["Inflow_Reg"] = df_merge["Net_Regular"].clip(lower=0) * df_merge["weight"]

    maps = (
        df_merge.groupby(["geo", "age_group_5", "sex"])[["Inflow_Ukr", "Inflow_Reg"]]
        .sum()
        .reset_index()
    )
    nat_totals = (
        maps.groupby(["age_group_5", "sex"])[["Inflow_Ukr", "Inflow_Reg"]]
        .sum()
        .reset_index()
        .rename(columns={"Inflow_Ukr": "Nat_Ukr", "Inflow_Reg": "Nat_Reg"})
    )

    maps = pd.merge(maps, nat_totals, on=["age_group_5", "sex"], how="left")
    maps["Lambda_Ukr"] = (maps["Inflow_Ukr"] / maps["Nat_Ukr"]).fillna(0)
    maps["Lambda_Reg"] = (maps["Inflow_Reg"] / maps["Nat_Reg"]).fillna(0)

    nat_totals["Total_Inflow"] = nat_totals["Nat_Ukr"] + nat_totals["Nat_Reg"]
    nat_totals["Alpha"] = (nat_totals["Nat_Ukr"] / nat_totals["Total_Inflow"]).fillna(0)

    final_key = pd.merge(
        maps, nat_totals[["age_group_5", "sex", "Alpha"]],
        on=["age_group_5", "sex"], how="left",
    )
    final_key["NUTS3_Share"] = (
        final_key["Alpha"] * final_key["Lambda_Ukr"]
        + (1 - final_key["Alpha"]) * final_key["Lambda_Reg"]
    )
    return final_key[["geo", "age_group_5", "sex", "NUTS3_Share"]]


def calculate_historical_patterns(df_hist, weights_dict):
    """Calculate historical anchor for Austrian migration."""
    df = add_time_weights(add_5_year_age_group(df_hist), weights_dict)
    df["Weighted_Value"] = df["OBS_VALUE"] * df["weight"]
    df_grouped = df.groupby(["geo", "age_group_5", "sex", "nationality"])
    df_pattern = df_grouped.apply(
        lambda x: x["Weighted_Value"].sum() / x["weight"].sum()
    ).reset_index(name="Hist_Avg")
    return df_pattern


def process_single_projection_direct(
    scenario_label, df_target_nat, df_base_nuts3, df_hist, df_ukr,
    sensitivity_map, weights_dict, anchor_multiplier=1.0,
):
    """
    Direct methodology with 3 critical fixes:
    1. Activity_Share removed (was causing double-weighting).
    2. Normalization at age_group_5 level.
    3. Metadata safety after outer merge.
    """
    # --- STEP 1: NATIONAL SHOCK ---
    target_clean = add_age_num_column(df_target_nat.copy())
    base_clean = add_age_num_column(df_base_nuts3.copy())

    df_base_nat_agg = (
        base_clean.groupby(["TIME_PERIOD", "age_num", "sex"])["OBS_VALUE"]
        .sum()
        .reset_index()
        .rename(columns={"OBS_VALUE": "OBS_VALUE_BASE_NAT"})
    )

    df_deltas = pd.merge(
        target_clean, df_base_nat_agg,
        on=["TIME_PERIOD", "age_num", "sex"], how="outer",
    )
    df_deltas["OBS_VALUE"] = df_deltas["OBS_VALUE"].fillna(0)
    df_deltas["OBS_VALUE_BASE_NAT"] = df_deltas["OBS_VALUE_BASE_NAT"].fillna(0)
    df_deltas["National_Delta"] = df_deltas["OBS_VALUE"] - df_deltas["OBS_VALUE_BASE_NAT"]
    df_deltas = add_5_year_age_group(df_deltas)

    # Smooth with sum-preserving correction
    delta_cols = ["TIME_PERIOD", "sex"]
    orig_sum = df_deltas.groupby(delta_cols)["National_Delta"].transform("sum")
    df_deltas = smooth_by_age(df_deltas, "National_Delta", 5, delta_cols)
    smooth_sum = df_deltas.groupby(delta_cols)["National_Delta"].transform("sum")
    df_deltas["National_Delta"] *= (orig_sum / smooth_sum).fillna(1.0)
    df_deltas = df_deltas[["TIME_PERIOD", "age_num", "sex", "age_group_5", "National_Delta"]]

    # --- STEP 2: PREPARE KEYS ---
    # A. Refugee key
    df_key_refugee = create_composite_shock_key(df_hist, df_ukr, weights_dict)

    full_grid = pd.MultiIndex.from_product(
        [df_hist["geo"].unique(), df_deltas["age_group_5"].unique(), ["M", "F"]],
        names=["geo", "age_group_5", "sex"],
    ).to_frame(index=False)

    df_key_refugee = pd.merge(
        full_grid, df_key_refugee, on=["geo", "age_group_5", "sex"], how="left"
    )

    df_key_data = df_hist[df_hist["nationality"] == "Foreign"][["geo", "OBS_VALUE"]].copy()
    df_key_data["Inflow"] = df_key_data["OBS_VALUE"].clip(lower=0)
    fb_geo = df_key_data.groupby("geo")["Inflow"].sum()
    fb_share = (fb_geo / fb_geo.sum()).reset_index(name="Fallback_Share")
    df_key_refugee = pd.merge(df_key_refugee, fb_share, on="geo", how="left")
    df_key_refugee["NUTS3_Share_Ref"] = (
        df_key_refugee["NUTS3_Share"]
        .fillna(df_key_refugee["Fallback_Share"])
        .fillna(0)
    )
    df_key_refugee = df_key_refugee.drop(columns=["NUTS3_Share", "Fallback_Share"])

    # B. Structural key
    df_base_struct = add_5_year_age_group(base_clean.copy())
    df_struct_agg = (
        df_base_struct.groupby(["TIME_PERIOD", "geo", "age_group_5", "sex"])["OBS_VALUE"]
        .sum()
        .reset_index()
    )
    df_struct_agg["Migration_Intensity"] = df_struct_agg["OBS_VALUE"].abs()
    df_struct_agg = df_struct_agg.sort_values(["geo", "age_group_5", "sex", "TIME_PERIOD"])
    df_struct_agg["Migration_Intensity"] = df_struct_agg.groupby(
        ["geo", "age_group_5", "sex"]
    )["Migration_Intensity"].transform(
        lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
    )

    struct_group_cols = ["TIME_PERIOD", "age_group_5", "sex"]
    df_struct_agg["National_Intensity"] = df_struct_agg.groupby(struct_group_cols)[
        "Migration_Intensity"
    ].transform("sum")

    mask_small_nat = df_struct_agg["National_Intensity"] < 10
    n_regions = df_struct_agg["geo"].nunique()
    df_struct_agg["NUTS3_Share_Struct"] = (
        df_struct_agg["Migration_Intensity"] / df_struct_agg["National_Intensity"]
    )
    if mask_small_nat.any():
        df_struct_agg.loc[mask_small_nat, "NUTS3_Share_Struct"] = 1.0 / n_regions
    df_struct_agg["NUTS3_Share_Struct"] = df_struct_agg["NUTS3_Share_Struct"].fillna(
        1.0 / n_regions
    )

    df_key_struct = df_struct_agg[
        ["TIME_PERIOD", "geo", "age_group_5", "sex", "NUTS3_Share_Struct"]
    ].drop_duplicates(subset=["TIME_PERIOD", "geo", "age_group_5", "sex"])

    # --- STEP 3: DISAGGREGATE SHOCK ---
    df_geo_list = df_hist[["geo"]].drop_duplicates()
    df_corr_cross = pd.merge(df_deltas, df_geo_list, how="cross")
    df_corr = pd.merge(
        df_corr_cross, df_key_refugee, on=["geo", "age_group_5", "sex"], how="left"
    )
    df_corr = pd.merge(
        df_corr, df_key_struct,
        on=["TIME_PERIOD", "geo", "age_group_5", "sex"], how="left",
    )
    df_corr["NUTS3_Share_Struct"] = df_corr["NUTS3_Share_Struct"].fillna(0)

    # Normalize refugee share
    share_sum_ref = df_corr.groupby(["TIME_PERIOD", "age_num", "sex"])[
        "NUTS3_Share_Ref"
    ].transform("sum")
    mask_ref = share_sum_ref != 0
    df_corr.loc[mask_ref, "NUTS3_Share_Ref"] = (
        df_corr.loc[mask_ref, "NUTS3_Share_Ref"] / share_sum_ref.loc[mask_ref]
    )

    # Omega decay (refugee -> structural transition)
    unique_years = df_corr["TIME_PERIOD"].unique()
    omega_map = {}
    for y in unique_years:
        if y <= 2026:
            omega_map[y] = 1.0
        elif y >= 2035:
            omega_map[y] = 0.0
        else:
            omega_map[y] = (2035 - y) / 9.0
    df_corr["Omega_Weight"] = df_corr["TIME_PERIOD"].map(omega_map)

    # Blend keys
    df_corr["NUTS3_Share_Final"] = (
        df_corr["NUTS3_Share_Ref"] * df_corr["Omega_Weight"]
        + df_corr["NUTS3_Share_Struct"] * (1 - df_corr["Omega_Weight"])
    )

    # Normalize final key
    share_sum = df_corr.groupby(["TIME_PERIOD", "age_num", "sex"])[
        "NUTS3_Share_Final"
    ].transform("sum")
    mask = share_sum != 0
    df_corr.loc[mask, "NUTS3_Share_Final"] = (
        df_corr.loc[mask, "NUTS3_Share_Final"] / share_sum.loc[mask]
    )
    df_corr["NUTS3_Delta_Correction"] = (
        df_corr["National_Delta"] * df_corr["NUTS3_Share_Final"].fillna(0)
    )

    # --- STEP 4: HISTORICAL ANCHOR ---
    hist_patterns = calculate_historical_patterns(df_hist, weights_dict)
    austrian_pattern = (
        hist_patterns[hist_patterns["nationality"] == "Austrian"][
            ["geo", "age_group_5", "sex", "Hist_Avg"]
        ]
        .rename(columns={"Hist_Avg": "Austrian_Base_Start"})
    )

    # --- STEP 5: DECOMPOSITION ---
    df_base = add_5_year_age_group(
        add_age_num_column(df_base_nuts3.rename(columns={"OBS_VALUE": "OBS_VALUE_BASE"}))
    )

    base_cols = ["TIME_PERIOD", "geo", "sex"]
    orig_base_sum = df_base.groupby(base_cols)["OBS_VALUE_BASE"].transform("sum")
    df_base = smooth_by_age(df_base, "OBS_VALUE_BASE", 5, base_cols)
    smooth_base_sum = df_base.groupby(base_cols)["OBS_VALUE_BASE"].transform("sum")
    df_base["OBS_VALUE_BASE"] *= (orig_base_sum / smooth_base_sum).fillna(1.0)

    df_base_2024 = df_base[df_base["TIME_PERIOD"] == 2024][
        ["geo", "age_num", "sex", "OBS_VALUE_BASE"]
    ].rename(columns={"OBS_VALUE_BASE": "Eurostat_Total_2024"})

    df_split = pd.merge(df_base, df_base_2024, on=["geo", "age_num", "sex"], how="left")
    df_split["Eurostat_Total_2024"] = df_split["Eurostat_Total_2024"].fillna(0)
    df_split = pd.merge(
        df_split, austrian_pattern, on=["geo", "age_group_5", "sex"], how="left"
    )
    df_split["beta_sensitivity"] = df_split["geo"].map(sensitivity_map).fillna(0.0)

    # Anchor
    df_split["Austrian_Base_Start"] = df_split["Austrian_Base_Start"].fillna(0)
    df_split = smooth_by_age(
        df_split, "Austrian_Base_Start", 7, ["TIME_PERIOD", "geo", "sex"]
    )
    df_split["Austrian_Base_Start"] *= anchor_multiplier
    df_split["Austrian_Anchor_Final"] = df_split["Austrian_Base_Start"]

    # Direct Beta formula
    df_split["Baseline_Change"] = (
        df_split["OBS_VALUE_BASE"] - df_split["Eurostat_Total_2024"]
    )
    df_split["Baseline_A"] = df_split["Austrian_Anchor_Final"] + (
        df_split["Baseline_Change"] * df_split["beta_sensitivity"]
    )
    df_split["Baseline_F"] = df_split["OBS_VALUE_BASE"] - df_split["Baseline_A"]

    # --- STEP 6: FINALIZE ---
    df_final = pd.merge(
        df_corr, df_split,
        on=["TIME_PERIOD", "geo", "age_num", "sex", "age_group_5"], how="outer",
    )

    df_final["projection"] = scenario_label
    df_final["Omega_Weight"] = df_final["Omega_Weight"].fillna(
        df_final["TIME_PERIOD"].map(omega_map)
    )
    df_final["Baseline_A"] = df_final["Baseline_A"].fillna(0)
    df_final["Baseline_F"] = df_final["Baseline_F"].fillna(0)
    df_final["NUTS3_Delta_Correction"] = df_final["NUTS3_Delta_Correction"].fillna(0)

    df_final["net_A_by_age"] = df_final["Baseline_A"]
    df_final["net_nonA_by_age"] = (
        df_final["Baseline_F"] + df_final["NUTS3_Delta_Correction"]
    )
    df_final["net_all"] = df_final["net_A_by_age"] + df_final["net_nonA_by_age"]

    cols_to_keep = [
        "projection", "TIME_PERIOD", "geo", "sex", "age_num",
        "net_all", "net_nonA_by_age", "net_A_by_age",
        "NUTS3_Delta_Correction", "Omega_Weight",
    ]
    if "age" in df_final.columns:
        cols_to_keep.insert(5, "age")

    return df_final[cols_to_keep]


# =============================================================================
# 8. SCENARIO EXECUTION
# =============================================================================

def run_projection_scenarios(df_real_hist, df_real_hist_ukr, df_mig_EP2023,
                              df_mig_EP2019, df_gross):
    """Run the full multi-scenario projection loop."""
    print("\n[1/6] Calculating sensitivity maps...")
    maps = {
        "SensPreCrisis": estimate_sensitivity_theilsen(df_gross, start_year=2010, end_year=2019),
        "SensCrisisEra": estimate_sensitivity_theilsen(df_gross, start_year=2015, end_year=2024),
        "SensPostCovid": estimate_sensitivity_theilsen(df_gross, start_year=2018, end_year=2024),
    }

    # Prepare input scenarios
    input_scenarios = []
    available_projections = df_mig_EP2023["projection"].unique()
    for proj in available_projections:
        df_target = df_mig_EP2023[df_mig_EP2023["projection"] == proj].copy()
        input_scenarios.append({"label": proj, "df_target": df_target})

    print(f"  Processing {len(input_scenarios)} EUROPOP2023 scenarios: "
          f"{list(available_projections)}")

    # Weight scenarios
    weight_scenarios = {
        "equal": {2022: 1 / 3, 2023: 1 / 3, 2024: 1 / 3},
        "recent": {2022: 0.2, 2023: 0.3, 2024: 0.5},
        "early": {2022: 0.5, 2023: 0.3, 2024: 0.2},
    }
    anchors = {"anchorLow": 0.8, "anchorBase": 1.0}

    df_base_nuts3 = df_mig_EP2019[df_mig_EP2019["projection"] == "BSL2019"].copy()
    total_scenarios = len(input_scenarios) * len(weight_scenarios) * len(anchors) * len(maps)
    counter = 0
    all_results = []

    print(f"\n[2/6] Running {total_scenarios} projection scenarios...")

    for inp in input_scenarios:
        base_label = inp["label"]
        df_target = inp["df_target"]

        if base_label == "BSL2019":
            current_weights = {"equal": weight_scenarios["equal"]}
            current_anchors = {"anchorBase": 1.0}
            current_maps = {"SensCrisisEra": maps["SensCrisisEra"]}
        else:
            current_weights = weight_scenarios
            current_anchors = anchors
            current_maps = maps

        for weight_name, w_dict in current_weights.items():
            structural_label = f"{base_label}{weight_name}"
            for sens_label, sens_map in current_maps.items():
                for anchor_name, anchor_val in current_anchors.items():
                    counter += 1
                    final_label = f"{structural_label}__{sens_label}__{anchor_name}"

                    if counter % 5 == 0 or counter == 1:
                        print(f"   [{counter}/{total_scenarios}] {final_label}")

                    df_res = process_single_projection_direct(
                        scenario_label=final_label,
                        df_target_nat=df_target,
                        df_base_nuts3=df_base_nuts3,
                        df_hist=df_real_hist,
                        df_ukr=df_real_hist_ukr,
                        sensitivity_map=sens_map,
                        weights_dict=w_dict,
                        anchor_multiplier=anchor_val,
                    )
                    all_results.append(df_res)

    print(f"\n[3/6] Concatenating results...")
    df_final_output = pd.concat(all_results, ignore_index=True)
    df_final_output = df_final_output[df_final_output["TIME_PERIOD"] >= 2022].copy()

    print(f"\n[4/6] --- SUCCESS ---")
    print(f"   Processed {counter} scenarios")
    print(f"   Output rows: {len(df_final_output):,}")

    return df_final_output, maps


# =============================================================================
# 9. OUTPUT SAVING
# =============================================================================

def save_parquet(df_final_projections, maps):
    """Save full projections as Parquet."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sensitivities_str = "_".join(map(str, maps))
    output_path = (
        f"PQT/dynres_net_migration_{sensitivities_str}_scenarios_{timestamp}.parquet"
    )

    df_final_projections = df_final_projections.round(4)
    df_final_projections.to_parquet(output_path, index=False, compression="snappy")
    print(f"  Parquet saved: {output_path}")
    return output_path


def save_collapsed_csv(df_final_projections):
    """Collapse scenarios to averages and save as CSV."""
    print("\n[5/6] Extracting and saving scenario totals...")
    start_time = time.time()

    df = df_final_projections.copy()
    print(f"  Total rows: {len(df):,}")

    # Fast string splitting via unique-value mapping
    unique_scenarios = df["projection"].unique()
    scenario_map = {s: s.split("__")[0] for s in unique_scenarios}
    df["projection_main"] = df["projection"].map(scenario_map)

    # Optimize types
    keep_cols = ["projection_main", "sex", "age", "geo", "TIME_PERIOD", "net_all"]
    subset = df[keep_cols].copy()
    for col in ["projection_main", "geo", "sex"]:
        subset[col] = subset[col].astype("category")
    subset["age"] = subset["age"].astype(str).astype("category")
    subset["net_all"] = subset["net_all"].fillna(0)

    # Collapse by averaging across sensitivity/anchor variants
    group_cols = ["projection_main", "geo", "sex", "age", "TIME_PERIOD"]
    df_collapsed = (
        subset.groupby(group_cols, observed=True)["net_all"].mean().reset_index()
    )
    df_collapsed["OBS_VALUE"] = df_collapsed["net_all"].round().astype(int)
    df_collapsed = df_collapsed.rename(columns={"projection_main": "projection"}).drop(
        columns=["net_all"]
    )

    output_filename = (
        r"XLSX/netmigration_EP2019-2023-AT_nuts3_age_sex_2022-2050_dynamicresidual.csv"
    )
    df_collapsed.to_csv(output_filename, index=False)

    elapsed = time.time() - start_time
    print(f"\n[6/6] --- SUCCESS in {elapsed:.1f} seconds ---")
    print(f"  CSV saved: {output_filename}")
    print(f"  Unique projections: {df_collapsed['projection'].unique().tolist()}")


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MIGRATION NET PROJECTION PIPELINE")
    print("=" * 70)

    # --- Step 1: Load internal & external migration (single-year ages) ---
    print("\n[Step 1] Loading single-year age migration data (2022-2024)...")
    int_template = "Data/Statcube/internal_migration-{year}_sex_nuts3_nat_{age}_mig_place_{place}.csv"
    ext_template = "Data/Statcube/international_migration-{year}_sex_nuts3_nat_{age}_net.csv"
    years = [2022, 2023, 2024]
    ages = ["0-59", "60-100"]

    int_mig_dest, int_mig_orig = load_internal_migration(
        int_template, years, ages, ["origin", "destination"]
    )
    ext_mig = load_external_migration(ext_template, years, ages)

    # --- Step 2: Calculate net migration (single-year) ---
    print("\n[Step 2] Processing single-year net migration...")
    df_real_hist = calculate_total_net_migration_single_year(
        ext_mig, int_mig_dest, int_mig_orig
    )

    # --- Step 3: Load Ukrainian migration (5-year age groups) ---
    print("\n[Step 3] Loading Ukrainian migration data...")
    ukr_int_template = "Data/Statcube/internal_migration-2022-2024_sex_nuts3_UKR_5age_mig_place_{place}.csv"
    ukr_ext_file = "Data/Statcube/international_migration-2022-2024_sex_nuts3_UKR_5age_net.csv"

    int_mig_dest_ukr, int_mig_orig_ukr = load_ukrainian_internal_migration(
        ukr_int_template, ["origin", "destination"]
    )
    ext_mig_ukr = load_ukrainian_external_migration(ukr_ext_file)

    # --- Step 4: Calculate Ukrainian net migration (splined) ---
    print("\n[Step 4] Processing Ukrainian net migration (spline expansion)...")
    df_real_hist_ukr = calculate_total_net_migration_splined(
        ext_mig_ukr, int_mig_dest_ukr, int_mig_orig_ukr
    )

    # --- Step 5: Load Eurostat projections ---
    print("\n[Step 5] Loading Eurostat projections...")
    df_mig_EP2023 = pd.read_csv(
        r"Data/Eurostat/migration_EP2023-AT_age_sex_2022-2050_scenarios.csv"
    )
    df_mig_EP2023["projection"] = df_mig_EP2023["projection"].astype(str) + "2023"

    df_mig_EP2019 = pd.read_csv(
        r"Data/Eurostat/migration_EP2019-AT_age_sex_nuts3_2019-2050.csv"
    )
    df_mig_EP2019["projection"] = df_mig_EP2019["projection"].astype(str) + "2019"

    # Smooth EP2019 baseline artifacts (5-year rolling mean)
    df_mig_EP2019 = df_mig_EP2019.sort_values(["geo", "sex", "age", "TIME_PERIOD"])
    df_mig_EP2019["OBS_VALUE"] = df_mig_EP2019.groupby(["geo", "sex", "age"])[
        "OBS_VALUE"
    ].transform(lambda x: x.rolling(window=5, center=True, min_periods=1).mean())
    print("  Applied 5-year rolling smooth to EP2019 baseline.")

    # --- Step 6: Load historical gross flows ---
    print("\n[Step 6] Loading historical gross flows (2002-2024)...")
    int_mig_dest_hist = load_and_clean_historical_csv(
        r"Data/Statcube/internal_migration-2002-2024_sex_nuts3_AT_mig_place_destination.csv"
    )
    int_mig_orig_hist = load_and_clean_historical_csv(
        r"Data/Statcube/internal_migration-2002-2024_sex_nuts3_AT_mig_place_origin.csv"
    )
    ext_mig_in = load_and_clean_historical_csv(
        r"Data/Statcube/international_migration-2002-2024_sex_nuts3_AT_mig_in.csv"
    )
    ext_mig_out = load_and_clean_historical_csv(
        r"Data/Statcube/international_migration-2002-2024_sex_nuts3_AT_mig_out.csv"
    )

    df_gross = pd.concat(
        [
            int_mig_dest_hist.assign(flow="in").rename(columns={
                "Time section": "year",
                "NUTS 3-unit of place of destination": "geo",
                "Nationality, pol. breakdown (level +3)": "nationality",
                "Sex": "sex",
                "Number": "value",
            }),
            int_mig_orig_hist.assign(flow="out").rename(columns={
                "Time section": "year",
                "NUTS 3-unit of place of origin": "geo",
                "Nationality, pol. breakdown (level +3)": "nationality",
                "Sex": "sex",
                "Number": "value",
            }),
            ext_mig_in.assign(flow="in").rename(columns={
                "Time section": "year",
                "NUTS 3-unit": "geo",
                "Nationality, pol. breakdown (level +3)": "nationality",
                "Sex": "sex",
                "Number": "value",
            }),
            ext_mig_out.assign(flow="out").rename(columns={
                "Time section": "year",
                "NUTS 3-unit": "geo",
                "Nationality, pol. breakdown (level +3)": "nationality",
                "Sex": "sex",
                "Number": "value",
            }),
        ],
        ignore_index=True,
    )
    df_gross = df_gross[df_gross["geo"] != "0"]

    # --- Step 7: Run projections ---
    print("\n[Step 7] Running multi-scenario projections...")
    df_final_projections, maps = run_projection_scenarios(
        df_real_hist, df_real_hist_ukr, df_mig_EP2023, df_mig_EP2019, df_gross
    )

    # --- Step 8: Save outputs ---
    print("\n[Step 8] Saving outputs...")
    save_parquet(df_final_projections, maps)
    save_collapsed_csv(df_final_projections)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()