#!/usr/bin/env python
"""
Approximate Austrian/EU net migration simulation.

Decomposes EUROPOP2019 total population projections into native (Austrian)
and foreign-born components by subtracting simulated migrant populations
and redistributing any resulting negative residuals via Gaussian weighting.

Outputs
-------
1. ``projected`` CSV  – cell-level population by (year, geo, sex, age, scenario, Nationality)
2. ``projected_all`` CSV – same as (1) but with IncomeGroup detail for migrants

Usage
-----
    python approx_at_eu_netmig.py [--input PATH] [--eurostat PATH] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "PQT/negtest_projected_mig_EU_2025_2051_h61_20251204_adjfert.parquet"
DEFAULT_EUROSTAT = "PQT/projected_AT_EU_2025_2051_20251204_adjfert.parquet"
DEFAULT_OUTDIR = "XLSX"
GAUSSIAN_SIGMA = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def convert_age_str(age_str: str) -> int:
    """Convert Eurostat age codes (``Y0``, ``Y_LT1``, ``Y_GE100``, …) to int."""
    if age_str == "Y_LT1":
        return 0
    if age_str == "Y_GE100":
        return 100
    return int(age_str[1:])


def extract_sim_key(scenario: str) -> str:
    """Return the migration-variant token from a simulation scenario string.

    Example: ``"BSL-BSL__r_full__..."`` → ``"BSL"``
    """
    head = scenario.split("__", 1)[0]
    parts = head.split("-")
    return parts[1] if len(parts) >= 2 else ""


def extract_euro_key(scenario: str) -> str:
    """Return the migration-variant token from a Eurostat scenario string.

    Example: ``"BSL-HMIGR-BSL-BSL"`` → ``"HMIGR"``
    """
    parts = scenario.split("-")
    return parts[1] if len(parts) >= 2 else ""


def redistribute_deficits(df: pd.DataFrame, sigma: float = GAUSSIAN_SIGMA) -> pd.DataFrame:
    """Redistribute negative native-population cells to nearby ages.

    Uses Gaussian weights centred on the deficit cell so that the
    subtracted mass is drawn predominantly from neighbouring ages.

    Parameters
    ----------
    df : DataFrame
        Must contain ``age_int`` and ``Pop_AT_raw`` columns.
    sigma : float
        Standard deviation (in years) of the Gaussian kernel.

    Returns
    -------
    DataFrame
        Copy of *df* with an added ``Pop_AT`` column and redistribution
        metadata stored in ``df.attrs``.
    """
    ages = df["age_int"].to_numpy()
    pop = df["Pop_AT_raw"].astype(float).to_numpy().copy()

    total_redistributed = 0.0
    redistribution_events = 0

    for idx in np.where(pop < 0)[0]:
        deficit = -pop[idx]
        pop[idx] = 0.0
        total_redistributed += deficit
        redistribution_events += 1

        donors = np.where(pop > 0)[0]
        if donors.size == 0:
            continue

        weights = np.exp(-((ages[donors] - ages[idx]) ** 2) / (2 * sigma ** 2))
        proportional = pop[donors] * weights
        total = proportional.sum()
        if total <= 0:
            continue

        pop[donors] -= (proportional / total) * deficit

    out = df.copy()
    out["Pop_AT"] = pop
    out.attrs["redistribution_total"] = total_redistributed
    out.attrs["redistribution_events"] = redistribution_events
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_projected_migration(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load projected migration parquet, return (with income groups, without).

    Returns
    -------
    projected_mig_ig : DataFrame
        Raw data with ``IncomeGroup`` retained.
    projected_mig : DataFrame
        Aggregated (summed) over ``IncomeGroup``.
    """
    logger.info("Loading projected migration from %s", path)
    projected_mig_ig = (
        pd.read_parquet(path)
        .rename(columns={"year": "TIME_PERIOD", "Pop": "OBS_VALUE"})
    )

    projected_mig = (
        projected_mig_ig
        .groupby(["TIME_PERIOD", "geo", "sex", "age", "scenario"], as_index=False)["OBS_VALUE"]
        .sum()
    )
    return projected_mig_ig, projected_mig


def load_eurostat_projections(path: str | Path) -> pd.DataFrame:
    """Load the Eurostat-approximating baseline projections.

    Appends ``-BSL`` to each scenario name (naturalisation baseline).
    """
    logger.info("Loading Eurostat projections from %s", path)
    df = (
        pd.read_parquet(path)
        .rename(columns={"year": "TIME_PERIOD", "Pop": "OBS_VALUE"})
    )
    df["scenario"] = df["scenario"] + "-BSL"
    return df


def build_eurostat_lookup(eurostat_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build a {migration_key → DataFrame} lookup from Eurostat projections."""
    lookup: dict[str, pd.DataFrame] = {}
    for scenario in eurostat_df["scenario"].unique():
        key = extract_euro_key(scenario)
        subset = (
            eurostat_df
            .loc[eurostat_df["scenario"] == scenario, ["TIME_PERIOD", "geo", "sex", "age", "OBS_VALUE"]]
            .rename(columns={"OBS_VALUE": "OBS_VALUE_obs"})
            .reset_index(drop=True)
        )
        lookup[key] = subset
    return lookup


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def process_scenario(
    sim_scen: str,
    dataset: ds.Dataset,
    euro_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Process a single simulation scenario.

    Returns (summary, clip_log, projected_chunk) or (None, None, None) if
    the scenario cannot be processed.
    """
    # Stream simulation rows for this scenario
    scanner = dataset.scanner(
        filter=ds.field("scenario") == sim_scen,
        columns=["year", "geo", "sex", "age", "Pop", "scenario"],
    )
    chunks = [batch.to_pandas() for batch in scanner.to_batches()]
    if not chunks:
        return None, None, None

    sim_df = (
        pd.concat(chunks, ignore_index=True)
        .rename(columns={"year": "TIME_PERIOD", "Pop": "OBS_VALUE_sim"})
        .assign(scenario=sim_scen)
        .groupby(["TIME_PERIOD", "geo", "sex", "age", "scenario"], as_index=False)
        .OBS_VALUE_sim.sum()
    )

    # Merge with Eurostat slice
    merged = sim_df.merge(euro_df, on=["TIME_PERIOD", "geo", "sex", "age"], how="inner")
    if merged.empty:
        return None, None, None

    merged["Pop_AT_raw"] = merged["OBS_VALUE_obs"] - merged["OBS_VALUE_sim"]
    merged["age_int"] = merged["age"].apply(convert_age_str)

    # --- Clipping log (before redistribution) ---
    original_clip = (
        merged
        .assign(cut_amount=lambda d: (-d["Pop_AT_raw"]).clip(lower=0))
        .groupby("TIME_PERIOD", as_index=False)["cut_amount"]
        .sum()
        .rename(columns={"cut_amount": "clipped_to_zero_original"})
        .assign(scenario=sim_scen)
    )

    # --- Redistribute deficits ---
    redistribution_log: list[dict] = []
    processed_groups: list[pd.DataFrame] = []

    for (time_period, geo, sex), group_df in merged.groupby(["TIME_PERIOD", "geo", "sex"]):
        redistributed = redistribute_deficits(group_df, sigma=GAUSSIAN_SIGMA)
        processed_groups.append(redistributed)
        redistribution_log.append({
            "TIME_PERIOD": time_period,
            "geo": geo,
            "sex": sex,
            "scenario": sim_scen,
            "redistributed_amount": redistributed.attrs.get("redistribution_total", 0),
            "redistribution_events": redistributed.attrs.get("redistribution_events", 0),
        })

    merged = pd.concat(processed_groups, ignore_index=True)

    # --- Clipping log (after redistribution) ---
    final_clip = (
        merged
        .assign(cut_amount=lambda d: (-d["Pop_AT"]).clip(lower=0))
        .groupby("TIME_PERIOD", as_index=False)["cut_amount"]
        .sum()
        .rename(columns={"cut_amount": "clipped_to_zero_final"})
        .assign(scenario=sim_scen)
    )

    combined_clip = original_clip.merge(final_clip, on=["TIME_PERIOD", "scenario"], how="outer")

    if redistribution_log:
        redist_summary = (
            pd.DataFrame(redistribution_log)
            .groupby(["TIME_PERIOD", "scenario"], as_index=False)
            .agg(
                total_redistributed=("redistributed_amount", "sum"),
                total_events=("redistribution_events", "sum"),
            )
        )
        combined_clip = combined_clip.merge(redist_summary, on=["TIME_PERIOD", "scenario"], how="left")

    # Final clip for any tiny remaining negatives
    merged["Pop_AT"] = merged["Pop_AT"].clip(lower=0)

    # --- Summary (fan-chart input) ---
    summary = merged.groupby(["TIME_PERIOD", "scenario"], as_index=False).agg(
        migrant_population=("OBS_VALUE_sim", "sum"),
        native_population=("Pop_AT", "sum"),
        total_population=("OBS_VALUE_obs", "sum"),
    )

    # --- Cell-level projected population ---
    projected_foreign = (
        sim_df
        .rename(columns={"OBS_VALUE_sim": "OBS_VALUE"})
        .assign(Nationality="Foreign country")
    )
    projected_native = (
        merged[["TIME_PERIOD", "geo", "sex", "age", "scenario", "Pop_AT"]]
        .rename(columns={"Pop_AT": "OBS_VALUE"})
        .assign(Nationality="Austria")
    )
    proj_chunk = pd.concat([projected_native, projected_foreign], ignore_index=True)

    dupes = proj_chunk.duplicated(
        subset=["TIME_PERIOD", "geo", "sex", "age", "scenario", "Nationality"]
    ).sum()
    if dupes:
        logger.warning("Found %d duplicate rows in scenario %s", dupes, sim_scen)

    return summary, combined_clip, proj_chunk


def run_pipeline(
    input_path: str | Path,
    eurostat_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full merge-and-decompose pipeline.

    Returns
    -------
    projected : DataFrame
        Cell-level native + foreign population (no income-group detail).
    projected_all : DataFrame
        Same as *projected* but migrants carry their ``IncomeGroup``.
    fan_df : DataFrame
        Yearly summary (migrant / native / total) per scenario.
    clip_log_df : DataFrame
        Clipping and redistribution diagnostics per scenario and year.
    """
    projected_mig_ig, _ = load_projected_migration(input_path)
    eurostat_df = load_eurostat_projections(eurostat_path)
    euro_lookup = build_eurostat_lookup(eurostat_df)

    dataset = ds.dataset(input_path, format="parquet")

    # Discover all simulation scenarios
    sim_scenarios: set[str] = set()
    for batch in dataset.scanner(columns=["scenario"]).to_batches():
        sim_scenarios.update(batch.column(0).to_pylist())
    sim_scenarios_sorted = sorted(sim_scenarios)
    logger.info("Found %d simulation scenarios", len(sim_scenarios_sorted))

    summary_list: list[pd.DataFrame] = []
    clip_log_list: list[pd.DataFrame] = []
    projected_list: list[pd.DataFrame] = []

    for sim_scen in sim_scenarios_sorted:
        sim_key = extract_sim_key(sim_scen)
        euro_df = euro_lookup.get(sim_key)
        if euro_df is None:
            logger.debug("No Eurostat match for scenario %s (key=%s), skipping", sim_scen, sim_key)
            continue

        summary, clip_log, proj_chunk = process_scenario(sim_scen, dataset, euro_df)
        if summary is not None:
            summary_list.append(summary)
        if clip_log is not None:
            clip_log_list.append(clip_log)
        if proj_chunk is not None:
            projected_list.append(proj_chunk)

    # Concatenate results
    fan_df = _safe_concat(summary_list, ["TIME_PERIOD", "scenario", "migrant_population", "native_population", "total_population"])
    clip_log_df = _safe_concat(clip_log_list, ["TIME_PERIOD", "scenario", "clipped_to_zero_original", "clipped_to_zero_final", "total_redistributed", "total_events"])
    projected = _safe_concat(projected_list, ["TIME_PERIOD", "geo", "sex", "age", "scenario", "OBS_VALUE", "Nationality"])

    # Deduplicate
    before = len(projected)
    projected = projected.drop_duplicates()
    after = len(projected)
    if before != after:
        logger.info("Dropped %d duplicate rows from projected", before - after)

    # Build projected_all: Austria rows from projected + income-group migrants
    projected_at = (
        projected[projected["Nationality"] == "Austria"]
        .rename(columns={"Nationality": "IncomeGroup"})
    )
    projected_all = pd.concat([projected_mig_ig, projected_at], ignore_index=True)

    return projected, projected_all, fan_df, clip_log_df


def _safe_concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Concatenate a list of DataFrames, returning an empty stub if empty."""
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_outputs(
    projected: pd.DataFrame,
    projected_all: pd.DataFrame,
    input_path: str | Path,
    outdir: str | Path,
) -> tuple[Path, Path]:
    """Write the two CSV outputs and return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem
    path_projected = outdir / f"negtest_projected_simEU_{stem}_adjasfr-pyfile.csv"
    path_projected_all = outdir / f"negtest_projected_simEU_{stem}_with_ig_adjasfr-pyfile.csv"

    logger.info("Saving projected → %s", path_projected)
    projected.to_csv(path_projected, index=False)

    logger.info("Saving projected_all → %s", path_projected_all)
    projected_all.to_csv(path_projected_all, index=False)

    return path_projected, path_projected_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose EUROPOP projections into native/migrant sub-populations.",
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Path to the projected migration parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--eurostat", default=DEFAULT_EUROSTAT,
        help="Path to the Eurostat-approximating baseline parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help="Directory for CSV outputs (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    projected, projected_all, fan_df, clip_log_df = run_pipeline(
        input_path=args.input,
        eurostat_path=args.eurostat,
    )

    path_proj, path_all = save_outputs(
        projected, projected_all,
        input_path=args.input,
        outdir=args.outdir,
    )

    # Quick sanity check
    logger.info(
        "Done. projected: %d rows, projected_all: %d rows",
        len(projected), len(projected_all),
    )
    logger.info("  → %s", path_proj)
    logger.info("  → %s", path_all)


if __name__ == "__main__":
    main()