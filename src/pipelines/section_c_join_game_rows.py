#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section C — Join Section A (games_core) with enriched Section B outputs:
- team_week_features
- injury_counts
- rest_travel
- qb_form
- qb_flags
Then add GAME-level features (weather/altitude, mismatches, volatility, luck),
avoid duplicate columns, and write data/processed/game_rows.parquet
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

def _parse_precip(text):
    if pd.isna(text): return pd.Series([0,0])
    s = str(text).lower()
    precip = int(any(w in s for w in ['rain','shower','drizzle','storm','precip']))
    snow   = int(any(w in s for w in ['snow','sleet','flurr','blizzard','hail']))
    return pd.Series([precip, snow])

def _join_side(core: pd.DataFrame, side_df: pd.DataFrame, side: str) -> pd.DataFrame:
    assert side in ("home","away")
    team_col = f"{side}_team"
    cols_keep = [c for c in side_df.columns if c not in ("season","week","team")]
    out = core.merge(
        side_df[["season","week","team"] + cols_keep],
        left_on=["season","week",team_col],
        right_on=["season","week","team"],
        how="left"
    ).drop(columns=["team"])
    rename = {c: f"{side}_{c}" for c in cols_keep}
    return out.rename(columns=rename)

def join_games_with_team_features(
    games_core_path: str | Path,
    team_week_feats_path: str | Path,
    injury_counts_path: str | Path,
    rest_travel_path: str | Path,
    qb_form_path: str | Path,
    qb_flags_path: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:

    games = pd.read_parquet(games_core_path)
    tw    = pd.read_parquet(team_week_feats_path)
    inj   = pd.read_parquet(injury_counts_path)
    rest  = pd.read_parquet(rest_travel_path)
    qbf   = pd.read_parquet(qb_form_path)
    qbf2  = pd.read_parquet(qb_flags_path)

    # sanity
    for df_chk, name in [(tw,'team_week_features'),(inj,'injury_counts'),
                         (rest,'rest_travel'),(qbf,'qb_form'),(qbf2,'qb_flags')]:
        for c in ["season","week","team"]:
            if c not in df_chk.columns:
                raise ValueError(f"{name} missing key column '{c}'")

    # Weather text → precip/snow; altitude proxy
    df = games.copy()
    wcol = "game_weather" if "game_weather" in df.columns else ("weather" if "weather" in df.columns else None)
    if wcol:
        df[["precip","snow"]] = df[wcol].apply(_parse_precip)
    else:
        df["precip"] = 0; df["snow"] = 0
    df["altitude_flag"] = (df["home_team"]=="DEN").astype("int8")

    # roof/surface one-hots: only add if they don't already exist
    has_roof = any(c.startswith("roof_") for c in df.columns)
    has_surf = any(c.startswith("surface_") for c in df.columns)
    if "roof" in df.columns and not has_roof:
        d = pd.get_dummies(df["roof"].astype(str).str.lower(), prefix="roof", dtype="float32")
        df = pd.concat([df, d], axis=1)
    if "surface" in df.columns and not has_surf:
        d = pd.get_dummies(df["surface"].astype(str).str.lower(), prefix="surface", dtype="float32")
        df = pd.concat([df, d], axis=1)

    # Attach home/away blocks
    df = _join_side(df, tw,   "home")
    df = _join_side(df, tw,   "away")
    df = _join_side(df, inj,  "home")
    df = _join_side(df, inj,  "away")
    df = _join_side(df, rest, "home")
    df = _join_side(df, rest, "away")
    df = _join_side(df, qbf,  "home")
    df = _join_side(df, qbf,  "away")
    df = _join_side(df, qbf2, "home")
    df = _join_side(df, qbf2, "away")

    # GAME-level derived features (mismatches, volatility, luck)
    def ok(c): return c in df.columns

    if ok("home_rest_days") and ok("away_rest_days"):
        df["rest_diff"] = df["home_rest_days"].fillna(0) - df["away_rest_days"].fillna(0)
    if ok("home_east_travel") and ok("away_east_travel"):
        df["east_travel_diff"] = df["home_east_travel"].fillna(0) - df["away_east_travel"].fillna(0)
    if ok("home_west_travel") and ok("away_west_travel"):
        df["west_travel_diff"] = df["home_west_travel"].fillna(0) - df["away_west_travel"].fillna(0)

    if ok("home_pass_epa_pp_l8") and ok("away_def_pass_epa_pp_l8"):
        df["pass_mismatch"] = df["home_pass_epa_pp_l8"] - df["away_def_pass_epa_pp_l8"]
    if ok("home_rush_epa_pp_l8") and ok("away_def_rush_epa_pp_l8"):
        df["rush_mismatch"] = df["home_rush_epa_pp_l8"] - df["away_def_rush_epa_pp_l8"]

    if ok("home_exp_pass_rate_l8") and ok("away_def_exp_pass_rate_l8"):
        df["explosive_pass_mismatch"] = df["home_exp_pass_rate_l8"] - df["away_def_exp_pass_rate_l8"]
    if ok("home_exp_run_rate_l8") and ok("away_def_exp_run_rate_l8"):
        df["explosive_run_mismatch"]  = df["home_exp_run_rate_l8"]  - df["away_def_exp_run_rate_l8"]
    if ok("home_exp_rate_l8") and ok("away_exp_rate_l8"):
        df["explosive_reliance_sum"]  = df["home_exp_rate_l8"].fillna(0) + df["away_exp_rate_l8"].fillna(0)

    if ok("home_sr_3d_pass_shr_l8") and ok("away_def_sr_3d_pass_l8"):
        df["3d_pass_mismatch"] = df["home_sr_3d_pass_shr_l8"] - df["away_def_sr_3d_pass_l8"]
    if ok("home_sr_3d_run_shr_l8") and ok("away_def_sr_3d_run_l8"):
        df["3d_run_mismatch"]  = df["home_sr_3d_run_shr_l8"]  - df["away_def_sr_3d_run_l8"]

    if ok("home_pass_epa_pp_trend") and ok("away_def_pass_epa_pp_trend"):
        df["momentum_mismatch_pass"] = df["home_pass_epa_pp_trend"] - df["away_def_pass_epa_pp_trend"]
    if ok("home_rush_epa_pp_trend") and ok("away_def_rush_epa_pp_trend"):
        df["momentum_mismatch_rush"] = df["home_rush_epa_pp_trend"] - df["away_def_rush_epa_pp_trend"]

    vol_home = df["home_epa_sd_l8"] if ok("home_epa_sd_l8") else 0
    vol_away = df["away_epa_sd_l8"] if ok("away_epa_sd_l8") else 0
    df["volatility_sum"] = pd.Series(vol_home).fillna(0) + pd.Series(vol_away).fillna(0)

    luck_h = df["home_fum_luck_l8"] if ok("home_fum_luck_l8") else 0
    luck_a = df["away_fum_luck_l8"] if ok("away_fum_luck_l8") else 0
    df["turnover_luck_gap"] = pd.Series(luck_h).fillna(0) - pd.Series(luck_a).fillna(0)

    if ok("home_st_epa_mean_l8") and ok("away_def_st_epa_mean_l8"):
        df["st_mismatch"] = df["home_st_epa_mean_l8"] - df["away_def_st_epa_mean_l8"]
    if ok("home_k_fg4055_rate_l8") and ok("away_k_fg4055_rate_l8"):
        df["kicker_edge_4055"] = df["home_k_fg4055_rate_l8"] - df["away_k_fg4055_rate_l8"]

    df["upset_propensity"] = (
        df["volatility_sum"].fillna(0).clip(lower=0) +
        df["turnover_luck_gap"].abs().fillna(0) +
        (df["momentum_mismatch_pass"].abs().fillna(0) if "momentum_mismatch_pass" in df else 0) +
        (df["momentum_mismatch_rush"].abs().fillna(0) if "momentum_mismatch_rush" in df else 0)
    )

    # Deduplicate any duplicate columns before saving (parquet requires unique names)
    if df.columns.duplicated().any():
        dup = list(df.columns[df.columns.duplicated()].unique())
        print("[WARN] Duplicate column names detected; dropping later duplicates:", dup)
        df = df.loc[:, ~df.columns.duplicated()]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print("Wrote:", out_path, "shape:", df.shape)
    return df

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[3]
    join_games_with_team_features(
        games_core_path   = BASE/'data'/'interim'/'games_core.parquet',
        team_week_feats_path = BASE/'data'/'interim'/'team_week_features.parquet',
        injury_counts_path   = BASE/'data'/'interim'/'injury_counts.parquet',
        rest_travel_path     = BASE/'data'/'interim'/'rest_travel.parquet',
        qb_form_path         = BASE/'data'/'interim'/'qb_form.parquet',
        qb_flags_path        = BASE/'data'/'interim'/'qb_flags.parquet',
        out_path             = BASE/'data'/'processed'/'game_rows.parquet'
    )
