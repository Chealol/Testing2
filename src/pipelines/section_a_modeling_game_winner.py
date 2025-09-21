
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section A — Modeling (Game Winner):
Target, Tokens, Schema, Vocabularies, and Environment Features.

Usage (example):
    import pandas as pd
    from section_a_modeling_game_winner import build_games_core, save_vocabs

    # schedules: DataFrame from nfl_data_py.import_schedules(years)
    games_core, vocabs = build_games_core(schedules=schedules, tie_mode="drop")

    # Optionally save
    games_core.to_parquet("games_core.parquet", index=False)
    save_vocabs(vocabs, "vocabs.json")

This module avoids network or package installs. It focuses on:
- Defining the label y_home
- Creating robust env features for the [GAME] token
- Preparing ID fields for [HOME_TEAM] and [AWAY_TEAM] tokens
- Building reproducible vocabularies with PAD=0, UNK=1
- Enforcing clean dtypes and providing safety checks

Next (Section B): compute rolled team-week numeric features and join as home_* / away_* blocks.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("section_a_modeling")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
TEAM_ALIASES = {
    # Common historical aliases → modern abbreviations used by nflverse
    "OAK": "LV",     # Oakland Raiders → Las Vegas Raiders
    "SD":  "LAC",    # San Diego Chargers → Los Angeles Chargers
    "STL": "LAR",    # St. Louis Rams → Los Angeles Rams
    "WSH": "WAS",    # Washington Redskins → Washington Football Team / Commanders
    "WFT": "WAS",    # Washington Football Team → WAS
    # Occasionally seen variants
    "JAC": "JAX",
    "LA":  "LAR",    # When older data uses 'LA'
}

PRIME_HOURS = {20, 21}  # 8–9 pm local/ET; adjust if your schedules use different tz

def _normalize_team(abbr: str) -> str:
    if pd.isna(abbr):
        return "UNK"
    abbr = str(abbr).strip().upper()
    return TEAM_ALIASES.get(abbr, abbr)

def _to_str(x) -> str:
    return "UNK" if pd.isna(x) else str(x)

def _parse_hour(gametime: str) -> int:
    """Parse 'HH:MM' → int hour; returns -1 on failure."""
    try:
        parts = str(gametime).split(":")
        return int(parts[0])
    except Exception:
        return -1

# ---------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

@dataclass
class Vocabs:
    team2idx: Dict[str, int]
    qb2idx: Dict[str, int]
    coach2idx: Dict[str, int]

def _build_vocab(values: pd.Series) -> Dict[str, int]:
    """Build vocab with PAD=0, UNK=1, then sorted unique tokens."""
    uniq = sorted({_to_str(x) for x in values.dropna().unique()})
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    # Start assigning from index 2
    for i, tok in enumerate(uniq, start=2):
        vocab[tok] = i
    return vocab

def _encode(series: pd.Series, vocab: Dict[str, int]) -> pd.Series:
    return series.map(lambda x: vocab.get(_to_str(x), vocab[UNK_TOKEN])).astype("int32")

def save_vocabs(vocabs: Vocabs, path: str | Path) -> None:
    data = {
        "team2idx": vocabs.team2idx,
        "qb2idx": vocabs.qb2idx,
        "coach2idx": vocabs.coach2idx,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_vocabs(path: str | Path) -> Vocabs:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Vocabs(**data)

# ---------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------
# REQUIRED_SCHEDULE_COLS: drop 'home_rest','away_rest'
REQUIRED_SCHEDULE_COLS = [
    "game_id","season","game_type","week",
    "home_team","away_team",
    "home_score","away_score",
    "roof","surface","temp","wind",
    "gametime","weekday",
    "home_qb_id","away_qb_id",
    "home_coach","away_coach",
]


def _check_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Schedules is missing required columns: {missing}")

def _add_label(df: pd.DataFrame, tie_mode: str="drop") -> pd.DataFrame:
    """Add y_home label; handle ties by dropping or setting 0.5."""
    if "home_score" not in df or "away_score" not in df:
        raise KeyError("home_score/away_score not found in schedules.")
    y = (df["home_score"] > df["away_score"]).astype("float32")
    is_tie = (df["home_score"] == df["away_score"]).fillna(False)
    if tie_mode not in {"drop", "half"}:
        raise ValueError("tie_mode must be 'drop' or 'half'")
    if tie_mode == "drop":
        df = df.loc[~is_tie].copy()
        y = y.loc[df.index]
    else:
        y = np.where(is_tie, 0.5, y).astype("float32")
    df["y_home"] = y
    return df

def _one_hot(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col].fillna("UNK").astype(str), prefix=prefix, dtype="int8")
    return dummies

def _build_game_env(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # robust rest_diff (Cell 5 recomputes from rest_travel; this is a harmless placeholder)
    hr = pd.to_numeric(df.get("home_rest"), errors="coerce")
    ar = pd.to_numeric(df.get("away_rest"), errors="coerce")
    out["rest_diff"] = hr.fillna(0).astype("float32") - ar.fillna(0).astype("float32")
    
    # division + prime flags
    out["is_div"] = df.get("div_game", 0)
    if out["is_div"].dtype != "int8":
        out["is_div"] = out["is_div"].fillna(0).astype("int8")

    # parse hour from HH:MM and mark 20–21 as prime
    hours = df["gametime"].astype(str).str.split(":").str[0].astype("int", errors="ignore")
    out["is_prime"] = hours.isin([20,21]).astype("int8")

    # numeric env with med-fill
    for col in ["temp","wind"]:
        x = pd.to_numeric(df[col], errors="coerce").astype("float32")
        med = float(np.nanmedian(x)) if x.isna().any() else float(x.median())
        out[col] = x.fillna(med)

    # **sanitize categorical labels** to avoid trailing spaces like 'surface_grass '
    roof_clean    = df["roof"].astype(str).str.strip().str.lower().replace({"nan":"UNK"})
    surface_clean = df["surface"].astype(str).str.strip().str.lower().replace({"nan":"UNK"})

    roof_oh = pd.get_dummies(roof_clean,    prefix="roof",    dtype="int8")
    surf_oh = pd.get_dummies(surface_clean, prefix="surface", dtype="int8")

    # ensure unique, trimmed columns
    roof_oh.columns = [c.strip() for c in roof_oh.columns]
    surf_oh.columns = [c.strip() for c in surf_oh.columns]

    out = pd.concat([out, roof_oh, surf_oh], axis=1)
    return out

def build_games_core(
    schedules: pd.DataFrame,
    tie_mode: str = "drop",
    keep_season_type: str = "REG",
) -> Tuple[pd.DataFrame, Vocabs]:
    """
    Build per-game schema for a 3-token Transformer:
      [GAME], [HOME_TEAM], [AWAY_TEAM] + label y_home.

    Returns:
      games_core (DataFrame): one row per game with env features, IDs, and label
      vocabs (Vocabs): team/qb/coach vocabularies with PAD/UNK

    Notes:
      - This function does NOT compute home_/away_ numeric blocks (Section B).
      - You should feed the returned DataFrame to your feature join step next.
    """
    df = schedules.copy()
    _check_required_columns(df, REQUIRED_SCHEDULE_COLS)

    # Filter season type
    if keep_season_type:
        df = df[df["game_type"] == keep_season_type].copy()

    # Normalize team abbreviations for stability
    df["home_team"] = df["home_team"].map(_normalize_team)
    df["away_team"] = df["away_team"].map(_normalize_team)

    # Target
    df = _add_label(df, tie_mode=tie_mode)

    # GAME env features
    env = _build_game_env(df)

    # Core identifiers
    core_cols = ["game_id","season","week","home_team","away_team",
                 "home_qb_id","away_qb_id","home_coach","away_coach","y_home"]
    core = df[core_cols].reset_index(drop=True)

    # Build vocabularies (PAD=0, UNK=1)
    team2idx  = _build_vocab(pd.concat([core["home_team"], core["away_team"]]))
    qb2idx    = _build_vocab(pd.concat([core["home_qb_id"], core["away_qb_id"]]))
    coach2idx = _build_vocab(pd.concat([core["home_coach"], core["away_coach"]]))
    vocabs = Vocabs(team2idx=team2idx, qb2idx=qb2idx, coach2idx=coach2idx)

    # Encode IDs
    core["home_team_id"]   = _encode(core["home_team"], team2idx)
    core["away_team_id"]   = _encode(core["away_team"], team2idx)
    core["home_qb_idx"]    = _encode(core["home_qb_id"], qb2idx)
    core["away_qb_idx"]    = _encode(core["away_qb_id"], qb2idx)
    core["home_coach_idx"] = _encode(core["home_coach"], coach2idx)
    core["away_coach_idx"] = _encode(core["away_coach"], coach2idx)

    # Merge env features
    games_core = pd.concat([core, env.reset_index(drop=True)], axis=1)

    # Types & ordering
    id_int_cols = ["home_team_id","away_team_id","home_qb_idx","away_qb_idx","home_coach_idx","away_coach_idx"]
    for c in id_int_cols:
        games_core[c] = games_core[c].astype("int32")

    # order columns: identifiers → ids → env → label
    env_cols = [c for c in games_core.columns if c.startswith("roof_") or c.startswith("surface_")]
    ordered = (
        ["game_id","season","week","home_team","away_team","home_qb_id","away_qb_id","home_coach","away_coach"]
        + ["home_team_id","away_team_id","home_qb_idx","away_qb_idx","home_coach_idx","away_coach_idx"]
        + ["rest_diff","is_div","is_prime","temp","wind"] + env_cols
        + ["y_home"]
    )
    games_core = games_core[ordered].copy()

    # Basic sanity checks
    assert games_core["game_id"].is_unique, "Duplicate game_id rows found."
    assert games_core["y_home"].between(0,1).all(), "y_home must be in [0,1] or 0.5 for ties."
    logger.info("Built games_core: %s rows, %s columns", games_core.shape[0], games_core.shape[1])
    logger.info("Vocab sizes — teams: %d, QBs: %d, coaches: %d",
                len(team2idx), len(qb2idx), len(coach2idx))

    return games_core, vocabs


# ---------------------------------------------------------------------
# CLI (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Section A — Build games_core (3-token schema)")
    parser.add_argument("--schedules_parquet", type=str, required=True, help="Path to schedules parquet/csv")
    parser.add_argument("--out_games_core", type=str, default="games_core.parquet", help="Output parquet path")
    parser.add_argument("--out_vocabs", type=str, default="vocabs.json", help="Output vocabs json path")
    parser.add_argument("--tie_mode", type=str, default="drop", choices=["drop","half"], help="Tie handling")
    args = parser.parse_args()

    # Load schedules (parquet or csv)
    if args.schedules_parquet.endswith(".csv"):
        schedules = pd.read_csv(args.schedules_parquet)
    else:
        schedules = pd.read_parquet(args.schedules_parquet)

    games_core, vocabs = build_games_core(schedules, tie_mode=args.tie_mode)
    # Save
    if args.out_games_core.endswith(".csv"):
        games_core.to_csv(args.out_games_core, index=False)
    else:
        games_core.to_parquet(args.out_games_core, index=False)
    save_vocabs(vocabs, args.out_vocabs)

    print("Wrote:", args.out_games_core, "and", args.out_vocabs)
