
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section B — Team-week rolling features for game winner model.
- Imports PBP (selected columns) using nfl_data_py
- Computes offense/defense team-week aggregates
- Builds shifted rolling means (L4, L8) and deltas
- Writes a compact table keyed by (season, week, team)

Output:
  data/interim/team_week_features.parquet
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from pathlib import Path

PBP_COLS = [
    'game_id','game_date','season_type','season','week',
    'posteam','defteam','epa','success','pass','rush','down',
    'yardline_100','pass_touchdown','rush_touchdown',
    'penalty','penalty_yards'
]

def load_pbp_years(years):
    dfs = []
    for y in years:
        df = nfl.import_pbp_data([y], columns=PBP_COLS, downcast=True)
        dfs.append(df)
    pbp = pd.concat(dfs, ignore_index=True)
    pbp = pbp[pbp['season_type']=='REG'].copy()
    pbp['game_date'] = pd.to_datetime(pbp['game_date'])
    pbp['is_early']  = pbp['down'].isin([1,2]).astype('float32')
    pbp['is_rz']     = (pbp['yardline_100'] <= 20).astype('float32')
    pbp['td']        = ((pbp['pass_touchdown']==1) | (pbp['rush_touchdown']==1)).astype('float32')
    pbp['penalty']   = pbp['penalty'].fillna(0).astype('float32')
    pbp['penalty_yards'] = pbp['penalty_yards'].fillna(0).astype('float32')
    return pbp

def _agg_team_week(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    g = df.groupby(['season','week', team_col], observed=True)
    agg = g.agg(
        plays=('epa','size'),
        epa_per_play=('epa','mean'),
        success_rate=('success','mean'),
        pass_rate=('pass','mean'),
        ed_pass_n=('pass', lambda s: float((s*df.loc[s.index,'is_early']).sum())),
        ed_plays=('is_early','sum'),
        rz_td_n=('td', lambda s: float((s*df.loc[s.index,'is_rz']).sum())),
        rz_plays=('is_rz','sum'),
        penalty_rate=('penalty','mean'),
        penalty_yds=('penalty_yards','sum'),
    ).reset_index().rename(columns={team_col:'team'})

    # safe rates
    agg['ed_pass_rate'] = np.where(agg['ed_plays']>0, agg['ed_pass_n']/agg['ed_plays'], 0.0).astype('float32')
    agg['rz_td_rate']   = np.where(agg['rz_plays']>0, agg['rz_td_n']/agg['rz_plays'], 0.0).astype('float32')
    agg['penalty_yds_per_play'] = np.where(agg['plays']>0, agg['penalty_yds']/agg['plays'], 0.0).astype('float32')

    keep = ['season','week','team','plays','epa_per_play','success_rate','pass_rate',
            'ed_pass_rate','rz_td_rate','penalty_rate','penalty_yds_per_play']
    return agg[keep].copy()

def _roll_shift_means(df: pd.DataFrame, group_key: str, cols: list[str], wins=(4,8)) -> pd.DataFrame:
    df = df.sort_values([group_key,'season','week'])
    out = df[['season','week',group_key]].copy()
    for k in cols:
        s = df.groupby(group_key, observed=True)[k].apply(lambda x: x.shift(1))
        s = s.reset_index(level=0, drop=True)
        out[k+'_l4_mean'] = s.groupby(df[group_key], observed=True).apply(lambda x: x.rolling(4, min_periods=1).mean()).values.astype('float32')
        out[k+'_l8_mean'] = s.groupby(df[group_key], observed=True).apply(lambda x: x.rolling(8, min_periods=1).mean()).values.astype('float32')
        out[k+'_delta_l4_l8'] = (out[k+'_l4_mean'] - out[k+'_l8_mean']).astype('float32')
    return out

def build_team_week_features(years: list[int]) -> pd.DataFrame:
    pbp = load_pbp_years(years)

    off = _agg_team_week(pbp[pbp['posteam'].notna()], 'posteam')
    off = off.add_prefix('off_').rename(columns={'off_season':'season','off_week':'week','off_team':'team'})

    deff = _agg_team_week(pbp[pbp['defteam'].notna()], 'defteam')
    deff = deff.add_prefix('def_').rename(columns={'def_season':'season','def_week':'week','def_team':'team'})

    tw = off.merge(deff, on=['season','week','team'], how='outer').fillna(0.0)

    # choose numeric columns to roll
    num_cols = [c for c in tw.columns if c not in ['season','week','team']]
    rolled = _roll_shift_means(tw, 'team', num_cols, wins=(4,8))

    # merge rolled back with keys
    feats = tw[['season','week','team']].merge(rolled, on=['season','week','team'], how='left')
    feats = feats.fillna(0.0)
    return feats

def main(years: list[int], out_path: str | Path):
    out_path = Path(out_path)
    feats = build_team_week_features(years)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print("Wrote:", out_path, "shape:", feats.shape)

if __name__ == "__main__":
    YEARS = list(range(2018, 2025))
    OUT = Path(__file__).resolve().parents[3] / "data" / "interim" / "team_week_features.parquet"
    main(YEARS, OUT)
