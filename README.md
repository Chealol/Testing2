# NFL Win Transformer

This repo predicts **game winners** with a 3-token Transformer: `[GAME], [HOME_TEAM], [AWAY_TEAM]`.

## Layout
- `data/raw/` : direct pulls from `nfl_data_py` (immutable)
- `data/interim/` : cleaned/merged (e.g., `games_core.parquet`, `vocabs.json`)
- `data/processed/` : model-ready tables (joined home/away/team features)
- `src/pipelines/` : Section A/B/C data pipelines
- `src/modeling/` : token encoder, Transformer, heads
- `src/train/` : train/backtest CLIs
- `notebooks/` : ordered exploration & runs