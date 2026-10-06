# Cross-market futures: intraday (Yahoo 1h → 4h)

Requires the quant repo with patches `patches/quantitative_trading_projects/000{1,2}-*.patch` applied.

```
CROSSMARKET_DATA=crossmarket/data python crossmarket/fetch_yahoo.py    # data stays local, git-ignored
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t80.py   # train only
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/log_t80.py   # appends T80 + verdict block
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t81.py   # momentum, 1h/4h entries
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/log_t81.py
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t82.py   # VWAP bands + TEMA/EMA, 1h
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/log_t82.py
```

- `intraday.py`: loads the 1h bars, flags contract-mix and roll bars, builds session 4h bars, and
  builds a clean (effectively back-adjusted) index. The docstring describes the Yahoo defects
  and the exclusion policy.
- `trend.py`: Donchian breakout and EMA crossover on the clean index, inverse-vol sizing with
  equal risk per market, plus the statistics.
- `tsmom.py`: time-series momentum. The 12m signal and vol come from roll-cleaned daily bars;
  entries are at the first 1h or 4h bar of each month, and P&L is computed on the clean intraday bars.
- `vwap_study.py`: monthly anchored VWAP with 1/2/3 sigma bands, TEMA(9) and EMA(50), event
  definitions and statistics for T82 (the design is fixed in its docstring).
- `out/test_log.xlsx`: the test log, continuing at T80. `out/t80_results.json`: the raw numbers.

Splits were confirmed on 2026-10-05 and live in `quant.data_splits` (patch 0002). Intraday: train
runs to 2025-09-30, validate to 2026-03-31, holdout after. Daily: train runs to 2014-12-31,
validate to 2020-12-31, holdout from 2021 on. Only train slices are ever loaded into a backtest.
