# Cross-market futures: intraday (Yahoo 1h → 4h)

Requires `quant.costs` with patch `patches/quantitative_trading_projects/0001-*.patch`.

```
CROSSMARKET_DATA=crossmarket/data python crossmarket/fetch_yahoo.py    # data stays local, git-ignored
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t80.py   # train only
PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/log_t80.py   # appends T80 + verdict block
```

- `intraday.py`: loads the 1h bars, flags contract-mix and roll bars, builds session 4h bars, and
  builds a clean (effectively back-adjusted) index. The docstring describes the Yahoo defects
  and the exclusion policy.
- `trend.py`: Donchian breakout and EMA crossover on the clean index, inverse-vol sizing with
  equal risk per market, plus the statistics.
- `out/test_log.xlsx`: the test log, continuing at T80. `out/t80_results.json`: the raw numbers.

The intraday split dates in `intraday.py` (train to 2025-09-30, validate to 2026-03-31, holdout
after) are proposed and not yet confirmed. Only the train slice is ever loaded into a backtest.
