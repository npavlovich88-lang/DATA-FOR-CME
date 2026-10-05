"""T81: time-series momentum replication (12m lookback, 1m hold, inverse vol) with 1h and 4h
entries, intraday TRAIN only.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t81.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import trend
import tsmom
from intraday import INTRADAY, TRAIN_END, load_1h_clean, load_4h, load_daily_clean

OUT = pathlib.Path(__file__).parent / "out"
SCORE_START = "2024-11-01"  # first month whose entry bar exists: the 1h pull starts 2024-10-06
LAG_SESSION = {"1h": 23, "4h": 6}  # one full Globex session later, the timing sensitivity


def cut(df: pd.DataFrame) -> pd.DataFrame:
    """Nothing past train is ever loaded. Intraday bars are cut by SESSION, not timestamp: the
    evening of 2025-09-30 opens the 2025-10-01 session, which is validate."""
    if "session" in df:
        return df[pd.to_datetime(df["session"]) < pd.Timestamp(TRAIN_END)]
    return df[df.index < pd.Timestamp(TRAIN_END)]


def scored(d: pd.DataFrame | pd.Series):
    return d[d.index >= SCORE_START]


def main() -> None:
    dailies = {s: cut(load_daily_clean(s)) for s in INTRADAY}
    loaders = {"1h": load_1h_clean, "4h": load_4h}
    res = {}
    for tf, load in loaders.items():
        bars = {s: cut(load(s)) for s in INTRADAY}
        net, lg = tsmom.run(bars, dailies)
        net, lg = scored(net), scored(lg)
        d = net.sum(axis=1)
        gross_d = scored(tsmom.run(bars, dailies, cost_mult=0.0)[0]).sum(axis=1)
        null = tsmom.random_sign_null(lg)
        monthly = (1 + d).groupby(d.index.to_period("M")).prod() - 1
        half = len(d) // 2
        best_month = monthly.idxmax()
        res[tf] = dict(
            base=trend.summary(d),
            gross_sharpe=trend.sharpe(gross_d),
            cost3x=trend.summary(scored(tsmom.run(bars, dailies, cost_mult=3.0)[0]).sum(axis=1)),
            lag_session=trend.summary(
                scored(tsmom.run(bars, dailies, lag_bars=LAG_SESSION[tf])[0]).sum(axis=1)
            ),
            halves=(trend.sharpe(d.iloc[:half]), trend.sharpe(d.iloc[half:])),
            sharpe_ex_best_month=trend.sharpe(d[d.index.to_period("M") != best_month]),
            best_month=str(best_month),
            null_p=float((null >= trend.sharpe(gross_d)).mean()),
            null_q95=float(np.quantile(null, 0.95)),
            monthly={str(k): float(v) for k, v in monthly.round(4).items()},
            per_market_ann_ret=(net.mean() * 252).round(4).to_dict(),
            window=(str(d.index[0].date()), str(d.index[-1].date())),
        )
    t = pd.DataFrame({s: tsmom.monthly_targets(dailies[s], len(INTRADAY))["ret12"] for s in INTRADAY})
    t = t.loc[pd.Period(SCORE_START, "M") :]
    res["positions"] = {str(k): "".join("L" if v > 0 else "S" for v in row) for k, row in t.iterrows()}
    res["markets"] = INTRADAY
    OUT.mkdir(exist_ok=True)
    (OUT / "t81_results.json").write_text(
        json.dumps(res, indent=1, default=lambda x: float(x) if not isinstance(x, pd.Period) else str(x))
    )


if __name__ == "__main__":
    main()
