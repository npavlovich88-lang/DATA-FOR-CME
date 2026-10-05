"""T87 (daily momentum calibration) and T88 (daily trend / momentum / band ideas), TRAIN, with
one pre-declared validate look each for T87 and for T88's train winner. Holdout untouched.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t87_t88.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

import daily
import trend
from quant.data_splits import CM_DAILY_TRAIN_END, CM_DAILY_VALIDATE_END

OUT = pathlib.Path(__file__).parent / "out"
SCORE_START = "2002-01-01"  # every rule has its 200-day / 12-month history by then
N_NULL = 200
BLOCKS = [("2002", "2004"), ("2005", "2009"), ("2010", "2014")]


def cut(bars: dict, end: str) -> dict:
    return {s: b[b.index < pd.Timestamp(end)] for s, b in bars.items()}


def score(d: pd.Series, start: str, end: str) -> pd.Series:
    return d[(d.index >= pd.Timestamp(start)) & (d.index < pd.Timestamp(end))]


def describe(d: pd.Series, per: pd.DataFrame) -> dict:
    s = trend.summary(d)
    s["blocks"] = {f"{a}-{b}": trend.sharpe(d[a:b]) for a, b in BLOCKS if len(d[a:b]) > 50}
    s["per_market_ann_ret"] = (per.loc[d.index].mean() * 252).round(4).to_dict()
    return s


def main() -> None:
    allbars = {s: daily.load(s) for s in daily.DAILY}
    train = cut(allbars, CM_DAILY_TRAIN_END)
    res: dict = {"train": {}, "validate": {}}
    obs = {}
    for st in daily.STRATS:
        d, per = daily.run(train, st)
        d = score(d, SCORE_START, CM_DAILY_TRAIN_END)
        r = describe(d, per)
        r["cost3x_sharpe"] = trend.sharpe(
            score(daily.run(train, st, cost_mult=3.0)[0], SCORE_START, CM_DAILY_TRAIN_END)
        )
        res["train"][st] = r
        obs[st] = r["sharpe"]
        print(f"{st:<14} SR {r['sharpe']:+.2f}")
    rng = np.random.default_rng(87)
    null = []
    for _ in range(N_NULL):
        pb = daily.permute_months(train, rng)
        null.append(
            {
                st: trend.sharpe(score(daily.run(pb, st)[0], SCORE_START, CM_DAILY_TRAIN_END))
                for st in daily.STRATS
            }
        )
    null = pd.DataFrame(null)
    null.to_csv(OUT / "t87_t88_null.csv", index=False)
    ranks = null.rank(ascending=False, method="max") / (len(null) + 1)
    minp = ranks.min(axis=1)
    for st in daily.STRATS:
        p = (1 + (null[st] >= obs[st]).sum()) / (len(null) + 1)
        res["train"][st]["null_p"] = float(p)
        res["train"][st]["null_median"] = float(null[st].median())
        res["train"][st]["family_p_minp"] = float((minp <= p).mean())

    # ---- pre-declared validate looks: the calibration rule, and T88's train winner
    winner = max(daily.STRATS[1:], key=lambda k: obs[k])
    res["t88_winner"] = winner
    val = cut(allbars, CM_DAILY_VALIDATE_END)  # train bars warm the indicators, strictly past
    for st in ("tsmom_12m", winner):
        d, per = daily.run(val, st)
        d = score(d, CM_DAILY_TRAIN_END, CM_DAILY_VALIDATE_END)
        v = trend.summary(d)
        v["per_market_ann_ret"] = (per.loc[d.index].mean() * 252).round(4).to_dict()
        v["by_year"] = {str(y): trend.sharpe(g) for y, g in d.groupby(d.index.year)}
        res["validate"][st] = v
        with (OUT / "looks.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "date": str(dt.date.today()),
                        "test": "T87" if st == "tsmom_12m" else "T88",
                        "segment": "daily validate",
                        "rule": st,
                        "look": 1,
                    }
                )
                + "\n"
            )
    (OUT / "t87_t88_results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
