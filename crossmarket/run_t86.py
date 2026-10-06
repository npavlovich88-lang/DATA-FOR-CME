"""T86: the pre-registered T84 rule on the HOLDOUT. One look, ever.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t86.py

Identical to run_t84 except the scored segment: sessions from VALIDATE_END to the last complete
bar. Indicators are warm from train + validate bars, which are strictly in the past.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

import run_t83 as t83
import trend
import vwap_study as vs
from intraday import INTRADAY, VALIDATE_END, load_1h_clean

OUT = pathlib.Path(__file__).parent / "out"
PREREG = json.loads((pathlib.Path(__file__).parent / "preregistered" / "T86.json").read_text())
RULE, HOLD, N_NULL = "reenter 1sd | recent TEMAxEMA cross", 23, 200
END = "2100-01-01"  # everything pulled; the live partial bar is dropped by load_1h


def load_holdout():
    markets, events = {}, {}
    for s in INTRADAY:
        b = load_1h_clean(s)
        ind = vs.build(s, END, anchor="Q")
        assert ind.index.equals(b.index)
        in_h = (pd.to_datetime(ind["session"]) >= pd.Timestamp(VALIDATE_END)).to_numpy()
        ind["scored"] = ind["scored"] & in_h
        hvol = np.sqrt((b["ret"] ** 2).ewm(span=vs.VOL_SPAN, min_periods=vs.VOL_SPAN).mean())
        key = pd.DataFrame({"session": ind["session"].to_numpy(), "hour": ind["hour"].to_numpy()})
        markets[s] = {
            "bars": b,
            "size": (t83.RISK / (hvol * np.sqrt(23))).to_numpy(),
            "key": key,
            "pos": {
                (ss, h): i
                for i, (ss, h) in enumerate(zip(key["session"], key["hour"], strict=True))
            },
            "scored": ind["scored"].to_numpy(),
            "in_h": in_h,
        }
        events[s] = {RULE: vs.events(ind.reset_index(drop=True))[RULE].to_numpy() * in_h}
    return markets, events


def holdout_only(c: dict) -> dict:
    c = dict(c)
    c["port"] = c["port"][c["port"].index >= pd.Timestamp(VALIDATE_END)]
    c["per"] = c["per"][c["per"].index >= pd.Timestamp(VALIDATE_END)]
    return c


def main() -> None:
    markets, events = load_holdout()
    res = {"prereg": PREREG}
    for label, kw in (("reference", {}), ("challenge", {"commission_all_in": 2.50})):
        c = holdout_only(t83.cell(markets, events, RULE, HOLD, **kw))
        res[label] = t83.stats(c)
        if label == "reference":
            daily, trades = c["port"], c["trades"]
    rng = np.random.default_rng(86)
    null = []
    for _ in range(N_NULL):
        pev = t83.permuted_events(markets, events, rng)
        for s in pev:
            pev[s][RULE] = pev[s][RULE] * markets[s]["in_h"]
        null.append(trend.sharpe(holdout_only(t83.cell(markets, pev, RULE, HOLD))["port"]))
    null = np.asarray(null)
    sr = res["reference"]["sharpe"]
    res["null_p"] = float((1 + (null >= sr).sum()) / (len(null) + 1))
    res["null_median"] = float(np.median(null))
    res["confirmed"] = bool(
        sr > 0 and res["reference"]["trade_mean_R"] > 0 and res["null_p"] < 0.05
    )
    m = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    res["monthly"] = {str(k): float(v) for k, v in m.items()}
    res["window"] = [str(daily.index[0].date()), str(daily.index[-1].date())]
    res["trades_by_month"] = {
        str(k): int(v)
        for k, v in trades.groupby(pd.to_datetime(trades["session"]).dt.to_period("M"))
        .size()
        .items()
    }
    (OUT / "t86_results.json").write_text(json.dumps(res, indent=1, default=float))
    with (OUT / "looks.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "date": str(dt.date.today()),
                    "test": "T86",
                    "segment": "holdout",
                    "rule": f"Q | {RULE} | hold 1s",
                    "look": 1,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
