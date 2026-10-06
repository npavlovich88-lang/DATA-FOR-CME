"""T84: the pre-registered quarterly-VWAP rule (preregistered/T84.json) on VALIDATE. One look.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t84.py

Indicators are computed over train + validate so the EMA/TEMA and the trailing vol are warm on
the first validate bar -- train bars are strictly in the past of every validate bar. Entries,
scoring and P&L are VALIDATE ONLY. Nothing past VALIDATE_END is loaded.
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
from intraday import INTRADAY, TRAIN_END, VALIDATE_END, load_1h_clean

OUT = pathlib.Path(__file__).parent / "out"
PREREG = json.loads((pathlib.Path(__file__).parent / "preregistered" / "T84.json").read_text())
RULE = "reenter 1sd | recent TEMAxEMA cross"
HOLD = 23
N_NULL = 200
LOOKS = OUT / "looks.jsonl"


def load_validate() -> tuple[dict, dict]:
    markets, events = {}, {}
    for s in INTRADAY:
        b = load_1h_clean(s)
        sess = pd.to_datetime(b["session"])
        b = b[sess < pd.Timestamp(VALIDATE_END)]
        ind = vs.build(s, VALIDATE_END, anchor="Q")
        assert ind.index.equals(b.index)
        in_val = (pd.to_datetime(ind["session"]) >= pd.Timestamp(TRAIN_END)).to_numpy()
        ind["scored"] = ind["scored"] & in_val
        hvol = np.sqrt((b["ret"] ** 2).ewm(span=vs.VOL_SPAN, min_periods=vs.VOL_SPAN).mean())
        size = (t83.RISK / (hvol * np.sqrt(23))).to_numpy()
        key = pd.DataFrame({"session": ind["session"].to_numpy(), "hour": ind["hour"].to_numpy()})
        markets[s] = {
            "bars": b,
            "size": size,
            "key": key,
            "pos": {
                (ss, h): i
                for i, (ss, h) in enumerate(zip(key["session"], key["hour"], strict=True))
            },
            "scored": ind["scored"].to_numpy(),
            "in_val": in_val,
        }
        ev = vs.events(ind.reset_index(drop=True))[RULE].to_numpy()
        events[s] = {RULE: ev * in_val}
    return markets, events


def validate_only(c: dict, markets: dict) -> dict:
    """Restrict the daily series to validate sessions (positions can only open there)."""
    c = dict(c)
    c["port"] = c["port"][c["port"].index >= pd.Timestamp(TRAIN_END)]
    c["per"] = c["per"][c["per"].index >= pd.Timestamp(TRAIN_END)]
    return c


def main() -> None:
    markets, events = load_validate()
    res = {"prereg": PREREG}
    for label, kw in (("reference", {}), ("challenge", {"commission_all_in": 2.50})):
        c = validate_only(t83.cell(markets, events, RULE, HOLD, **kw), markets)
        res[label] = t83.stats(c)
        if label == "reference":
            daily = c["port"]
            trades = c["trades"]
    rng = np.random.default_rng(84)
    null = []
    for _ in range(N_NULL):
        pev = t83.permuted_events(markets, events, rng)
        for s in pev:  # a permuted event may not land outside validate
            pev[s][RULE] = pev[s][RULE] * markets[s]["in_val"]
        null.append(
            trend.sharpe(validate_only(t83.cell(markets, pev, RULE, HOLD), markets)["port"])
        )
    null = np.asarray(null)
    sr = res["reference"]["sharpe"]
    res["null_p"] = float((1 + (null >= sr).sum()) / (len(null) + 1))
    res["null_median"] = float(np.median(null))
    ref = res["reference"]
    res["confirmed"] = bool(sr > 0 and ref["trade_mean_R"] > 0 and res["null_p"] < 0.05)
    res["monthly"] = {
        str(k): float(v)
        for k, v in ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1).items()
    }
    res["trades_by_month"] = {
        str(k): int(v)
        for k, v in trades.groupby(pd.to_datetime(trades["session"]).dt.to_period("M"))
        .size()
        .items()
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "t84_results.json").write_text(json.dumps(res, indent=1, default=float))
    with LOOKS.open("a") as f:
        f.write(
            json.dumps(
                {
                    "date": str(dt.date.today()),
                    "test": "T84",
                    "segment": "validate",
                    "rule": f"Q | {RULE} | hold 1s",
                    "look": 1,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
