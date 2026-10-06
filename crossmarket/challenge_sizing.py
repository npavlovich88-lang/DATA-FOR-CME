"""Sizing the T84 rule for the 2026 CME Group University Trading Challenge.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/challenge_sizing.py

RULES THAT BIND (2026 rules PDF, pages 9-13)
    start balance            $1,000,000 (the account shows $999,000)
    commission               $2.50 per contract per side, all-in
    daily loss lockout       lose 20% of the available balance in one day -> locked that day
    daily minimum            10 contracts executed per day, else $1,000 penalty
    expiry                   $1,000 per contract not liquidated by expiration
    last day                 flat 15 minutes before the close on Fri 2026-10-30
    ALGORITHMIC TRADING IS NOT PERMITTED -- every order is entered by a person.

WHAT THIS COMPUTES
    1. The rule's daily P&L at 0.25% risk per trade, train + validate, at challenge costs.
    2. Block-bootstrapped 20-session paths (the challenge is 20 sessions): max drawdown and
       worst day at several risk levels -- once with the historical mean, once DEMEANED,
       because T84 did not confirm the edge and the sizing must survive it being zero.
    3. Contracts per signal per market on $999,000 at each risk level, full-size and micro,
       using the latest price and hourly vol. That uses bars past VALIDATE_END for sizing
       only -- the price level and vol, never a P&L -- so nothing is evaluated on holdout.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import run_t83 as t83
import run_t84 as t84
import vwap_study as vs
from intraday import INTRADAY, TRAIN_END, load_1h_clean
from quant.costs import POINT_VALUE

OUT = pathlib.Path(__file__).parent / "out"
CAPITAL = 999_000
RULE, HOLD = "reenter 1sd | recent TEMAxEMA cross", 23
RISKS = (0.0025, 0.005, 0.01, 0.02, 0.03)
SESSIONS = 20
N_PATHS = 20_000
BLOCK = 5

# Micro contracts on CME Globex: USD per 1.0 move in the same quote units as the full contract.
# ZB/ZN have no micro futures in the same units (the micro Treasury contracts are yield-quoted).
MICRO = {
    "ES": ("MES", 5.0), "NQ": ("MNQ", 2.0), "RTY": ("M2K", 5.0), "YM": ("MYM", 0.5),
    "CL": ("MCL", 100.0), "NG": ("MNG", 1_000.0), "HG": ("MHG", 2_500.0),
    "6E": ("M6E", 12_500.0), "6B": ("M6B", 6_250.0), "6J": ("MJY", 1_250_000.0),
}


def daily_returns() -> pd.Series:
    """Train and validate daily P&L of the rule at RISK=0.25%, challenge commissions."""
    a = "Q"
    tm, ti = {}, {}
    for s in INTRADAY:
        b = load_1h_clean(s)
        b = b[pd.to_datetime(b["session"]) < pd.Timestamp(TRAIN_END)]
        ind = vs.build(s, TRAIN_END, anchor=a).reset_index(drop=True)
        hvol = np.sqrt((b["ret"] ** 2).ewm(span=vs.VOL_SPAN, min_periods=vs.VOL_SPAN).mean())
        tm[s] = {"bars": b, "size": (t83.RISK / (hvol * np.sqrt(23))).to_numpy()}
        ti[s] = {RULE: vs.events(ind)[RULE].to_numpy()}
    train = t83.cell(tm, ti, RULE, HOLD, commission_all_in=2.50)["port"]
    vm, ve = t84.load_validate()
    val = t84.validate_only(t83.cell(vm, ve, RULE, HOLD, commission_all_in=2.50), vm)["port"]
    return pd.concat([train, val]).sort_index()


def paths(d: np.ndarray, rng) -> np.ndarray:
    k = int(np.ceil(SESSIONS / BLOCK))
    starts = rng.integers(0, len(d), (N_PATHS, k))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_PATHS, -1)
    return d[idx[:, :SESSIONS] % len(d)]


def path_stats(p: np.ndarray) -> dict:
    eq = np.cumprod(1 + p, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(p), 1)), eq], axis=1), axis=1)[:, 1:]
    dd = (eq / peak - 1).min(axis=1)
    return {
        "median_return": float(np.median(eq[:, -1] - 1)),
        "p05_return": float(np.quantile(eq[:, -1] - 1, 0.05)),
        "p95_return": float(np.quantile(eq[:, -1] - 1, 0.95)),
        "p99_max_dd": float(np.quantile(dd, 0.01)),
        "p999_max_dd": float(np.quantile(dd, 0.001)),
        "p999_worst_day": float(np.quantile(p.min(axis=1), 0.001)),
        "prob_dd_beyond_20pct": float((dd <= -0.20).mean()),
    }


def contracts_table() -> pd.DataFrame:
    rows = {}
    for s in INTRADAY:
        b = load_1h_clean(s)
        hv = float(np.sqrt((b["ret"] ** 2).ewm(span=vs.VOL_SPAN).mean()).iloc[-1])
        px = float(b["close"].iloc[-1])
        sig_1s = hv * np.sqrt(23)  # one-session 1-sigma move, fraction of price
        row = {"price": px, "session_sigma_pct": 100 * sig_1s}
        for r in RISKS:
            usd = r * CAPITAL
            row[f"full@{r:.2%}"] = usd / (sig_1s * px * POINT_VALUE[s])
            if s in MICRO:
                row[f"micro@{r:.2%}"] = usd / (sig_1s * px * MICRO[s][1])
        row["micro"] = MICRO.get(s, ("-", 0))[0]
        rows[s] = row
    return pd.DataFrame(rows).T


def main() -> None:
    d = daily_returns()
    rng = np.random.default_rng(2026)
    out = {"days": len(d), "daily_mean_at_0.25pct": float(d.mean()),
           "daily_sd_at_0.25pct": float(d.std()), "worst_day_at_0.25pct": float(d.min()),
           "hist_max_dd_at_0.25pct": float(
               ((1 + d).cumprod() / (1 + d).cumprod().cummax() - 1).min()
           ),
           "with_edge": {}, "demeaned": {}}
    base = d.to_numpy()
    for label, x in (("with_edge", base), ("demeaned", base - base.mean())):
        for r in RISKS:
            out[label][f"{r:.2%}"] = path_stats(paths(x * (r / t83.RISK), rng))
    tab = contracts_table()
    tab.to_csv(OUT / "challenge_contracts.csv")
    out["contracts"] = tab.round(2).to_dict("index")
    (OUT / "challenge_sizing.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
