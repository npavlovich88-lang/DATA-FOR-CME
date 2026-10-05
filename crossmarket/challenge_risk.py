"""Risk level for the one-month CME University Trading Challenge, after the holdout failed.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/challenge_risk.py

THE PREMISE CHANGED WITH T86. The rule did not hold up on the holdout, so the sizing can no
longer assume an edge. Every path below is drawn from the rule's real daily P&L over all two
years (train + validate + holdout, challenge costs) with its mean set to ZERO: it keeps the
real day-to-day volatility, fat days and clustering, and assumes the rule makes nothing on
average. Costs are still in it (each trade pays $2.50/side and the spread), so the zero-mean
paths are slightly OPTIMISTIC about a strategy that, at worst, is pure cost.

"AGGRESSIVE BUT SOMEWHAT SAFE", made precise:
    safe        P(account ever 20% below its start within the 20 sessions) <= 1%, and the
                1-in-1000 worst single day below 10% (half the 20% one-day lockout)
    aggressive  the largest risk per trade that stays safe -- it maximises the spread of
                outcomes, which is what a top-5 finish in a ranking by final balance needs

BRAKES (fixed rules a person can follow, simulated here):
    at -10% from the starting balance   halve the risk per trade
    at -15%                             stop opening new trades for the rest of the event
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import challenge_sizing as cs
import run_t83 as t83
import run_t86 as t86

OUT = pathlib.Path(__file__).parent / "out"
RISKS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
SESSIONS, N_PATHS, BLOCK = 20, 20_000, 5


def all_daily() -> pd.Series:
    d = cs.daily_returns()  # train + validate, challenge costs, at t83.RISK
    hm, he = t86.load_holdout()
    h = t86.holdout_only(t83.cell(hm, he, t86.RULE, t86.HOLD, commission_all_in=2.50))["port"]
    return pd.concat([d, h]).sort_index()


def simulate(unit: np.ndarray, risk: float, brakes: bool, rng) -> dict:
    k = int(np.ceil(SESSIONS / BLOCK))
    starts = rng.integers(0, len(unit), (N_PATHS, k))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_PATHS, -1)
    r = unit[idx[:, :SESSIONS] % len(unit)] * (risk / t83.RISK)
    eq = np.ones(N_PATHS)
    low = np.ones(N_PATHS)
    worst_day = np.zeros(N_PATHS)
    for t in range(SESSIONS):
        scale = np.ones(N_PATHS)
        if brakes:
            scale = np.where(eq <= 0.90, 0.5, 1.0)
            scale = np.where(eq <= 0.85, 0.0, scale)
        day = r[:, t] * scale
        worst_day = np.minimum(worst_day, day)
        eq = eq * (1 + day)
        low = np.minimum(low, eq)
    return {
        "median_final": float(np.median(eq) - 1),
        "p10_final": float(np.quantile(eq, 0.10) - 1),
        "p90_final": float(np.quantile(eq, 0.90) - 1),
        "p99_final": float(np.quantile(eq, 0.99) - 1),
        "prob_below_minus20": float((low <= 0.80).mean()),
        "prob_below_minus10": float((low <= 0.90).mean()),
        "p999_worst_day": float(np.quantile(worst_day, 0.001)),
    }


def main() -> None:
    d = all_daily()
    unit = (d - d.mean()).to_numpy()  # zero edge
    rng = np.random.default_rng(1030)
    res = {"days": len(d), "mean_daily_at_0.25pct": float(d.mean()),
           "sd_daily_at_0.25pct": float(d.std()), "results": {}}
    for brakes in (False, True):
        for r in RISKS:
            res["results"][f"{r:.1%} | brakes {'on' if brakes else 'off'}"] = simulate(
                unit, r, brakes, rng
            )
    ok = [
        (float(k.split("%")[0]) / 100, k)
        for k, v in res["results"].items()
        if "on" in k and v["prob_below_minus20"] <= 0.01 and v["p999_worst_day"] > -0.10
    ]
    res["recommended"] = max(ok)[1] if ok else None
    (OUT / "challenge_risk.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
