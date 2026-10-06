"""Competition plan for the T88 rule (EMA200 + 12-month momentum), daily, on $999,000.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/challenge_daily_plan.py

Risk level: the rule's own daily P&L, train + validate only (2002-2020; the daily holdout is not
read), at challenge costs, block-bootstrapped into 18-session paths (Oct 6 -> Oct 30), with the
mean set to zero AND with it kept. Scale = multiple of the research size (10%/sqrt(12) vol per
market). Brakes simulated: halve size at -10% from start, flat at -15%.
Today's directions and contract counts use the latest bars for the SIGNAL and SIZE only.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import daily
import trend
from quant.costs import POINT_VALUE
from quant.data_splits import CM_DAILY_VALIDATE_END

OUT = pathlib.Path(__file__).parent / "out"
CAPITAL, SESSIONS, N_PATHS, BLOCK = 999_000, 18, 20_000, 5
SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
MICRO = {
    "ES": ("MES", 5.0),
    "NQ": ("MNQ", 2.0),
    "RTY": ("M2K", 5.0),
    "YM": ("MYM", 0.5),
    "CL": ("MCL", 100.0),
    "NG": ("MNG", 1000.0),
    "HG": ("MHG", 2500.0),
    "6E": ("M6E", 12500.0),
    "6B": ("M6B", 6250.0),
    "6J": ("MJY", 1250000.0),
}


def daily_pnl() -> pd.Series:
    bars = {s: daily.load(s) for s in daily.DAILY}
    bars = {s: b[b.index < pd.Timestamp(CM_DAILY_VALIDATE_END)] for s, b in bars.items()}
    per = {}
    for s, b in bars.items():
        if len(b) < 300:
            continue
        m = trend.pnl_from_weights(
            b, s, daily.target_weights(b, "ema200_mom"), commission_all_in=2.50
        )
        per[s] = m["gross"] - m["cost"]
    d = pd.DataFrame(per).fillna(0.0).sum(axis=1)
    return d[d.index >= "2002-01-01"]


def sim(unit, k, rng):
    nb = int(np.ceil(SESSIONS / BLOCK))
    st = rng.integers(0, len(unit), (N_PATHS, nb))
    idx = (st[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_PATHS, -1)
    r = unit[idx[:, :SESSIONS] % len(unit)] * k
    eq, low, worst = np.ones(N_PATHS), np.ones(N_PATHS), np.zeros(N_PATHS)
    for t in range(SESSIONS):
        sc = np.where(eq <= 0.85, 0.0, np.where(eq <= 0.90, 0.5, 1.0))
        day = r[:, t] * sc
        worst = np.minimum(worst, day)
        eq *= 1 + day
        low = np.minimum(low, eq)
    return {
        "p10": float(np.quantile(eq, 0.1) - 1),
        "median": float(np.median(eq) - 1),
        "p90": float(np.quantile(eq, 0.9) - 1),
        "p_minus20": float((low <= 0.8).mean()),
        "p_minus10": float((low <= 0.9).mean()),
        "worst_day_1in1000": float(np.quantile(worst, 0.001)),
    }


def today(k: float) -> pd.DataFrame:
    rows = {}
    for s in daily.DAILY:
        b = daily.load(s)
        w = daily.target_weights(b, "ema200_mom").iloc[-1] * k
        px = float(b["close"].iloc[-1])
        sig = float(np.sqrt((b["ret"] ** 2).ewm(com=60).mean().iloc[-1] * 261))
        e200 = float(daily.ema(b["clean_idx"], 200).iloc[-1])
        mom = float(np.log(b["clean_idx"]).diff(252).iloc[-1])
        n_full = w * CAPITAL / (px * POINT_VALUE[s])
        rows[s] = {
            "as of": str(b.index[-1].date()),
            "above EMA200": bool(b["clean_idx"].iloc[-1] > e200),
            "12m return": round(mom * 100, 1),
            "direction": "LONG" if w > 0 else "SHORT" if w < 0 else "FLAT",
            "contracts": round(float(n_full), 1),
            "micro": MICRO.get(s, ("-", None))[0],
            "micro contracts": round(float(w * CAPITAL / (px * MICRO[s][1])), 0)
            if s in MICRO
            else None,
            "notional $": round(float(abs(w) * CAPITAL)),
            "annual vol %": round(sig * 100, 1),
        }
    return pd.DataFrame(rows).T


def main() -> None:
    d = daily_pnl()
    rng = np.random.default_rng(1006)
    res = {"days": len(d), "sharpe_2002_2020": trend.sharpe(d), "with_edge": {}, "zero_edge": {}}
    for lab, x in (("with_edge", d.to_numpy()), ("zero_edge", (d - d.mean()).to_numpy())):
        for k in SCALES:
            res[lab][str(k)] = sim(x, k, rng)
    ok = [
        k
        for k in SCALES
        if res["zero_edge"][str(k)]["p_minus20"] <= 0.01
        and res["zero_edge"][str(k)]["worst_day_1in1000"] > -0.10
    ]
    k = max(ok)
    res["chosen_scale"] = k
    t = today(k)
    t.to_csv(OUT / "challenge_today.csv")
    res["today"] = t.to_dict("index")
    (OUT / "challenge_daily_plan.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({x: res[x] for x in ("days", "sharpe_2002_2020", "chosen_scale")}))
    print(pd.DataFrame(res["zero_edge"]).T.round(3).to_string())
    print(pd.DataFrame(res["with_edge"]).T.round(3).to_string())
    print(t.to_string())


if __name__ == "__main__":
    main()
