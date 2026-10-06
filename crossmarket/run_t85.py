"""T85: optimisation of the VWAP-band re-entry idea on TRAIN, with overfitting controls.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t85.py

GRID, FIXED BEFORE RUNNING (624 cells)
    anchor        W, M, Q
    distance      re-entry level k = 0.5, 1.0, 1.5, 2.0 sd (buy: z crosses back above -k;
                  sell: z crosses back below +k)
    confirmation  none, or a fast-over-slow cross in the trade direction within the last
                  3, 6 or 12 bars, for four fast/slow pairs:
                  TEMA9/EMA50 (T82-T84), EMA9/EMA21, TEMA20/EMA100, EMA20/EMA50
    hold          4, 12, 23, 46 bars
Everything else exactly as T83/T84: one position per market, constant risk, costs, rolls.

SELECTION: highest train Sharpe at reference costs. CONTROLS, all on train only:
    * PBO by CSCV (quant.pbo, S=16): how often the in-sample winner lands below the median
      out of sample across the 12,870 half/half splits of the train days.
    * expected best Sharpe of N noise strategies: sqrt(2 ln N) / sqrt(years), the bar the
      winner must clear before its Sharpe means anything (an upper bound for correlated
      cells, which are fewer effective trials).
The train winner then gets ONE validate look (look #2 on this idea, logged).
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

import run_t83 as t83
import run_t84 as t84
import trend
import vwap_study as vs
from intraday import INTRADAY, TRAIN_END, VALIDATE_END, load_1h_clean
from quant.pbo import cscv

OUT = pathlib.Path(__file__).parent / "out"
ANCHORS = ("W", "M", "Q")
KS = (0.5, 1.0, 1.5, 2.0)
PAIRS = {
    "TEMA9/EMA50": ("tema", 9, 50),
    "EMA9/EMA21": ("ema", 9, 21),
    "TEMA20/EMA100": ("tema", 20, 100),
    "EMA20/EMA50": ("ema", 20, 50),
}
RECENT = (3, 6, 12)
HOLDS = (4, 12, 23, 46)


def gen_events(ind: pd.DataFrame, k: float, conf) -> np.ndarray:
    z, z1 = ind["z"], ind["z"].shift(1)
    ev = ((z > -k) & (z1 <= -k)).astype(int) - ((z < k) & (z1 >= k)).astype(int)
    if conf is not None:
        pair, n = conf
        kind, f, sl = PAIRS[pair]
        fast = vs.tema(ind["c"], f) if kind == "tema" else vs.ema(ind["c"], f)
        slow = vs.ema(ind["c"], sl)
        up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        ru = up.astype(int).rolling(n, min_periods=1).max().astype(bool)
        rd = dn.astype(int).rolling(n, min_periods=1).max().astype(bool)
        ev = ev.where(((ev == 1) & ru) | ((ev == -1) & rd), 0)
    return (ev.where(ind["scored"], 0).fillna(0).astype(int)).to_numpy()


def load(anchor: str, end: str, score_from: str | None = None):
    markets, inds = {}, {}
    for s in INTRADAY:
        b = load_1h_clean(s)
        b = b[pd.to_datetime(b["session"]) < pd.Timestamp(end)]
        ind = vs.build(s, end, anchor=anchor).reset_index(drop=True)
        if score_from is not None:
            ind["scored"] = ind["scored"] & (ind["session"] >= pd.Timestamp(score_from))
        hvol = np.sqrt((b["ret"] ** 2).ewm(span=vs.VOL_SPAN, min_periods=vs.VOL_SPAN).mean())
        markets[s] = {"bars": b, "size": (t83.RISK / (hvol * np.sqrt(23))).to_numpy()}
        inds[s] = ind
    return markets, inds


def configs():
    confs = [None] + [(p, n) for p in PAIRS for n in RECENT]
    return list(itertools.product(ANCHORS, KS, confs, HOLDS))


def name(cfg) -> str:
    a, k, conf, h = cfg
    c = "no cross" if conf is None else f"{conf[0]} within {conf[1]}"
    return f"{a} | {k}sd | {c} | hold {h}"


def main() -> None:
    cols, rows = {}, {}
    data = {a: load(a, TRAIN_END) for a in ANCHORS}
    for cfg in configs():
        a, k, conf, h = cfg
        markets, inds = data[a]
        events = {s: {"x": gen_events(inds[s], k, conf)} for s in INTRADAY}
        c = t83.cell(markets, events, "x", h)
        n = name(cfg)
        cols[n] = c["port"]
        st = trend.summary(c["port"])
        rows[n] = dict(
            sharpe=st["sharpe"],
            ann_ret=st["ann_ret"],
            max_dd=st["max_dd"],
            trades=len(c["trades"]),
            trade_mean_R=float(c["trades"]["net"].mean() / t83.RISK)
            if len(c["trades"])
            else np.nan,
        )
    M = pd.DataFrame(cols).fillna(0.0)
    tab = pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)
    pbo = cscv(M.to_numpy(), S=16)
    years = len(M) / 252
    n_cells = M.shape[1]
    bar = float(np.sqrt(2 * np.log(n_cells)) / np.sqrt(years))
    winner = tab.index[0]
    wcfg = next(c for c in configs() if name(c) == winner)

    # ---- ONE validate look for the train winner
    a, k, conf, h = wcfg
    vm, vi = load(a, VALIDATE_END, score_from=TRAIN_END)
    vev = {s: {"x": gen_events(vi[s], k, conf)} for s in INTRADAY}
    vc = t84.validate_only(t83.cell(vm, vev, "x", h), vm)
    vstat = t83.stats(vc)
    with (OUT / "looks.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "date": str(dt.date.today()),
                    "test": "T85",
                    "segment": "validate",
                    "rule": winner,
                    "look": 2,
                }
            )
            + "\n"
        )

    def marg(col):  # aggregate by factor before the best cell (check 3)
        parts = tab.index.to_series().str.split(" \\| ", expand=True)
        return tab.groupby(parts[col].to_numpy())["sharpe"].agg(["mean", "median", "max", "size"])

    res = {
        "cells": n_cells,
        "train_days": len(M),
        "noise_bar_sharpe": bar,
        "pbo": float(pbo["pbo"]),
        "pbo_detail": {
            k_: (float(v) if np.isscalar(v) else None)
            for k_, v in pbo.items()
            if k_ not in ("lambda", "is_oos", "winners", "oos_all")
        },
        "top20": tab.head(20).round(4).to_dict("index"),
        "share_positive": float((tab["sharpe"] > 0).mean()),
        "by_anchor": marg(0).round(3).to_dict("index"),
        "by_k": marg(1).round(3).to_dict("index"),
        "by_conf": marg(2).round(3).to_dict("index"),
        "by_hold": marg(3).round(3).to_dict("index"),
        "prereg_cell_train": tab.loc["Q | 1.0sd | TEMA9/EMA50 within 6 | hold 23"].to_dict(),
        "prereg_cell_rank": int(list(tab.index).index("Q | 1.0sd | TEMA9/EMA50 within 6 | hold 23"))
        + 1,
        "winner": winner,
        "winner_validate": vstat,
    }
    tab.to_csv(OUT / "t85_grid.csv")
    (OUT / "t85_results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
