"""T80: trend following (Donchian breakout + EMA crossover) on clean 4h bars, TRAIN ONLY.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t80.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import trend
from intraday import INTRADAY, TRAIN_END, load_4h

OUT = pathlib.Path(__file__).parent / "out"
N_PERM = 200


def train_bars(strict: bool = False) -> dict:
    bars = {}
    for s in INTRADAY:
        b = load_4h(s, strict=strict)
        # cut by SESSION: the evening of 2025-09-30 opens the 2025-10-01 (validate) session
        bars[s] = b[pd.to_datetime(b["session"]) < pd.Timestamp(TRAIN_END)]
    return bars


def score_window(d: pd.Series) -> pd.Series:
    return d.iloc[trend.WARMUP_DAYS :]


def permuted(bars: dict, rng) -> dict:
    """Shuffle whole sessions, SAME order in every market: keeps each day's intraday path,
    each market's return distribution and the cross-market correlation; destroys only the
    time ordering that a trend rule needs (standing check 7: the null re-runs the rule)."""
    sessions = sorted(set().union(*(set(b["session"]) for b in bars.values())))
    order = rng.permutation(len(sessions))
    remap = {sessions[i]: sessions[j] for i, j in enumerate(order)}
    out = {}
    for s, b in bars.items():
        pieces = [g for _, g in b.groupby(b["session"].map(remap), sort=True)]
        rets = np.concatenate([g["ret"].to_numpy() for g in pieces])
        nb = b.iloc[: len(rets)].copy()
        nb["ret"] = rets
        nb["excluded"] = np.concatenate([g["excluded"].to_numpy() for g in pieces])
        nb["clean_idx"] = np.exp(np.cumsum(rets))
        out[s] = nb
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bars = train_bars()
    res = {}
    for name, cfg in trend.CONFIGS.items():
        d, per = trend.portfolio(bars, cfg)
        d, per = score_window(d), per.iloc[trend.WARMUP_DAYS :]
        half = len(d) // 2
        res[name] = dict(
            base=trend.summary(d),
            cost3x=trend.summary(score_window(trend.portfolio(bars, cfg, cost_mult=3.0)[0])),
            lag2=trend.summary(score_window(trend.portfolio(bars, cfg, lag=2)[0])),
            halves=(trend.sharpe(d.iloc[:half]), trend.sharpe(d.iloc[half:])),
            per_market_ann_ret=(per.mean() * 252).round(4).to_dict(),
            window=(str(d.index[0].date()), str(d.index[-1].date())),
        )
    strict = train_bars(strict=True)
    for name, cfg in trend.CONFIGS.items():
        res[name]["strict"] = trend.summary(score_window(trend.portfolio(strict, cfg)[0]))

    rng = np.random.default_rng(80)
    null = {k: [] for k in trend.CONFIGS}
    for _ in range(N_PERM):
        pb = permuted(bars, rng)
        for name, cfg in trend.CONFIGS.items():
            null[name].append(trend.sharpe(score_window(trend.portfolio(pb, cfg)[0])))
    null_df = pd.DataFrame(null)
    fam_max = null_df.max(axis=1)
    for name in trend.CONFIGS:
        sr = res[name]["base"]["sharpe"]
        res[name]["perm_p"] = float((null_df[name] >= sr).mean())
        res[name]["perm_p_family"] = float((fam_max >= sr).mean())  # best-of-4 under the null
    (OUT / "t80_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
