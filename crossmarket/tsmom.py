"""Time-series momentum (Moskowitz, Ooi & Pedersen 2012) with 1h / 4h entries.

Rule, exactly as published and fixed before any result:
    signal   sign of the trailing 12-month return, measured at each month end
    hold     one calendar month
    size     inverse ex-ante volatility, equal risk per market: weight = (TARGET_VOL/sqrt(N))
             / sigma, sigma from an EWMA of daily squared returns, centre of mass 60 days, as MOP

The 12-month lookback and sigma read the roll-cleaned DAILY series up to the last session of the
month. Those bars are strictly in the past of every traded bar, which is why the lookback may
reach back before the intraday train start. Entry is at the close of the FIRST 1h or 4h bar of
the month's first session (opens 18:00 ET, an hour after the daily close the signal used). All
P&L runs on the clean intraday bars, with roll exclusion and quant.costs as in trend.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend import TARGET_VOL, pnl_from_weights

LOOKBACK_MONTHS = 12
VOL_COM = 60


def monthly_targets(daily: pd.DataFrame, n_markets: int) -> pd.DataFrame:
    """One row per month M: the weight to hold during M, decided at the end of M-1."""
    me = daily.groupby(daily.index.to_period("M")).tail(1)  # last session of each month
    lr = np.log(me["clean_idx"])
    ret12 = lr - lr.shift(LOOKBACK_MONTHS)
    sig = np.sqrt((daily["ret"] ** 2).ewm(com=VOL_COM, min_periods=VOL_COM).mean() * 261)
    sigma = sig.reindex(me.index)
    w = np.sign(ret12) * (TARGET_VOL / np.sqrt(n_markets)) / sigma
    out = pd.DataFrame({"ret12": ret12, "sigma": sigma, "w": w})
    out.index = me.index.to_period("M") + 1  # decided at end of M-1, held during M
    return out


def bar_weights(b: pd.DataFrame, targets: pd.DataFrame, lag_bars: int = 0) -> pd.Series:
    """Target weight at each bar's close. Every bar of month M targets w[M], so the first bar
    of M is held at w[M-1] and switches at its close -- entry at the close of the first bar."""
    month = pd.to_datetime(b["session"]).dt.to_period("M")
    w = month.map(targets["w"]).astype(float)
    w.index = b.index
    return w.shift(lag_bars) if lag_bars else w


def run(bars: dict, dailies: dict, *, cost_mult: float = 1.0, lag_bars: int = 0):
    """Per-session net returns per market, and the gross returns each market would have earned
    LONG at the same size (used for the random-sign null)."""
    n = len(bars)
    net, long_gross = {}, {}
    for s, b in bars.items():
        t = monthly_targets(dailies[s], n)
        w = bar_weights(b, t, lag_bars)
        m = pnl_from_weights(b, s, w, cost_mult=cost_mult)
        net[s] = (m["gross"] - m["cost"]).groupby(m["session"]).sum()
        lg = m["gross"] * np.sign(m["w"]).replace(0, np.nan)  # |w| * r: the unsigned bet
        long_gross[s] = lg.fillna(0.0).groupby(m["session"]).sum()
    net, long_gross = pd.DataFrame(net).fillna(0.0), pd.DataFrame(long_gross).fillna(0.0)
    net.index = long_gross.index = pd.to_datetime(net.index)
    return net, long_gross


def random_sign_null(long_gross: pd.DataFrame, n: int = 5000, seed: int = 81) -> np.ndarray:
    """Sharpe of the same sized positions with a random direction per market-month (standing
    check 7: the null keeps the rule's sizing and monthly holding and randomises only the
    momentum call). Gross of costs; compare against the strategy's gross Sharpe."""
    rng = np.random.default_rng(seed)
    months = long_gross.index.to_period("M")
    codes, uniq = pd.factorize(months)
    x = long_gross.to_numpy()
    out = np.empty(n)
    for i in range(n):
        sg = rng.choice([-1.0, 1.0], size=(len(uniq), x.shape[1]))
        d = (x * sg[codes]).sum(axis=1)
        out[i] = d.mean() / d.std(ddof=1) * np.sqrt(252)
    return out
