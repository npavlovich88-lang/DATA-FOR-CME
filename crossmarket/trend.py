"""Trend following on clean 4h bars: Donchian breakout and EMA crossover, inverse-vol sized.

Every parameter is fixed here, before any result is seen, and every config is reported -- the
grid is four cells, small enough that there is no "best cell" to cherry-pick (standing check 3).

Sizing: each market targets the same annualised volatility, TARGET_VOL / sqrt(n_markets), from
an EWMA of its own clean 4h returns, lagged one bar. This is constant DOLLAR risk per market
(standing check 16), not constant contracts. Weights are notional fractions of capital, i.e.
fractional contracts -- real contract rounding needs several million in capital at this target.

Execution: the signal at the close of bar t is traded at that close and earns bar t+1's return.
With hourly-continuous Globex trading the close of t and the open of t+1 are the same instant
except over the daily halt and the weekend. A two-bar delay is run as a sensitivity check.

Costs: quant.costs round turn per contract, half charged per side on every change in contracts,
and one full round turn on the position held through each excluded (roll) stretch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.costs import CostModel

BARS_PER_DAY = 6  # measured: median 4h bars per Globex session (intraday.load_4h)
ANN = 252 * BARS_PER_DAY
TARGET_VOL = 0.10
VOL_SPAN = 20 * BARS_PER_DAY

CONFIGS = {
    "breakout_20d": dict(kind="breakout", entry=20, exit=10),
    "breakout_55d": dict(kind="breakout", entry=55, exit=20),
    "ema_10_40d": dict(kind="ema", fast=10, slow=40),
    "ema_20_80d": dict(kind="ema", fast=20, slow=80),
}
WARMUP_DAYS = 80  # longest lookback in the grid; no config is scored before it has a signal


def breakout_signal(px: pd.Series, entry: int, exit: int) -> pd.Series:
    """Donchian on closes. Long on a close above the prior `entry`-day high close, flat on a
    close below the prior `exit`-day low; mirror for shorts.

    The rolling windows are shifted one bar so they EXCLUDE the bar being compared (standing
    check 9) -- otherwise a close can never exceed a max that already contains it.
    """
    ne, nx = entry * BARS_PER_DAY, exit * BARS_PER_DAY
    hi_e = px.rolling(ne, min_periods=ne).max().shift(1)
    lo_e = px.rolling(ne, min_periods=ne).min().shift(1)
    hi_x = px.rolling(nx, min_periods=nx).max().shift(1)
    lo_x = px.rolling(nx, min_periods=nx).min().shift(1)
    p = px.to_numpy()
    he, le, hx, lx = (s.to_numpy() for s in (hi_e, lo_e, hi_x, lo_x))
    pos = np.zeros(len(p))
    cur = 0.0
    for i in range(len(p)):
        if np.isnan(he[i]):
            pos[i] = 0.0
            continue
        if cur > 0 and p[i] < lx[i] or cur < 0 and p[i] > hx[i]:
            cur = 0.0
        if p[i] > he[i]:
            cur = 1.0
        elif p[i] < le[i]:
            cur = -1.0
        pos[i] = cur
    return pd.Series(pos, index=px.index)


def ema_signal(px: pd.Series, fast: int, slow: int) -> pd.Series:
    lp = np.log(px)
    f = lp.ewm(span=fast * BARS_PER_DAY, adjust=False).mean()
    s = lp.ewm(span=slow * BARS_PER_DAY, adjust=False).mean()
    sig = np.sign(f - s)
    sig.iloc[: slow * BARS_PER_DAY] = 0.0  # EWMA has not forgotten its seed yet (check 4)
    return sig


def signal(px: pd.Series, cfg: dict) -> pd.Series:
    if cfg["kind"] == "breakout":
        return breakout_signal(px, cfg["entry"], cfg["exit"])
    return ema_signal(px, cfg["fast"], cfg["slow"])


def market_pnl(
    b: pd.DataFrame, sym: str, cfg: dict, n_markets: int, *, cost_mult: float = 1.0, lag: int = 1
) -> pd.DataFrame:
    """Per-bar net return contribution of one market, as a fraction of capital."""
    vol = np.sqrt((b["ret"] ** 2).ewm(span=VOL_SPAN, min_periods=VOL_SPAN).mean() * ANN)
    w = signal(b["clean_idx"], cfg) * (TARGET_VOL / np.sqrt(n_markets)) / vol
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).shift(lag - 1)  # w[t] earns r[t+1]
    return pnl_from_weights(b, sym, w, cost_mult=cost_mult)


def pnl_from_weights(b: pd.DataFrame, sym: str, w: pd.Series, *, cost_mult: float = 1.0):
    """Net P&L of target weights `w` decided at each bar's close; w[t] earns bar t+1."""
    w = w.fillna(0.0)
    r = np.expm1(b["ret"])  # clean simple return
    held = w.shift(1).fillna(0.0)
    gross = held * r
    c = CostModel.for_reference_broker(sym)
    notional = b["close"].abs() * c.point_value
    per_side = c.round_turn_usd / 2 / notional * cost_mult  # fraction of notional per side
    trade_cost = (w - held).abs() * per_side.shift(1).fillna(per_side.iloc[0])
    # one round turn whenever a position is carried into an excluded stretch: the roll
    roll_start = b["excluded"] & ~b["excluded"].shift(1, fill_value=False)
    roll_cost = held.abs() * 2 * per_side * roll_start
    return pd.DataFrame(
        {"gross": gross, "cost": trade_cost + roll_cost, "w": held, "session": b["session"]}
    )


def portfolio(bars: dict, cfg: dict, **kw) -> tuple[pd.Series, pd.DataFrame]:
    """Daily (per-session) net portfolio return, and the per-market daily net contributions."""
    n = len(bars)
    per = {}
    for sym, b in bars.items():
        m = market_pnl(b, sym, cfg, n, **kw)
        per[sym] = (m["gross"] - m["cost"]).groupby(m["session"]).sum()
    per_df = pd.DataFrame(per).fillna(0.0)
    per_df.index = pd.to_datetime(per_df.index)
    return per_df.sum(axis=1), per_df


# ------------------------------------------------------------------------------------- stats
def sharpe(d: pd.Series) -> float:
    return float(d.mean() / d.std(ddof=1) * np.sqrt(252)) if d.std() > 0 else np.nan


def max_drawdown(d: pd.Series) -> float:
    eq = (1 + d).cumprod()
    return float((eq / eq.cummax() - 1).min())


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def block_bootstrap_sharpe(d: pd.Series, block: int = 10, n: int = 2000, seed: int = 0):
    """95% interval for the annualised Sharpe, circular block bootstrap (standing check 20)."""
    rng = np.random.default_rng(seed)
    x = d.to_numpy()
    k = int(np.ceil(len(x) / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(x), k)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % len(x)
        s = x[idx[: len(x)]]
        out[i] = s.mean() / s.std(ddof=1) * np.sqrt(252)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def summary(d: pd.Series) -> dict:
    m = (1 + d).groupby(d.index.to_period("M")).prod() - 1
    lo, hi = block_bootstrap_sharpe(d)
    top3 = d.nlargest(3).index
    return dict(
        days=len(d),
        ann_ret=float(d.mean() * 252),
        ann_vol=float(d.std() * np.sqrt(252)),
        sharpe=sharpe(d),
        sharpe_ci=(lo, hi),
        max_dd=max_drawdown(d),
        months=len(m),
        t_monthly=t_stat(m),
        t_daily=t_stat(d),
        sharpe_ex_top3=sharpe(d.drop(top3)),  # standing check 17
    )
