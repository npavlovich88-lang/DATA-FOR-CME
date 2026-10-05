"""Daily cross-market strategies on 26 years of Yahoo `=F` data (T87 calibration, T88 ideas).

DATA: roll-cleaned daily bars (intraday.load_daily_clean: one roll per scheduled window
removed, bad closes treated as missing). 13 markets -- the 12 intraday ones plus SB, whose
DAILY series is research grade. RTY starts 2017 and so has no train data.

FIXED BEFORE ANY RESULT (pre-registered in preregistered/T87_T88.json):

  S1 tsmom_12m      sign of the 12-month return at each month end, held one month
                    (Moskowitz-Ooi-Pedersen; T87, the calibration -- published Sharpe 0.3-0.8)
  S2 ema200_side    long above the 200-day EMA, short below, daily
  S3 ema50_200      long when EMA50 > EMA200, short below, daily
  S4 ema200_mom     S2 and the 252-day return must agree, else flat, daily
  S5 qvwap_trend    quarterly anchored VWAP 1sd re-entry, only in the direction of the
                    200-day EMA side; hold 10 days
  S6 yvwap_trend    the same with a YEARLY anchor; hold 10 days
  S7 z20_pullback   own calculation: z = (close - SMA20) / SD20; buy when z crosses back above
                    -1.5 with close > EMA200 and 252-day return > 0; sell mirror; hold 10 days

  Sizing: every market targets 10%/sqrt(12) annualised vol from an EWMA of daily squared
  returns (centre of mass 60, as MOP), lagged one day -- equal risk per market (check 16).
  State strategies (S1-S4) carry that weight while the signal holds; event strategies (S5-S7)
  fix it at entry. Signal at the close of day t earns day t+1. quant.costs per side on every
  change, one round turn per roll held through.

  All prices for signals are the CLEAN INDEX (back-adjusted), never raw Yahoo closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import trend
from intraday import load_daily_clean

DAILY = ["ES", "NQ", "RTY", "YM", "ZB", "ZN", "CL", "NG", "HG", "6E", "6B", "6J", "SB"]
TARGET_VOL = 0.10
N_FOR_SIZING = 12
HOLD = 10
STRATS = (
    "tsmom_12m",
    "ema200_side",
    "ema50_200",
    "ema200_mom",
    "qvwap_trend",
    "yvwap_trend",
    "z20_pullback",
)


def load(sym: str) -> pd.DataFrame:
    d = load_daily_clean(sym)
    raw = pd.read_csv(trend_data_path(sym), index_col=0)
    raw.index = (
        pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York").tz_localize(None)
    ).normalize()
    raw = raw[~raw.index.duplicated(keep="last")].reindex(d.index)
    a = d["clean_idx"] / d["close"]
    out = pd.DataFrame(
        {
            "close": d["close"],
            "ret": d["ret"],
            "clean_idx": d["clean_idx"],
            "excluded": d["roll"],  # trend.pnl_from_weights charges a round turn here
            "session": d.index,
            "tp": ((raw["high"] * a + raw["low"] * a + d["clean_idx"]) / 3).fillna(d["clean_idx"]),
            "volume": raw["volume"].fillna(0.0),
        },
        index=d.index,
    )
    return out


def trend_data_path(sym: str):
    from intraday import DATA

    return DATA / f"{sym}_1d.csv"


def vol_weight(b: pd.DataFrame) -> pd.Series:
    sig = np.sqrt((b["ret"] ** 2).ewm(com=60, min_periods=60).mean() * 261)
    return (TARGET_VOL / np.sqrt(N_FOR_SIZING) / sig).replace([np.inf, -np.inf], np.nan)


def anchored_vwap_z(b: pd.DataFrame, freq: str) -> pd.Series:
    """z of the clean close against a VWAP anchored at each calendar period. Periods with any
    zero-volume day (early FX and HG files carry none) fall back to equal weights."""
    per = b.index.to_period(freq)
    v = b["volume"].copy()
    has_zero = (v <= 0).groupby(per).transform("any")
    v[has_zero] = 1.0
    g = per.to_numpy()
    cv = v.groupby(g).cumsum()
    vw = (v * b["tp"]).groupby(g).cumsum() / cv
    ex2 = (v * b["tp"] ** 2).groupby(g).cumsum() / cv
    sd = np.sqrt((ex2 - vw**2).clip(lower=0))
    z = (b["clean_idx"] - vw) / sd
    age = pd.Series(1, index=b.index).groupby(g).cumsum()
    return z.where((age > 5) & (sd > 0))  # first five days of a period: bands still forming


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def target_weights(b: pd.DataFrame, strat: str) -> pd.Series:
    """Target weight at each day's close."""
    px = b["clean_idx"]
    w = vol_weight(b)
    e200 = ema(px, 200)
    side = np.sign(px - e200).where(e200.notna(), 0.0)
    mom = np.log(px).diff(252)
    if strat == "tsmom_12m":
        me = b.groupby(b.index.to_period("M")).tail(1).index
        lr = np.log(px.loc[me])
        sig = np.sign(lr - lr.shift(12))
        wm = (sig * w.loc[me]).reindex(b.index).ffill()  # decided at month end, held a month
        return wm.fillna(0.0)
    if strat == "ema200_side":
        return (side * w).fillna(0.0)
    if strat == "ema50_200":
        e50 = ema(px, 50)
        return (np.sign(e50 - e200) * w).fillna(0.0)
    if strat == "ema200_mom":
        agree = np.where(np.sign(mom) == side, side, 0.0)
        return (pd.Series(agree, index=b.index) * w).fillna(0.0)
    if strat in ("qvwap_trend", "yvwap_trend"):
        z = anchored_vwap_z(b, "Q" if strat == "qvwap_trend" else "Y")
        z1 = z.shift(1)
        ev = ((z > -1) & (z1 <= -1) & (side > 0)).astype(int) - (
            (z < 1) & (z1 >= 1) & (side < 0)
        ).astype(int)
        return events_to_weights(ev, w)
    if strat == "z20_pullback":
        m20, s20 = px.rolling(20).mean(), px.rolling(20).std()
        z = (px - m20) / s20
        z1 = z.shift(1)
        ev = ((z > -1.5) & (z1 <= -1.5) & (side > 0) & (mom > 0)).astype(int) - (
            (z < 1.5) & (z1 >= 1.5) & (side < 0) & (mom < 0)
        ).astype(int)
        return events_to_weights(ev, w)
    raise KeyError(strat)


def events_to_weights(ev: pd.Series, w: pd.Series) -> pd.Series:
    """One position per market, fixed size from entry, held HOLD days, signals while open
    ignored (check 18: statistics on trades actually taken)."""
    e = ev.fillna(0).to_numpy()
    ww = w.to_numpy()
    out = np.zeros(len(e))
    i = 0
    while i < len(e):
        if e[i] != 0 and np.isfinite(ww[i]):
            end = min(i + HOLD, len(e))
            out[i:end] = e[i] * ww[i]
            i = end + 1
        else:
            i += 1
    return pd.Series(out, index=ev.index)


def run(bars: dict, strat: str, *, cost_mult: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    per = {}
    for s, b in bars.items():
        if len(b) < 300:
            continue
        m = trend.pnl_from_weights(b, s, target_weights(b, strat), cost_mult=cost_mult)
        per[s] = m["gross"] - m["cost"]
    df = pd.DataFrame(per).fillna(0.0)
    return df.sum(axis=1), df


def permute_months(bars: dict, rng) -> dict:
    """Null for every strategy: whole calendar months reshuffled, the SAME order in every market
    (keeps within-month paths, vol clustering inside a month and cross-market correlation;
    destroys the month-to-month persistence trend and momentum rules need). The rule is then
    re-run on the shuffled clean index (check 7)."""
    months = sorted(set().union(*(set(b.index.to_period("M")) for b in bars.values())))
    order = dict(zip(months, rng.permutation(len(months)), strict=True))
    out = {}
    for s, b in bars.items():
        key = b.index.to_period("M").map(order)
        nb = b.iloc[np.argsort(key.to_numpy(), kind="stable")].copy()
        a = (nb["tp"] / nb["clean_idx"]).to_numpy()
        nb.index = b.index  # keep the calendar; the returns now arrive in shuffled months
        nb["session"] = b.index
        nb["clean_idx"] = np.exp(nb["ret"].cumsum())
        nb["tp"] = nb["clean_idx"] * a
        out[s] = nb
    return out
