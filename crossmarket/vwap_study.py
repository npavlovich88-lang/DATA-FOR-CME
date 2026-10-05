"""Monthly anchored VWAP with 1/2/3 sigma bands, TEMA(9) and EMA(50) on clean 1h bars.

THE QUESTION (user, 2026-10-05): where relative to the monthly VWAP bands are the best buying and
selling opportunities, when do they occur, and do crosses between price, VWAP, TEMA(9) and
EMA(50) help?

EVERYTHING BELOW IS FIXED BEFORE ANY RESULT IS SEEN. The grid is reported in full; there is no
"best cell" without the family it was picked from (standing check 3).

INDICATORS -- computed in CLEAN-INDEX space, so a roll mid-month does not drag the VWAP across
two contracts. Every quantity here is scale-invariant, so the clean index (a back-adjusted
price) gives the same bands, crosses and z-scores as a correctly adjusted price would.
    typical price   (H + L + C) / 3, H and L mapped by each bar's clean/raw ratio
    VWAP            sum(v * tp) / sum(v), reset at the first session of each calendar month
    sigma           volume-weighted sd of tp around the running VWAP, same anchor
    z               (C - VWAP) / sigma; zones split at 0, +-1, +-2, +-3
    TEMA(9)         3*E1 - 3*E2 + E3 of close, span 9 bars
    EMA(50)         close, span 50 bars
Excluded (roll / contract-mix) bars get zero volume weight and never generate an event.
The first TWO sessions after each monthly reset are not scored: sigma from a handful of bars is
noise, and a band touch there is an artefact of the reset (standing check 4).

OUTCOME -- forward log return of the clean index over h = 4 bars, 1 session (23 bars) and
5 sessions (115 bars), divided by the market's trailing hourly vol * sqrt(h) measured AT the
event bar. Units are "sigmas of the horizon": 0.10 means a tenth of a typical h-bar move.
Signed in the trade direction for events (buy +, sell -).

STATISTICS
  * t clustered by calendar WEEK across all markets: twelve markets on one day share the
    calendar (checks 2, 19, 21), and 1- and 5-session forward windows overlap across days, so a
    session cluster would still overstate t.
  * drift-adjusted mean: minus each market's unconditional mean forward return over the same
    window, so a buy signal in a rising market does not get credit for the rise (check 1).
  * random-timing control: every event is moved to another session of the same month by ONE
    permutation shared across markets, keeping market, hour and direction, 300 times. Events
    that bunch together in calendar time stay bunched (checks 7 and 11).
  * Holm correction across every event family tested.
  * cost: one quant.costs round turn in the same sigma units, per event -- the bar a
    forecast must clear before it is a trade (the lesson that outranks the 23 checks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday import load_1h, flag_1h
from quant.costs import CostModel

H = {"4h": 4, "1s": 23, "5s": 115}
ZONES = [-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf]
ZONE_LABELS = ["<-3", "-3..-2", "-2..-1", "-1..0", "0..1", "1..2", "2..3", ">3"]
RESET_SKIP_SESSIONS = 2
RECENT = 6  # bars: "a cross happened recently" for confirmation variants
VOL_SPAN = 23 * 20


def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def tema(x: pd.Series, span: int) -> pd.Series:
    e1 = ema(x, span)
    e2 = ema(e1, span)
    return 3 * e1 - 3 * e2 + ema(e2, span)


def build(sym: str, train_end: str) -> pd.DataFrame:
    """One market's clean 1h bars with every indicator, the outcomes and the cost, train only."""
    raw = load_1h(sym)
    d = flag_1h(raw, sym)
    d = d[pd.to_datetime(d["session"]) < pd.Timestamp(train_end)].copy()  # cut by session
    ci = np.exp(d["ret"].cumsum())
    a = ci / d["close"]
    tp = (d["high"] * a + d["low"] * a + ci) / 3
    v = d["volume"].where(~d["excluded"], 0.0).fillna(0.0)
    month = pd.to_datetime(d["session"]).dt.to_period("M")
    g = month.to_numpy()
    cv = v.groupby(g).cumsum()
    vwap = (v * tp).groupby(g).cumsum() / cv
    ex2 = (v * tp**2).groupby(g).cumsum() / cv
    sigma = np.sqrt((ex2 - vwap**2).clip(lower=0))
    out = pd.DataFrame(index=d.index)
    out["sym"] = sym
    out["session"] = pd.to_datetime(d["session"])
    out["month"] = month
    out["hour"] = d.index.hour
    out["c"] = ci
    out["vwap"] = vwap
    out["z"] = (ci - vwap) / sigma
    out["tema"] = tema(ci, 9)
    out["ema"] = ema(ci, 50)
    out["excluded"] = d["excluded"]
    sess_no = out.groupby("month")["session"].transform(lambda s: s.rank(method="dense"))
    out["scored"] = (
        (sess_no > RESET_SKIP_SESSIONS) & ~out["excluded"] & np.isfinite(out["z"]) & (sigma > 0)
    )
    out.iloc[:300, out.columns.get_loc("scored")] = False  # EMA(50)/TEMA warm-up
    hvol = np.sqrt((d["ret"] ** 2).ewm(span=VOL_SPAN, min_periods=VOL_SPAN).mean())
    c = CostModel.for_reference_broker(sym)
    cost_frac = c.round_turn_usd / (d["close"] * c.point_value)
    lc = np.log(ci)
    for k, h in H.items():
        out[f"f_{k}"] = (lc.shift(-h) - lc) / (hvol * np.sqrt(h))  # NaN past the train end
        out[f"cost_{k}"] = cost_frac / (hvol * np.sqrt(h))
    out["zone"] = pd.cut(out["z"], ZONES, labels=ZONE_LABELS)
    return out


def events(b: pd.DataFrame) -> dict[str, pd.Series]:
    """Every event family, as a +1 (buy) / -1 (sell) / 0 series. Crosses compare bar t with
    bar t-1, both known at the close of t; nothing reads a bar after t."""
    z, z1 = b["z"], b["z"].shift(1)
    c, c1 = b["c"], b["c"].shift(1)
    vw, vw1 = b["vwap"], b["vwap"].shift(1)
    te, te1 = b["tema"], b["tema"].shift(1)
    em, em1 = b["ema"], b["ema"].shift(1)

    def cross(a, a1, x, x1):
        up = (a > x) & (a1 <= x1)
        dn = (a < x) & (a1 >= x1)
        return up.astype(int) - dn.astype(int)

    tema_ema = cross(te, te1, em, em1)
    px_vwap = cross(c, c1, vw, vw1)
    tema_vwap = cross(te, te1, vw, vw1)
    trend_up = te > em
    recent_up = (tema_ema == 1).astype(int).rolling(RECENT, min_periods=1).max().astype(bool)
    recent_dn = (tema_ema == -1).astype(int).rolling(RECENT, min_periods=1).max().astype(bool)

    ev = {
        "TEMA x EMA": tema_ema,
        "price x VWAP": px_vwap,
        "TEMA x VWAP": tema_vwap,
    }
    for k in (1, 2, 3):  # mean reversion: back inside the band
        buy = (z > -k) & (z1 <= -k)
        sell = (z < k) & (z1 >= k)
        ev[f"reenter {k}sd"] = buy.astype(int) - sell.astype(int)
    for k in (1, 2):  # breakout: through the band, with the move
        buy = (z > k) & (z1 <= k)
        sell = (z < -k) & (z1 >= -k)
        ev[f"breakout {k}sd"] = buy.astype(int) - sell.astype(int)
    # "are crosses needed": the same entries split by the TEMA/EMA trend state, and by a
    # TEMA x EMA cross in the same direction within the last RECENT bars
    for k in (1, 2):
        base = ev[f"reenter {k}sd"]
        with_t = ((base == 1) & trend_up) | ((base == -1) & ~trend_up)
        ev[f"reenter {k}sd | with TEMA>EMA trend"] = base.where(with_t, 0)
        ev[f"reenter {k}sd | against trend"] = base.where(~with_t & (base != 0), 0)
        rc = ((base == 1) & recent_up) | ((base == -1) & recent_dn)
        ev[f"reenter {k}sd | recent TEMAxEMA cross"] = base.where(rc, 0)
    base = px_vwap
    with_t = ((base == 1) & trend_up) | ((base == -1) & ~trend_up)
    ev["price x VWAP | with trend"] = base.where(with_t, 0)
    ev["price x VWAP | against trend"] = base.where(~with_t & (base != 0), 0)
    base = tema_ema
    on_side = ((base == 1) & (c > vw)) | ((base == -1) & (c < vw))
    ev["TEMA x EMA | price on VWAP side"] = base.where(on_side, 0)
    ev["TEMA x EMA | price off VWAP side"] = base.where(~on_side & (base != 0), 0)
    return {k: s.where(b["scored"], 0).fillna(0).astype(int) for k, s in ev.items()}


def clustered_t(x: pd.Series, cluster: pd.Series) -> tuple[float, int]:
    s = x.groupby(cluster.to_numpy()).sum()  # session SUM, not mean (check 19)
    cnt = x.groupby(cluster.to_numpy()).size()
    if len(s) < 3:
        return np.nan, len(s)
    # ratio estimator: total / total count, sd from cluster residuals
    mean = s.sum() / cnt.sum()
    resid = s - mean * cnt
    se = np.sqrt((resid**2).sum() * len(s) / (len(s) - 1)) / cnt.sum()
    return float(mean / se), len(s)


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    m = len(order)
    adj = (order * (m - np.arange(m))).cummax().clip(upper=1.0)
    return adj.reindex(p.index)
