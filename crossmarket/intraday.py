"""Yahoo 1h -> clean 4h bars for the cross-market futures.

WHAT IS WRONG WITH YAHOO'S HOURLY `=F` SERIES (measured 2026-10-05 on the 2024-10 .. 2026-10 pull)

1. CONTRACT MIXING around rolls. For days at a time near an expiry, each hourly bar takes prices
   from BOTH the expiring and the next contract: the high-low range spans the calendar spread
   (ES 2025-12-16: 60-110 pt hourly ranges where 15-25 is normal) and closes alternate between
   the two contracts (NG 2025-09-24: 3.11, 2.84, 3.13, 3.16, 2.88 ...). Highs and lows are
   unusable there, and close-to-close returns carry fictional +-20 sigma moves.
2. ROLL GAPS. The switch to the next contract is a level shift nobody earned (ES 2024-12-20
   +25 sigma at the open).
3. SB is broken beyond repair at 1h: it alternates contracts for most of November 2025 with no
   expiry nearby. Excluded from the intraday set, as the handoff already advised.

POLICY

A bar is EXCLUDED when it falls in that market's scheduled roll window (calendar, known in
advance) AND either (a) its day shows contract mixing -- at least 3 hourly ranges, and 20% of the
day's bars, above 4x the trailing 20-day median range -- or (b) its return exceeds 6 robust
sigma. Real event days (2024-11-05, the April 2025 tariff week, CL March 2026) fall outside the
roll windows and are KEPT; excluding them would quietly delete the days trend following is
supposed to earn on.

An excluded bar contributes zero return. Signals are computed on the CLEAN INDEX, exp(cumsum of
non-excluded log returns) -- in effect a back-adjusted series -- and never on raw highs/lows.
Holding a position through an excluded stretch is charged one round turn (the roll).

A more aggressive variant (`strict=True`) excludes every range-flagged day whether or not it is
in a roll window. It exists for the sensitivity check, not as the default.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(os.environ.get("CROSSMARKET_DATA", pathlib.Path(__file__).parent / "data"))
INTRADAY = ["ES", "NQ", "RTY", "YM", "ZB", "ZN", "CL", "NG", "HG", "6E", "6B", "6J"]  # no SB
TZ = "America/New_York"
SESSION_START_H = 18  # CME Globex day opens 18:00 ET; a bar belongs to the session it opens in

# Proposed intraday split -- same dates for every market. NOT YET CONFIRMED, and so not written
# into quant.data_splits. Only TRAIN is used by anything in this directory.
TRAIN_END = "2025-10-01"
VALIDATE_END = "2026-04-01"


def _third_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    return d + pd.Timedelta(days=(weekday - d.weekday()) % 7 + 14)


def roll_window_days(sym: str, start: pd.Timestamp, end: pd.Timestamp) -> set:
    """Session dates on which this market's front contract can be rolling, from the calendar.

    Generous by design: a missed roll pollutes returns, an over-wide window only costs a few
    clean bars that were going to be kept anyway unless they also look broken.
    """
    days: set = set()
    years = range(start.year, end.year + 1)

    def add(a, b):
        days.update(pd.date_range(a, b, freq="D").date)

    for y in years:
        for m in range(1, 13):
            if sym in ("ES", "NQ", "RTY", "YM") and m in (3, 6, 9, 12):
                x = _third_weekday(y, m, 4)  # third Friday
                add(x - pd.Timedelta(days=7), x + pd.Timedelta(days=3))
            elif sym in ("6E", "6B", "6J") and m in (3, 6, 9, 12):
                x = _third_weekday(y, m, 2)  # third Wednesday
                add(x - pd.Timedelta(days=9), x + pd.Timedelta(days=3))
            elif sym in ("ZB", "ZN") and m in (2, 5, 8, 11):  # first notice ~ month end
                add(pd.Timestamp(y, m, 20), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                    + pd.Timedelta(days=7))
            elif sym == "HG" and m in (2, 4, 6, 8, 11):
                add(pd.Timestamp(y, m, 20), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                    + pd.Timedelta(days=7))
            elif sym == "CL":
                add(pd.Timestamp(y, m, 14), pd.Timestamp(y, m, 25))
            elif sym == "NG":
                add(pd.Timestamp(y, m, 19), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                    + pd.Timedelta(days=2))
    return days


def load_1h(sym: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{sym}_1h.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True).tz_convert(TZ)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    # the last row of a live pull is a partial bar stamped at the request time, not on the hour
    return d[(d.index.minute == 0) & (d.index.second == 0)]


def session_date(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx + pd.Timedelta(hours=24 - SESSION_START_H)).date


def flag_1h(d: pd.DataFrame, sym: str, *, strict: bool = False) -> pd.DataFrame:
    d = d.copy()
    r = np.log(d["close"]).diff()
    rt = r[d.index < pd.Timestamp(TRAIN_END, tz=TZ)]  # scale from train only: holdout untouched
    sd = 1.4826 * (rt - rt.median()).abs().median()
    rng = np.log(d["high"] / d["low"])
    base = rng.rolling(24 * 20, min_periods=200).median().shift(1)
    big = rng > 4 * base
    d["session"] = session_date(d.index)
    g = pd.DataFrame({"big": big, "s": d["session"]}).groupby("s")["big"].agg(["sum", "mean"])
    mixed_days = set(g.index[(g["sum"] >= 3) & (g["mean"] >= 0.2)])
    window = roll_window_days(sym, d.index[0].tz_localize(None), d.index[-1].tz_localize(None))
    in_win = d["session"].isin(window)
    mixed = d["session"].isin(mixed_days)
    spike = r.abs() > 6 * sd
    d["mixed_day"] = mixed
    d["excluded"] = (in_win & (mixed | spike)) | (mixed if strict else False)
    # the first bar AFTER an excluded run is measured from an excluded close, which may be the
    # other contract -- its return carries the level shift and is dropped too
    drop = d["excluded"] | d["excluded"].shift(1, fill_value=False)
    d["ret"] = r.where(~drop, 0.0).fillna(0.0)
    return d


def to_4h(d: pd.DataFrame) -> pd.DataFrame:
    """Session-anchored 4h bars: 18-22, 22-02, 02-06, 06-10, 10-14, 14-18 ET."""
    hrs_into = (d.index.hour - SESSION_START_H) % 24
    key = pd.Series(d["session"].astype(str), index=d.index) + "_" + (hrs_into // 4).astype(str)
    g = d.groupby(key.to_numpy(), sort=False)
    out = pd.DataFrame(
        {
            "ts": g.apply(lambda x: x.index[-1]),
            "close": g["close"].last(),
            "ret": g["ret"].sum(),  # clean log return: excluded hours contribute zero
            "excluded": g["excluded"].any(),
            "session": g["session"].first(),
        }
    ).set_index("ts").sort_index()
    out["clean_idx"] = np.exp(out["ret"].cumsum())
    return out


def load_4h(sym: str, *, strict: bool = False) -> pd.DataFrame:
    return to_4h(flag_1h(load_1h(sym), sym, strict=strict))
