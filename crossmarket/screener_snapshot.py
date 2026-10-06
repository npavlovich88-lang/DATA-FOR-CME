"""What the TradingView screener should show on the last closed hourly bar, from the research
pipeline (Yahoo, roll-cleaned). Use it to sanity-check the Pine table; small differences are
expected (TradingView's own data, volume and back-adjustment).

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/screener_snapshot.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import vwap_study as vs
from intraday import INTRADAY, load_1h_clean
from quant.costs import POINT_VALUE

BAND, RECENT, NEAR, SKIP = 1.0, 6, 0.25, 5
ACCOUNT, RISK = 999_000, 0.015


def snapshot() -> pd.DataFrame:
    rows = {}
    for s in INTRADAY:
        ind = vs.build(s, "2100-01-01", anchor="Q")
        b = load_1h_clean(s)
        t, e = ind["tema"], ind["ema"]
        up = (t > e) & (t.shift(1) <= e.shift(1))
        dn = (t < e) & (t.shift(1) >= e.shift(1))
        pos = np.arange(len(ind))
        last_up = pd.Series(np.where(up, pos, np.nan)).ffill().to_numpy()
        last_dn = pd.Series(np.where(dn, pos, np.nan)).ffill().to_numpy()
        i = len(ind) - 1
        z, z1 = ind["z"].iloc[i], ind["z"].iloc[i - 1]
        bsu, bsd = i - last_up[i], i - last_dn[i]
        au, ad = bsu < RECENT, bsd < RECENT
        sess_no = ind.groupby("period")["session"].transform(lambda x: x.rank(method="dense"))
        warm = sess_no.iloc[i] <= SKIP
        ev = vs.events(ind.reset_index(drop=True))["reenter 1sd | recent TEMAxEMA cross"]
        if warm:
            st = "warm-up"
        elif ev.iloc[i] == 1:
            st = "SIGNAL BUY"
        elif ev.iloc[i] == -1:
            st = "SIGNAL SELL"
        elif z <= -(BAND - NEAR):
            st = "WATCH BUY armed" if au else "WATCH BUY"
        elif z >= BAND - NEAR:
            st = "WATCH SELL armed" if ad else "WATCH SELL"
        else:
            st = "-"
        hv = np.sqrt((b["ret"] ** 2).ewm(span=460).mean()).iloc[-1]
        px = b["close"].iloc[-1]
        rows[s] = {
            "bar (ET)": ind.index[i].strftime("%Y-%m-%d %H:%M"),
            "z": round(float(z), 2),
            "z prev": round(float(z1), 2),
            "trend": "TEMA>EMA" if t.iloc[i] > e.iloc[i] else "TEMA<EMA",
            "x-up": int(bsu),
            "x-dn": int(bsd),
            "status": st,
            "size @1.5%": round(
                float(ACCOUNT * RISK / (hv * np.sqrt(23) * px * POINT_VALUE[s])), 1
            ),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    print(snapshot().to_string())
