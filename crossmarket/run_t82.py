"""T82: monthly anchored VWAP bands + TEMA(9)/EMA(50) on clean 1h bars -- information test.

PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t82.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm

import vwap_study as vs
from intraday import INTRADAY, TRAIN_END

OUT = pathlib.Path(__file__).parent / "out"
PRIMARY = "1s"  # declared before running: one-session horizon carries the Holm family
N_CTRL = 300
BUCKETS = {18: "18-22", 22: "22-02", 2: "02-06", 6: "06-10", 10: "10-14", 14: "14-18"}


def bucket(hour: pd.Series) -> pd.Series:
    start = ((hour - 18) % 24 // 4 * 4 + 18) % 24
    return start.map(BUCKETS)


def main() -> None:
    # integer index: timestamps repeat across markets, so they cannot be the key
    allb = pd.concat([vs.build(s, TRAIN_END) for s in INTRADAY]).reset_index(names="ts")
    allb["bucket"] = bucket(allb["hour"])
    sess_no = allb.groupby(["sym", "month"])["session"].rank(method="dense")
    allb["month_age"] = pd.cut(sess_no, [0, 7, 14, 99], labels=["d3-7", "d8-14", "d15+"])
    sc = allb[allb["scored"]].copy()
    for k in vs.H:
        sc[f"d_{k}"] = sc[f"f_{k}"] - sc.groupby("sym")[f"f_{k}"].transform("mean")
    # forward windows of 1 and 5 sessions overlap across days, so a session cluster is too
    # small; a calendar WEEK is the cluster for every t below
    sc["week"] = sc["session"].dt.to_period("W")
    mid = sc["session"].min() + (sc["session"].max() - sc["session"].min()) / 2
    res: dict = {
        "window": [str(sc["session"].min().date()), str(sc["session"].max().date())],
        "scored_bars": len(sc),
        "sessions": int(sc["session"].nunique()),
    }

    # ---------------------------------------------------------------- A: state tables (long)
    def state_table(by):
        rows = {}
        for key, g in sc.groupby(by, observed=True):
            r = {"bars": len(g)}
            for k in vs.H:
                x = g[f"d_{k}"].dropna()
                r[f"mean_{k}"] = float(x.mean())
                r[f"t_{k}"] = vs.clustered_t(x, g.loc[x.index, "week"])[0]
            rows[" | ".join(map(str, key)) if isinstance(key, tuple) else str(key)] = r
        return rows

    sc["trend"] = np.where(sc["tema"] > sc["ema"], "TEMA>EMA", "TEMA<EMA")
    sc["vwap_vs_ema"] = np.where(sc["vwap"] > sc["ema"], "VWAP>EMA", "VWAP<EMA")
    res["A_zone"] = state_table("zone")
    res["A_trend"] = state_table("trend")
    res["A_vwap_vs_ema"] = state_table("vwap_vs_ema")
    res["A_zone_x_trend"] = state_table(["zone", "trend"])

    # ---------------------------------------------------------------- B: event families
    ev_rows, per_bucket, per_age, per_market = {}, {}, {}, {}
    rng = np.random.default_rng(82)
    fin = sc[np.isfinite(sc[f"d_{PRIMARY}"])]
    lookup = dict(zip(zip(fin["sym"], fin["session"].to_numpy(), fin["hour"]), fin[f"d_{PRIMARY}"]))
    month_sessions = {m: np.array(sorted(g.unique())) for m, g in fin.groupby("month")["session"]}
    evs: dict[str, list[pd.Series]] = {}
    for _, b in allb.groupby("sym", sort=False):
        for name, sig in vs.events(b).items():
            evs.setdefault(name, []).append(sig[sig != 0])
    for name, parts in evs.items():
        dirs = pd.concat(parts)
        e = sc.loc[sc.index.intersection(dirs.index)].copy()
        if e.empty:
            continue
        e["dir"] = dirs.reindex(e.index)
        r = {
            "events": len(e),
            "sessions": int(e["session"].nunique()),
            "buys": int((e["dir"] > 0).sum()),
            "sells": int((e["dir"] < 0).sum()),
        }
        for k in vs.H:
            sraw = (e[f"f_{k}"] * e["dir"]).dropna()
            sadj = (e[f"d_{k}"] * e["dir"]).dropna()
            r[f"raw_{k}"] = float(sraw.mean())
            r[f"adj_{k}"] = float(sadj.mean())
            r[f"t_{k}"] = vs.clustered_t(sadj, e.loc[sadj.index, "week"])[0]
            r[f"cost_{k}"] = float(e[f"cost_{k}"].mean())
            r[f"net_{k}"] = r[f"raw_{k}"] - r[f"cost_{k}"]
        x = (e[f"d_{PRIMARY}"] * e["dir"]).dropna()
        ee = e.loc[x.index]
        r["buy_adj"] = float(x[ee["dir"] > 0].mean()) if (ee["dir"] > 0).any() else np.nan
        r["sell_adj"] = float(x[ee["dir"] < 0].mean()) if (ee["dir"] < 0).any() else np.nan
        for k in ("4h", PRIMARY):  # standing check 17: drop the three best SESSIONS (by sum)
            xk = (ee[f"d_{k}"] * ee["dir"]).dropna()
            top = xk.groupby(ee.loc[xk.index, "session"]).sum().nlargest(3).index
            r[f"adj_{k}_ex_top3"] = float(xk[~ee.loc[xk.index, "session"].isin(top)].mean())
        r["half1"] = float(x[ee["session"] < mid].mean())
        r["half2"] = float(x[ee["session"] >= mid].mean())
        # control: move every event to another session of the same month, by ONE permutation
        # shared across markets, keeping market and hour -- events that bunch together in
        # calendar time stay bunched, so the control has the same clustering as the signal
        ctrl = np.empty(N_CTRL)
        ev_m = ee["month"].to_numpy()
        ev_s = ee["session"].to_numpy()
        ev_key = list(zip(ee["sym"], ee["hour"]))
        dirs_arr = ee["dir"].to_numpy()
        for i in range(N_CTRL):
            perm = {}
            for m, ss in month_sessions.items():
                perm.update(zip(ss, rng.permutation(ss)))
            vals = np.array(
                [lookup.get((k[0], perm[s_], k[1]), np.nan) for k, s_ in zip(ev_key, ev_s)]
            )
            ok = np.isfinite(vals)
            ctrl[i] = np.mean(vals[ok] * dirs_arr[ok]) if ok.any() else np.nan
        del ev_m
        r["ctrl_p"] = float((ctrl[np.isfinite(ctrl)] >= x.mean()).mean())
        r["p_primary"] = float(2 * norm.sf(abs(r[f"t_{PRIMARY}"])))
        ev_rows[name] = r
        per_bucket[name] = x.groupby(ee["bucket"]).agg(["mean", "size"]).round(4).to_dict("index")
        per_age[name] = (
            x.groupby(ee["month_age"], observed=True)
            .agg(["mean", "size"])
            .round(4)
            .to_dict("index")
        )
        per_market[name] = x.groupby(ee["sym"]).mean().round(3).to_dict()
    ev = pd.DataFrame(ev_rows).T
    ev["p_holm"] = vs.holm(ev["p_primary"])
    res["B_events"] = ev.to_dict("index")
    res["B_by_bucket"] = per_bucket
    res["B_by_month_age"] = per_age
    res["B_by_market"] = per_market
    OUT.mkdir(exist_ok=True)
    (OUT / "t82_results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
