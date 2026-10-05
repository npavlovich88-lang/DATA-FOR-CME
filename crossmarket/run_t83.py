"""T83: weekly / monthly / quarterly anchored VWAP band signals, traded as a 12-market portfolio.

    PYTHONPATH=<quant repo>/src:crossmarket python crossmarket/run_t83.py

THE IDEA (user, 2026-10-05): watch every market at once; whichever one signals gets traded, so
the edge is spread across markets instead of waiting on one.

FIXED BEFORE ANY RESULT IS SEEN
    anchors    weekly (W), monthly (M), quarterly (Q) VWAP with 1/2/3 sigma bands
    rules      four, the ones T82 showed matter:
                 reenter 1sd                          back inside +-1 sigma (reversion)
                 reenter 2sd                          back inside +-2 sigma
                 breakout 1sd                         through +-1 sigma, with the move
                 reenter 1sd | recent TEMAxEMA cross  T82's candidate (in-sample at M!)
    exits      fixed hold: 4 bars (4h) or 23 bars (one session); no stops, no targets
    trading    any market that signals while flat in that market is entered at the signal
               bar's close; one position per market; no pyramiding; signals while a position
               is open are ignored, so every statistic is on trades actually taken (check 18)
    sizing     constant dollar risk: each trade sized so a one-session 1-sigma move is RISK of
               capital, from the market's trailing hourly vol at entry (check 16)
    P&L        marked every bar on clean 1h bars, quant.costs per side, roll cost on any
               trade held into an excluded stretch -- the same engine as T80/T81
    controls   every cell re-simulated with its entries moved to random sessions of the same
               month by one permutation shared across markets, keeping market and hour, so
               signals that bunch in calendar time stay bunched (checks 7, 11); family p two
               ways: best Sharpe of all 24 cells under the null, and Westfall-Young min-p
               (each cell against its own null, then best of 24)
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pandas as pd

import trend
import vwap_study as vs
from intraday import INTRADAY, TRAIN_END, load_1h_clean

OUT = pathlib.Path(__file__).parent / "out"
ANCHORS = ("W", "M", "Q")
RULES = ("reenter 1sd", "reenter 2sd", "breakout 1sd", "reenter 1sd | recent TEMAxEMA cross")
HOLDS = {"4h": 4, "1s": 23}
RISK = 0.0025  # one-session 1-sigma move per trade, fraction of capital
N_NULL = int(os.environ.get("T83_NULL", "600"))


def simulate(ev: np.ndarray, size: np.ndarray, hold: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a +-1 event array into target weights and trade labels. w[i] is decided at the
    close of bar i and earns bar i+1, so a trade entered at i holds w over i..i+hold-1 and is
    flat again from the close of i+hold. Labels cover i..i+hold, where its costs land."""
    n = len(ev)
    w = np.zeros(n)
    lab = np.full(n, -1)
    i, k = 0, 0
    nz = np.flatnonzero(ev)
    j = 0
    while j < len(nz):
        i = nz[j]
        if not np.isfinite(size[i]):
            j += 1
            continue
        end = min(i + hold, n - 1)
        w[i:end] = ev[i] * size[i]
        lab[i : end + 1] = k
        k += 1
        j = np.searchsorted(nz, end + 1)  # next entry strictly after the exit bar
    return w, lab


def cell(markets: dict, events: dict, rule: str, hold: int) -> dict:
    daily, trades, active = {}, [], []
    for s, m in markets.items():
        ev = events[s][rule]
        w, lab = simulate(ev, m["size"], hold)
        b = m["bars"]
        p = trend.pnl_from_weights(b, s, pd.Series(w, index=b.index))
        net = (p["gross"] - p["cost"]).to_numpy()
        daily[s] = pd.Series(net, index=b.index).groupby(b["session"].to_numpy()).sum()
        active.append(pd.Series(w != 0, index=b.index))
        if (lab >= 0).any():
            t = pd.DataFrame({"lab": lab, "net": net, "session": b["session"].to_numpy()})
            t = t[t["lab"] >= 0].groupby("lab").agg(net=("net", "sum"), session=("session", "first"))
            t["sym"] = s
            trades.append(t)
    per = pd.DataFrame(daily).fillna(0.0)
    per.index = pd.to_datetime(per.index)
    port = per.sum(axis=1)
    tr = pd.concat(trades) if trades else pd.DataFrame(columns=["net", "session", "sym"])
    conc = pd.concat(active, axis=1, sort=True).fillna(False).sum(axis=1)
    return {"port": port, "per": per, "trades": tr, "concurrent": float(conc.mean())}


def permuted_events(markets: dict, events: dict, rng) -> dict:
    """Move every event to another session of the same calendar month, one permutation shared
    by all markets, same hour, same direction; dropped if that bar does not exist or is not
    scored."""
    sessions = sorted(set().union(*(set(m["key"]["session"]) for m in markets.values())))
    by_month: dict = {}
    for ss in sessions:
        by_month.setdefault(pd.Timestamp(ss).to_period("M"), []).append(ss)
    perm = {}
    for ss in by_month.values():
        perm.update(zip(ss, rng.permutation(ss), strict=True))
    out = {}
    for s, m in markets.items():
        key = m["key"]
        pos = m["pos"]
        new_sess = key["session"].map(perm)
        tgt = [pos.get((a, h), -1) for a, h in zip(new_sess, key["hour"], strict=True)]
        tgt = np.asarray(tgt)
        out[s] = {}
        for rule, ev in events[s].items():
            nz = np.flatnonzero(ev)
            e2 = np.zeros_like(ev)
            t = tgt[nz]
            ok = (t >= 0) & m["scored"][np.clip(t, 0, None)]
            e2[t[ok]] = ev[nz[ok]]
            out[s][rule] = e2
        del pos
    return out


def stats(c: dict) -> dict:
    d = c["port"]
    tr = c["trades"]
    half = len(d) // 2
    week = pd.to_datetime(tr["session"]).dt.to_period("W") if len(tr) else None
    r_units = tr["net"] / RISK if len(tr) else pd.Series(dtype=float)
    out = trend.summary(d)
    out.update(
        trades=len(tr),
        trades_per_day=len(tr) / max(len(d), 1),
        concurrent=c["concurrent"],
        trade_mean_R=float(r_units.mean()) if len(tr) else np.nan,
        trade_t=vs.clustered_t(r_units, week)[0] if len(tr) > 3 else np.nan,
        win_rate=float((tr["net"] > 0).mean()) if len(tr) else np.nan,
        halves=(trend.sharpe(d.iloc[:half]), trend.sharpe(d.iloc[half:])),
        per_market_R={k: round(float(v), 3) for k, v in (tr.groupby("sym")["net"].sum() / RISK).items()},
        mean_market_sharpe=float(np.nanmean([trend.sharpe(c["per"][k]) for k in c["per"]])),
        avg_pair_corr=float(
            c["per"].corr().where(~np.eye(c["per"].shape[1], dtype=bool)).stack().mean()
        ),
    )
    return out


def main() -> None:
    res: dict = {}
    null_rows = []
    cells: dict = {}
    data = {}
    for a in ANCHORS:
        markets, events = {}, {}
        for s in INTRADAY:
            b = load_1h_clean(s)
            b = b[pd.to_datetime(b["session"]) < pd.Timestamp(TRAIN_END)]  # train, by session
            ind = vs.build(s, TRAIN_END, anchor=a)
            assert ind.index.equals(b.index)
            hvol = np.sqrt((b["ret"] ** 2).ewm(span=vs.VOL_SPAN, min_periods=vs.VOL_SPAN).mean())
            size = (RISK / (hvol * np.sqrt(23))).to_numpy()
            key = pd.DataFrame({"session": ind["session"].to_numpy(), "hour": ind["hour"].to_numpy()})
            markets[s] = {
                "bars": b,
                "size": size,
                "key": key,
                "pos": {(ss, h): i for i, (ss, h) in enumerate(zip(key["session"], key["hour"], strict=True))},
                "scored": ind["scored"].to_numpy(),
            }
            ev_all = vs.events(ind.reset_index(drop=True))
            events[s] = {r: ev_all[r].to_numpy() for r in RULES}
        data[a] = (markets, events)
        for rule in RULES:
            for hk, h in HOLDS.items():
                c = cell(markets, events, rule, h)
                name = f"{a} | {rule} | hold {hk}"
                cells[name] = (a, rule, h)
                res[name] = stats(c)
                print(f"{name:<52} SR {res[name]['sharpe']:+.2f} trades {res[name]['trades']}")

    rng = np.random.default_rng(83)
    for _ in range(N_NULL):
        row = {}
        # one permutation of sessions per null draw, shared by every anchor, rule and hold
        state = rng.bit_generator.state
        for a in ANCHORS:
            markets, events = data[a]
            rng.bit_generator.state = state
            pev = permuted_events(markets, events, rng)
            for rule in RULES:
                for hk, h in HOLDS.items():
                    row[f"{a} | {rule} | hold {hk}"] = trend.sharpe(cell(markets, pev, rule, h)["port"])
        null_rows.append(row)
    null = pd.DataFrame(null_rows)
    null.to_csv(OUT / "t83_null.csv", index=False)
    fam = null.max(axis=1)
    # Westfall-Young min-p: each cell judged against ITS OWN null, then the family adjusts for
    # the best of the 24. Needed because the timing null is not centred on zero: random entries
    # lose to costs in most cells but keep month drift in the breakout cells, so a max-Sharpe
    # family test is decided by whichever cells have the highest baseline, not the best signal.
    ranks = null.rank(ascending=False, method="max")  # 1 = the best draw in its column
    null_p = ranks / (len(null) + 1)
    null_minp = null_p.min(axis=1)
    for name in res:
        sr = res[name]["sharpe"]
        res[name]["null_p"] = float((null[name] >= sr).mean())
        res[name]["family_p"] = float((fam >= sr).mean())
        res[name]["null_median"] = float(null[name].median())
        p_obs = (1 + (null[name] >= sr).sum()) / (len(null) + 1)
        res[name]["null_p_plus1"] = float(p_obs)
        res[name]["family_p_minp"] = float((null_minp <= p_obs).mean())
    OUT.mkdir(exist_ok=True)
    (OUT / "t83_results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
