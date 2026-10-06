"""Append T87 (daily momentum calibration) and T88 (daily ideas) to out/test_log.xlsx."""

from __future__ import annotations

import json

from log_t80 import OUT, append
from quant.verdict import Verdict


def fmt(v):
    lo, hi = v["sharpe_ci"]
    return f"Sharpe {v['sharpe']:+.2f} [95% {lo:+.2f},{hi:+.2f}]"


def main() -> None:
    r = json.loads((OUT / "t87_t88_results.json").read_text())
    tr, va = r["train"], r["validate"]
    t = tr["tsmom_12m"]
    v87 = Verdict.edge(
        "T87 daily time-series momentum 12m/1m, 13 markets -- CALIBRATION",
        config="12m lookback, 1m hold, inverse-vol equal risk (as published)",
        net=f"train {t['sharpe']:+.2f}, validate {va['tsmom_12m']['sharpe']:+.2f}",
        reason=f"Pipeline WORKS: train {fmt(t)} sits inside the published 0.3-0.8 band -- "
        "not ~2 (leakage) and not ~0 (broken). Validate "
        f"{va['tsmom_12m']['sharpe']:+.2f}.",
        evidence=[
            f"Train 2002-2014: ann {t['ann_ret']:+.1%} at {t['ann_vol']:.1%} vol, max DD "
            f"{t['max_dd']:.1%}, t monthly {t['t_monthly']:+.2f} on {t['months']} months, "
            f"3x cost {t['cost3x_sharpe']:+.2f}; blocks "
            + ", ".join(f"{k} {x:+.2f}" for k, x in t["blocks"].items()),
            f"Month-shuffle null p {t['null_p']:.3f}; family min-p {t['family_p_minp']:.2f}.",
            f"Validate 2015-2020: {fmt(va['tsmom_12m'])}; by year "
            + ", ".join(f"{k} {x:+.2f}" for k, x in va["tsmom_12m"]["by_year"].items()),
        ],
        caveat="A replication of a published effect, run as a pipeline check; it was never a "
        "discovery.",
        next_step="The daily pipeline can be trusted for T88.",
    )
    append("T87", ["T87", "2026-10-05", "Daily TSMOM replication (calibration)",
                   "Yahoo =F daily, 13 markets, roll-cleaned", "train 2002-2014 + validate look",
                   "S1 as published", f"train {t['sharpe']:+.2f} / validate "
                   f"{va['tsmom_12m']['sharpe']:+.2f}", "1,7,16,17,20,21,22; 3x cost; blocks",
                   v87.state, v87.reason, v87.next_step], v87.render())

    w = r["t88_winner"]
    ev = [f"{k}: train {fmt(x)}, 3x cost {x['cost3x_sharpe']:+.2f}, blocks "
          + "/".join(f"{b:+.2f}" for b in x["blocks"].values())
          + f", null p {x['null_p']:.3f}, family p {x['family_p_minp']:.2f}"
          for k, x in tr.items() if k != "tsmom_12m"]
    vw = va[w]
    ev.append(f"Validate {w}: {fmt(vw)}, ann {vw['ann_ret']:+.1%}, max DD {vw['max_dd']:.1%}; "
              "by year " + ", ".join(f"{k} {x:+.2f}" for k, x in vw["by_year"].items()))
    ev.append("The VWAP-band ideas (quarterly and yearly) LOSE on daily; the own-calculation "
              "z20 pullback is positive but not distinguishable from the null.")
    v88 = Verdict.edge(
        "T88 daily trend/momentum/band ideas, 13 markets",
        config=f"{w}: long when close > EMA200 AND 252-day return > 0, short when both < 0, "
        "else flat; inverse-vol equal risk; daily",
        net=f"train {tr[w]['sharpe']:+.2f}, validate {vw['sharpe']:+.2f}",
        reason=f"{w} has the best train Sharpe ({tr[w]['sharpe']:+.2f}), survives the "
        f"family-wise month-shuffle null (min-p {tr[w]['family_p_minp']:.2f}), is positive in "
        f"all three train blocks, and repeats on validate ({vw['sharpe']:+.2f}).",
        evidence=ev,
        caveat="Daily holdout (2021-) untouched. A 0.6-Sharpe trend rule draws down 20-26% at a "
        "10% vol target and has losing years (2016). It is a slow edge: in any one month it is "
        "mostly noise.",
        next_step="One holdout look before real money; size for the drawdowns, not the Sharpe.",
    )
    append("T88", ["T88", "2026-10-05", "Daily ideas: EMA200, EMA50/200, EMA200+momentum, "
                   "Q/Y VWAP band re-entry with trend, own z20 pullback",
                   "Yahoo =F daily, 13 markets", "train 2002-2014 + validate look (winner)",
                   "6 strategies (S2-S7)", f"winner {w}: train {tr[w]['sharpe']:+.2f} / "
                   f"validate {vw['sharpe']:+.2f}", "1,3,7,12,16,17,20,21,22; family min-p",
                   v88.state, v88.reason, v88.next_step], v88.render())
    print(v87.render())
    print(v88.render())


if __name__ == "__main__":
    main()
