"""Append T83 (W/M/Q anchored VWAP band signals, traded as a 12-market portfolio) to the log."""

from __future__ import annotations

import json

from log_t80 import OUT, append
from quant.verdict import Verdict

CAND = "reenter 1sd | recent TEMAxEMA cross"


def main() -> None:
    r = json.loads((OUT / "t83_results.json").read_text())
    best = max(r, key=lambda k: r[k]["sharpe"])
    b = r[best]
    fam_min = min(v["family_p_minp"] for v in r.values())
    best_p = min(r, key=lambda k: r[k]["family_p_minp"])
    neg = sum(v["sharpe"] < 0 for v in r.values())

    def line(k):
        v = r[k]
        lo, hi = v["sharpe_ci"]
        return (
            f"{k}: SR {v['sharpe']:+.2f} [{lo:+.1f},{hi:+.1f}], {v['trades']} trades "
            f"({v['trades_per_day']:.1f}/day, {v['concurrent']:.1f} open), "
            f"{v['trade_mean_R']:+.3f}R/trade net, max DD {v['max_dd']:.1%}, halves "
            f"{v['halves'][0]:+.2f}/{v['halves'][1]:+.2f}, timing-null p "
            f"{v['null_p_plus1']:.3f} (null median SR {v['null_median']:+.2f}), family p "
            f"min-p {v['family_p_minp']:.2f} / max-SR {v['family_p']:.2f}"
        )

    cand = [k for k in r if CAND in k]
    others = [k for k in r if CAND not in k]
    ev = [line(k) for k in sorted(r, key=lambda k: -r[k]["sharpe"])]
    ev.append(
        f"{neg} of {len(r)} cells lose money after costs. The plain band rules (re-entry 1sd, "
        "2sd, breakout) lose at every anchor except one breakout cell: "
        + ", ".join(f"{k.split(' | ')[0]} {k.split(' | ')[1]} {k.split(' | ')[2]} "
                    f"{r[k]['sharpe']:+.2f}" for k in others if r[k]["sharpe"] > 0)
        + "."
    )
    ev.append(
        "The timing null is not centred on zero: random entries lose to costs (null median SR "
        f"{min(v['null_median'] for v in r.values()):+.2f} to "
        f"{max(v['null_median'] for v in r.values()):+.2f}); breakout cells keep the month's "
        "drift. Hence the Westfall-Young min-p family test, not best raw Sharpe."
    )
    ev.append(
        "Spreading across markets: average pairwise correlation of per-market daily P&L is "
        f"{min(v['avg_pair_corr'] for v in r.values()):+.2f} to "
        f"{max(v['avg_pair_corr'] for v in r.values()):+.2f}, so the portfolio does diversify; "
        "but diversifying a zero-mean signal gives a zero-mean portfolio with more costs. More "
        "markets multiply trades, not edge."
    )
    ev.append(
        f"The {CAND} rule is positive at all three anchors ("
        + ", ".join(f"{k.split(' | ')[0]} {k.split(' | ')[2]} {r[k]['sharpe']:+.2f}" for k in cand)
        + "). The M cells are in-sample -- that rule was picked from T82 on this same data -- so "
        "only W and Q count as new, and they share the same bars and calendar."
    )
    if fam_min < 0.05:
        state = Verdict.signal_only
        reason = (
            f"{best_p} survives the best-of-{len(r)} timing null (Westfall-Young family p "
            f"{r[best_p]['family_p_minp']:.2f}) on one year of train. A traded result on train, "
            "picked from a rule T82 selected on the same data -- not confirmed anywhere."
        )
    else:
        state = Verdict.no_edge
        reason = (
            f"No cell survives the best-of-{len(r)} random-timing null (smallest Westfall-Young "
            f"family p {fam_min:.2f}, {best_p}); the best Sharpe, {best}, is "
            f"{b['sharpe']:+.2f} with a 95% interval of {b['sharpe_ci'][0]:+.1f} to "
            f"{b['sharpe_ci'][1]:+.1f}."
        )
    v = state(
        "T83 W/M/Q anchored VWAP bands, 4 rules x 2 holds, 12-market portfolio, train",
        reason=reason,
        evidence=ev,
        caveat="One year of Yahoo 1h, roll-cleaned; fixed holds, no stops; constant risk "
        "0.25% of capital per one-session sigma per trade; fractional contracts.",
        next_step=f"If anything goes forward it is '{CAND}' alone, pre-registered at one anchor "
        "and one hold chosen NOW from W/Q only, for ONE validate look. The plain band rules are "
        "finished.",
    )
    block = v.render()
    row = [
        "T83",
        "2026-10-05",
        "Weekly/monthly/quarterly anchored VWAP bands, traded as a 12-market signal portfolio",
        "Yahoo =F 1h, 12 markets, roll-cleaned, clean-index indicators",
        "intraday train, first 1/2/5 sessions after each W/M/Q reset unscored",
        "3 anchors x 4 rules x 2 holds (4h, 1 session) = 24 cells; one position per market",
        f"best: {best} SR {b['sharpe']:+.2f}, family p min-p {b['family_p_minp']:.2f}",
        "7,11,12,16,17,18,20,21 applied; costs and rolls in P&L; Westfall-Young timing null",
        v.state,
        v.reason,
        v.next_step,
    ]
    append("T83", row, block)
    print(block)


if __name__ == "__main__":
    main()
