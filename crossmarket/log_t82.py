"""Append T82 (monthly anchored VWAP bands + TEMA(9)/EMA(50), 1h) to out/test_log.xlsx."""

from __future__ import annotations

import json

from log_t80 import OUT, append
from quant.verdict import Verdict

CAND = "reenter 1sd | recent TEMAxEMA cross"


def main() -> None:
    r = json.loads((OUT / "t82_results.json").read_text())
    ev = r["B_events"]
    c = ev[CAND]

    def line(name):
        e = ev[name]
        return (
            f"{name}: n={int(e['events'])}, 4h {e['adj_4h']:+.3f} (t {e['t_4h']:+.2f}), "
            f"1 session {e['adj_1s']:+.3f} (t {e['t_1s']:+.2f}), 5 sessions {e['adj_5s']:+.3f}; "
            f"net of cost 1s {e['net_1s']:+.3f}; Holm p {e['p_holm']:.2f}"
        )

    zone = r["A_zone"]
    v = Verdict.no_edge(
        "T82 monthly anchored VWAP 1/2/3sd + TEMA(9)/EMA(50), 1h, 12 markets, train",
        reason=f"No event family survives Holm correction across {len(ev)} families at the "
        "pre-declared one-session horizon (smallest adjusted p "
        f"{min(e['p_holm'] for e in ev.values()):.2f}, on an 8-event cell). Crosses on their "
        "own -- TEMAxEMA, price x VWAP, TEMA x VWAP -- are measured precisely at zero.",
        evidence=[
            f"Units: forward move / (hourly vol * sqrt(h)), drift-adjusted per market, t "
            f"clustered by week; {r['scored_bars']} scored bars, {r['sessions']} sessions, "
            f"{r['window'][0]}..{r['window'][1]}.",
            "Zones (1 session, long): "
            + ", ".join(f"{k} {zone[k]['mean_1s']:+.3f}" for k in zone)
            + ". No zone is a buy or sell zone at 4h or 1 session; at 5 sessions price below "
            "VWAP drifts up and above +1sd drifts down (|t| <= 1.6) -- weak pull to the VWAP.",
            line("TEMA x EMA"),
            line("price x VWAP"),
            line("TEMA x VWAP"),
            line("reenter 1sd"),
            line("reenter 2sd"),
            line("breakout 1sd"),
            line("reenter 1sd | with TEMA>EMA trend"),
            line(CAND),
            f"Candidate ({CAND}): both halves positive ({c['half1']:+.3f}/{c['half2']:+.3f}), "
            f"buys {c['buy_adj']:+.3f} / sells {c['sell_adj']:+.3f}, 4h effect halves to "
            f"{c['adj_4h_ex_top3']:+.3f} without its best 3 sessions, session-permutation "
            f"control p {c['ctrl_p']:.3f}. Effect is gone by 5 sessions.",
            "2sd/3sd re-entry buys LOSE (falling knives in this window); 2sd 'with trend' cells "
            "have 8 and 19 events and collapse without their best sessions.",
        ],
        caveat="One year of Yahoo 1h, roll-cleaned; indicators in clean-index space. An "
        "information test: no exits simulated, so even the candidate is not a strategy.",
        next_step=f"Pre-register '{CAND}' alone -- entry, 4h and 1-session exits, costs -- and "
        "spend ONE validate look on it. Nothing else from this grid goes forward.",
    )
    block = v.render()
    row = [
        "T82",
        "2026-10-05",
        "Monthly anchored VWAP 1/2/3sd bands + TEMA(9)/EMA(50) on 1h: zones, crosses, confluence",
        "Yahoo =F 1h, 12 markets, roll-cleaned, clean-index indicators",
        f"intraday train {r['window'][0]}..{r['window'][1]}, first 2 sessions/month unscored",
        f"8 zones x trend state; {len(ev)} event families; horizons 4h / 1 session / 5 sessions",
        f"best: {CAND} 4h {c['adj_4h']:+.3f}sd (t {c['t_4h']:+.2f}), Holm p {c['p_holm']:.2f}",
        "1,2,3,4,7,9,11,12,17,19,20(week clusters),21 applied; Holm; cost per event",
        v.state,
        v.reason,
        v.next_step,
    ]
    append("T82", row, block)
    print(block)


if __name__ == "__main__":
    main()
