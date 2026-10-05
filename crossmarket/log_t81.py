"""Append T81 (time-series momentum, 1h/4h entries) to out/test_log.xlsx with its verdict block."""

from __future__ import annotations

import json

from log_t80 import OUT, append
from quant.verdict import Verdict


def main() -> None:
    r = json.loads((OUT / "t81_results.json").read_text())
    h, f = r["1h"], r["4h"]
    ev = []
    for tf in ("1h", "4h"):
        v, b = r[tf], r[tf]["base"]
        lo, hi = b["sharpe_ci"]
        ev.append(
            f"{tf} entry: net Sharpe {b['sharpe']:+.2f} [95% {lo:+.1f},{hi:+.1f}], gross "
            f"{v['gross_sharpe']:+.2f}, 3x cost {v['cost3x']['sharpe']:+.2f}, one session late "
            f"{v['lag_session']['sharpe']:+.2f}; ann {b['ann_ret']:+.1%} at {b['ann_vol']:.1%} vol, "
            f"max DD {b['max_dd']:.1%}, t monthly {b['t_monthly']:+.2f} on {b['months']} months; "
            f"halves {v['halves'][0]:+.2f}/{v['halves'][1]:+.2f}; random-sign null p "
            f"{v['null_p']:.2f}"
        )
    pm = h["per_market_ann_ret"]
    ev += [
        "Per market (1h, ann. % of capital): "
        + ", ".join(f"{k} {x * 100:+.1f}" for k, x in pm.items())
        + ". Losses are ZB, ZN and NG; the four equity indices are one correlated gain "
        "(standing check 21).",
        "1h and 4h entries differ by 0.05 Sharpe: for a one-month hold, which intraday bar the "
        "position is entered on is immaterial -- the entry is not where this rule lives or dies.",
        "Calibration: NOT the 2.0 leakage case (the interval excludes it). Zero and the published "
        "0.3-0.8 both sit inside the interval, so this sample cannot say whether the pipeline "
        "is right.",
    ]
    v = Verdict.inconclusive(
        "T81 time-series momentum 12m/1m, 1h + 4h entries, 12 markets, train",
        reason=f"{h['base']['months']} monthly rebalances cannot calibrate a strategy whose "
        f"published Sharpe is 0.3-0.8: the 95% interval is {h['base']['sharpe_ci'][0]:+.1f} to "
        f"{h['base']['sharpe_ci'][1]:+.1f}. A true 0.5-Sharpe rule loses money in roughly one "
        "year in three.",
        evidence=ev,
        caveat="Signal and sizing read roll-cleaned daily Yahoo bars (one roll per scheduled "
        "window removed; NG's 12m return flips from +12% raw to -36% clean, so roll handling "
        "decides direction). Trading P&L is on roll-cleaned 1h/4h bars. Train window "
        f"{h['window'][0]}..{h['window'][1]} is a single year that trend followers widely "
        "reported as poor.",
        next_step="The momentum DIRECTION is not testable on intraday train alone; it can only "
        "serve as a regime filter for 4h/1h entry rules, each judged on its own entries.",
    )
    block = v.render()
    row = [
        "T81",
        "2026-10-05",
        "Time-series momentum replication: 12m lookback, 1m hold, inverse-vol, equal risk",
        "signal/vol: roll-cleaned Yahoo daily; P&L: roll-cleaned Yahoo 1h and 4h, 12 markets",
        f"intraday train, scored {h['window'][0]}..{h['window'][1]} ({h['base']['days']} days)",
        "entry at close of first 1h bar / first 4h bar of each month",
        f"1h SR {h['base']['sharpe']:+.2f}; 4h SR {f['base']['sharpe']:+.2f}",
        "1,7,16,17,20,21,22 applied; costs 0x/1x/3x; one-session-late entry",
        v.state,
        v.reason,
        v.next_step,
    ]
    append("T81", row, block)
    print(block)


if __name__ == "__main__":
    main()
