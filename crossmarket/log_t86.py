"""Append T86 (holdout look, pre-registered T84 rule) to out/test_log.xlsx."""

from __future__ import annotations

import json

from log_t80 import OUT, append
from quant.verdict import Verdict


def main() -> None:
    r = json.loads((OUT / "t86_results.json").read_text())
    v = r["reference"]
    lo, hi = v["sharpe_ci"]
    verdict = Verdict.no_edge(
        "T86 pre-registered Q-VWAP 1sd re-entry + TEMA9xEMA50, HOLDOUT (only look)",
        reason=f"The rule loses on the holdout: Sharpe {v['sharpe']:+.2f} [95% {lo:+.2f},"
        f"{hi:+.2f}], {v['trade_mean_R']:+.3f}R per trade over {v['trades']} trades, timing "
        f"null p {r['null_p']:.2f}. Train +1.80, validate +1.43, holdout {v['sharpe']:+.2f}: "
        "the decay pattern of a fitted rule, not a stable edge.",
        evidence=[
            f"Window {r['window'][0]}..{r['window'][1]}, {v['days']} sessions; challenge costs "
            f"Sharpe {r['challenge']['sharpe']:+.2f}.",
            f"Max DD {v['max_dd']:.2%}; halves {v['halves'][0]:+.2f}/{v['halves'][1]:+.2f}; "
            f"monthly " + ", ".join(f"{k} {x:+.2%}" for k, x in r["monthly"].items()) + ".",
            "Per market (R): " + ", ".join(f"{k} {x:+.1f}" for k, x in v["per_market_R"].items()),
            f"It still beats random entry timing (null median Sharpe {r['null_median']:+.2f}): "
            "random entries pay costs for nothing, this rule roughly breaks even.",
        ],
        caveat="The holdout is now spent for this rule family. Any re-tuned variant can no "
        "longer be tested on Yahoo intraday data -- it would need new data (live forward).",
        next_step="Treat the rule as unproven. If it is traded at all, size it for zero edge "
        "and judge it only on fresh live data.",
    )
    append("T86", [
        "T86", "2026-10-05", "HOLDOUT: T84 rule unchanged (preregistered/T86.json)",
        "Yahoo =F 1h, 12 markets", f"holdout {r['window'][0]}..{r['window'][1]}, look 1 of 1",
        "single pre-registered rule",
        f"SR {v['sharpe']:+.2f}, {v['trade_mean_R']:+.3f}R/trade, null p {r['null_p']:.2f}",
        "7,11,16,17,18,20,21; costs reference + challenge", verdict.state, verdict.reason,
        verdict.next_step], verdict.render())
    print(verdict.render())


if __name__ == "__main__":
    main()
