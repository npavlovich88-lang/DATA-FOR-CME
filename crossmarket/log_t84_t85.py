"""Append T84 (validate look, pre-registered quarterly rule) and T85 (train optimisation, its
winner's validate look) to out/test_log.xlsx."""

from __future__ import annotations

import json

from log_t80 import OUT, append
from quant.verdict import Verdict


def t84() -> None:
    r = json.loads((OUT / "t84_results.json").read_text())
    v, c = r["reference"], r["challenge"]
    lo, hi = v["sharpe_ci"]
    ev = [
        f"Validate {r['prereg']['segment']}: Sharpe {v['sharpe']:+.2f} [95% {lo:+.2f},{hi:+.2f}], "
        f"ann {v['ann_ret']:+.2%} at {v['ann_vol']:.2%} vol, max DD {v['max_dd']:.2%}, "
        f"{v['trades']} trades, {v['trade_mean_R']:+.3f}R/trade (t {v['trade_t']:+.2f}), win "
        f"{v['win_rate']:.0%}.",
        f"Challenge costs ($2.50/side all-in): Sharpe {c['sharpe']:+.2f}, "
        f"{c['trade_mean_R']:+.3f}R/trade.",
        f"Random-timing null p {r['null_p']:.3f} (median null Sharpe {r['null_median']:+.2f}); "
        "pre-registered threshold 0.05.",
        f"Halves {v['halves'][0]:+.2f}/{v['halves'][1]:+.2f}; Sharpe without best 3 days "
        f"{v['sharpe_ex_top3']:+.2f}; monthly "
        + ", ".join(f"{k} {x:+.2%}" for k, x in r["monthly"].items())
        + ".",
        "Per market (R): " + ", ".join(f"{k} {x:+.1f}" for k, x in v["per_market_R"].items()),
    ]
    verdict = Verdict.inconclusive(
        "T84 pre-registered Q-VWAP 1sd re-entry + TEMA9xEMA50, VALIDATE look 1",
        reason=f"NOT CONFIRMED under the pre-registered criterion: Sharpe and per-trade mean "
        f"are positive but the timing null p is {r['null_p']:.3f} > 0.05, and the result rests "
        "on October 2025 (second half flat, negative without its best 3 days).",
        evidence=ev,
        caveat="One validate look spent. The sign held out of sample; the size did not reach "
        "significance in six months.",
        next_step="No further validate looks on this rule. The holdout (2026-04 onward) is the "
        "only clean test left and is ONE look, ever.",
    )
    append(
        "T84",
        [
            "T84",
            "2026-10-05",
            "VALIDATE: Q anchored VWAP 1sd re-entry + recent TEMA9xEMA50 "
            "cross, hold 1 session (preregistered/T84.json)",
            "Yahoo =F 1h, 12 markets",
            "validate 2025-10-01..2026-03-31, look 1",
            "single pre-registered rule",
            f"SR {v['sharpe']:+.2f}, {v['trade_mean_R']:+.3f}R/trade, null p {r['null_p']:.3f}",
            "7,11,16,17,18,20,21; costs reference + challenge",
            verdict.state,
            verdict.reason,
            verdict.next_step,
        ],
        verdict.render(),
    )


def t85() -> None:
    r = json.loads((OUT / "t85_results.json").read_text())
    w = r["winner_validate"]
    conf = r["by_conf"]
    cross_means = [v["mean"] for k, v in conf.items() if k != "no cross"]
    ev = [
        f"{r['cells']} cells on {r['train_days']} train days; {r['share_positive']:.0%} have "
        f"positive Sharpe. Best of {r['cells']} noise strategies would show about "
        f"{r['noise_bar_sharpe']:.1f}; the best cell shows "
        f"{max(v['sharpe'] for v in r['top20'].values()):.2f}.",
        f"PBO {r['pbo']:.2f} over {int(r['pbo_detail']['n_combinations'])} CSCV splits; "
        f"degradation slope {r['pbo_detail']['degradation_slope']:+.2f} (negative = better "
        "in-sample predicts worse out-of-sample).",
        f"Train winner {r['winner']} ({r['top20'][r['winner']]['trades']:.0f} trades) on "
        f"validate: Sharpe {w['sharpe']:+.2f}, {w['trades']} trades, "
        f"{w['trade_mean_R']:+.3f}R/trade.",
        f"Factor means (read before any cell, check 3): no confirmation cross "
        f"{conf['no cross']['mean']:+.2f}; every one of the 12 cross variants is higher "
        f"({min(cross_means):+.2f} to {max(cross_means):+.2f}). Hold 4 bars "
        f"{r['by_hold']['hold 4']['mean']:+.2f} (costs); 1.5sd the best distance on average "
        f"({r['by_k']['1.5sd']['mean']:+.2f}).",
        f"The pre-registered T84 cell ranks {r['prereg_cell_rank']} of {r['cells']} on train "
        f"and is the only top-20 cell with more than 141 trades "
        f"({r['prereg_cell_train']['trades']:.0f}). The cells above it trade 6-105 times.",
    ]
    verdict = Verdict.no_edge(
        "T85 optimisation: anchor x distance x confirmation x hold, 624 cells, train",
        reason=f"Optimising the settings does not find anything better: PBO {r['pbo']:.2f}, "
        f"no cell clears the {r['noise_bar_sharpe']:.1f} noise bar, and the train winner "
        f"loses on validate (Sharpe {w['sharpe']:+.2f}).",
        evidence=ev,
        caveat="The robust, factor-level finding is that a confirmation cross matters: "
        "unconfirmed band re-entries are the worst family by a wide margin.",
        next_step="Keep the T84 rule as specified; do not re-tune it. Validate look 2 is spent.",
    )
    append(
        "T85",
        [
            "T85",
            "2026-10-05",
            "Optimisation of VWAP-band re-entry: anchor, distance, "
            "confirmation cross (4 pairs x 3 windows), hold",
            "Yahoo =F 1h, 12 markets",
            "train (selection) + validate look 2 (winner only)",
            f"{r['cells']} cells",
            f"winner {r['winner']}: train SR {r['top20'][r['winner']]['sharpe']:+.2f}, validate "
            f"{w['sharpe']:+.2f}; PBO {r['pbo']:.2f}",
            "3,12,16,18 + PBO/CSCV + noise bar",
            verdict.state,
            verdict.reason,
            verdict.next_step,
        ],
        verdict.render(),
    )


if __name__ == "__main__":
    t84()
    t85()
