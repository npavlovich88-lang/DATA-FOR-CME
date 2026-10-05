# Q-VWAP re-entry screener (TradingView)

`vwap_reentry_screener.pine` watches 12 CME futures for the T84 rule on hourly bars. It shows a
table of which markets are near a setup, and it can raise an alert when one fires. You execute
every trade by hand, because the CME University Trading Challenge does not permit
algorithmic trading.

## Install
1. In TradingView, open the Pine Editor, paste the file in, then **Save** and **Add to chart**.
   Any chart and any timeframe works; everything is computed on 1-hour bars.
2. For alerts: Alerts → Create → Condition: *Q-VWAP re-entry screener* → **Any alert() function
   call**. You'll get one message per market per closed hourly bar when it signals.
3. If the script fails to compile on `backadjustment`, set the input *Back-adjusted continuous
   contracts* off, or delete the `ticker.modify(...)` call in `sym()`. Back-adjustment needs
   Pine v6.

The script hasn't been compiled here (there's no Pine compiler in the research environment).
Paste it in and send back any error message. To sanity-check its numbers, compare the table
with `python crossmarket/screener_snapshot.py`, which computes the same columns from the research
data. Expect small differences from TradingView's own data and volume.

## Reading the table
| column | meaning |
|---|---|
| z | distance from the quarterly VWAP in band sd. −1 is the lower band, +1 the upper |
| trend | TEMA(9) vs EMA(50), hourly |
| x-up / x-dn | hourly bars since the last TEMA/EMA cross up / down |
| status | **SIGNAL BUY/SELL**: the rule fired on the last closed hour. **WATCH … armed**: price is at or beyond a band and a matching cross is fresh, so a move back inside the band fires the rule. **WATCH**: at a band, no fresh cross yet. **warm-up**: first 5 sessions of the quarter |
| size | contracts for the chosen risk per trade |

**Exit:** 23 hourly bars after entry, which is about one Globex session. There's no stop and no target in the tested rule.

## Status of the rule
Train Sharpe +1.80, validate +1.43 (not significant), **holdout −0.25** (T84–T86 in
`out/test_log.xlsx`). It's an unproven idea. Size it as if it has no edge (see
`out/challenge_risk.json`).

## Challenge reminders
- At least 10 contracts per day, or a $1,000 penalty. On a day with no signal, buy and sell 5 MES (about $31).
- Close CL and NG before their expiries. Each contract still open at expiration costs $1,000.
- Be flat 15 minutes before the close on Fri 2026-10-30.
- Losing 20% in one day locks the account for the rest of that day.
