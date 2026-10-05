# Patches for `Quantitative_Trading_Projects`

The cross-market code belongs in `npavlovich88-lang/Quantitative_Trading_Projects`, but this
session can only push to `DATA-FOR-CME`. So changes ship here as `git format-patch` files.
To apply one, run this from a checkout of the quant repo:

```
git am /path/to/DATA-FOR-CME/patches/quantitative_trading_projects/0001-*.patch
```

## 0001: per-instrument costs for the 13 cross-market futures (step 1)

This patch changes `src/quant/costs.py` and adds `tests/test_costs_futures.py`. MES and MNQ
behave exactly as before, and the existing cost tests still pass.

Round-turn cost for 1 contract, at the IBKR reference commission of $0.85/side:

| root | exch | tick | point value | tick $ | spread (ticks) | fees RT $ | round turn $ | ticks |
|---|---|---|---|---|---|---|---|---|
| ES | CME | 0.25 | 50 | 12.50 | 1 | 4.48 | 16.98 | 1.36 |
| NQ | CME | 0.25 | 20 | 5.00 | 1 | 4.48 | 9.48 | 1.90 |
| RTY | CME | 0.10 | 50 | 5.00 | 1 | 4.48 | 9.48 | 1.90 |
| YM | CBOT | 1 | 5 | 5.00 | 1 | 4.48 | 9.48 | 1.90 |
| ZB | CBOT | 1/32 | 1,000 | 31.25 | 1 | 3.46 | 34.71 | 1.11 |
| ZN | CBOT | 1/64 | 1,000 | 15.625 | 1 | 3.46 | 19.09 | 1.22 |
| CL | NYMEX | 0.01 | 1,000 | 10.00 | 1 | 4.92 | 14.92 | 1.49 |
| NG | NYMEX | 0.001 | 10,000 | 10.00 | 2 | 4.92 | 24.92 | 2.49 |
| HG | COMEX | 0.0005 | 25,000 | 12.50 | 2 | 4.92 | 29.92 | 2.39 |
| 6E | CME | 0.00005 | 125,000 | 6.25 | 2 | 4.92 | 17.42 | 2.79 |
| 6B | CME | 0.0001 | 62,500 | 6.25 | 2 | 4.92 | 17.42 | 2.79 |
| 6J | CME | 0.0000005 | 12,500,000 | 6.25 | 2 | 4.92 | 17.42 | 2.79 |
| SB | ICE US | 0.01¢ | 1,120 per ¢ | 11.20 | 1 | 5.72 | 16.92 | 1.51 |

"Fees RT" means round-turn fees: 2 × (commission $0.85 + exchange fee + NFA $0.01).

Where each kind of number comes from, and how far to trust it:

- **Tick size and point value** come from the exchange contract specs. They are solid.
- **Exchange fees** are non-member per-side rates taken from broker pass-through schedules:
  E-mini equity $1.38, Treasuries $0.87, NYMEX/COMEX $1.60, FX $1.60, ICE sugar $2.00.
  - Only search snippets were available, because the CME, IBKR and TradeStation pages are
    blocked from the session container.
  - The ZN, 6B and 6J fees were assumed equal to their group's quoted figure.
  - The sugar fee comes from ICE's 2016 schedule.
  - CME changed fees on 2026-10-01, so verify all of these. Fees are 15-35% of the round turn,
    so they matter less than the spread.
- **Spreads are assumed, not measured.** No level-1 data exists for these markets.
  - One tick is used for the deep front months.
  - Two ticks are used for NG, HG and all three FX contracts. 6E and 6J halved their tick in
    2019, so one tick before that equals two ticks now.
- **Commission has no default.** `CostModel.for_reference_broker()` uses $0.85/side, named
  explicitly as the reference rate.
- **`check_price_units`** fails if a Yahoo series isn't in the units its point value assumes.
  For example, SB must be in cents/lb and 6J in USD per yen. A unit mismatch would scale every
  P&L by 100.
