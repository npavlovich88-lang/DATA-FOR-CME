"""Download Yahoo `=F` continuous front-month bars for the 13 cross-market futures.

Yahoo serves 60m bars for the last ~730 days only; older windows are refused. Daily bars go
back to ~2000. Output is CSV under $CROSSMARKET_DATA (default ./data, git-ignored): the data is
Yahoo's and is not redistributed in this repository.

    python crossmarket/fetch_yahoo.py            # both intervals
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.request

import pandas as pd

SYMBOLS = ["ES", "NQ", "RTY", "YM", "ZB", "ZN", "CL", "NG", "HG", "6E", "6B", "6J", "SB"]
OUT = pathlib.Path(os.environ.get("CROSSMARKET_DATA", "data"))
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(sym: str, interval: str) -> pd.DataFrame:
    now = int(time.time())
    p1 = now - 729 * 86400 if interval == "60m" else 0
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}=F"
        f"?period1={p1}&period2={now}&interval={interval}&includePrePost=false"
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                res = json.load(r)["chart"]["result"][0]
            break
        except Exception as e:  # network blips: retry with backoff
            if attempt == 3:
                raise
            print(f"  {sym} {interval}: {e}; retrying", file=sys.stderr)
            time.sleep(2 ** (attempt + 1))
    q = res["indicators"]["quote"][0]
    tz = res["meta"]["exchangeTimezoneName"]
    df = pd.DataFrame(
        {k: q[k] for k in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz),
    )
    df.index.name = "ts"
    return df.dropna(subset=["open", "high", "low", "close"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for interval, tag in (("60m", "1h"), ("1d", "1d")):
        for s in SYMBOLS:
            df = fetch(s, interval)
            df.to_csv(OUT / f"{s}_{tag}.csv")
            print(f"{s:>4} {tag}: {len(df):>6} bars  {df.index[0]} .. {df.index[-1]}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
