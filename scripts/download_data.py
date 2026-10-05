"""Download SPY 1-minute bars (and daily adjusted bars) from the Alpaca Market Data API.

Usage (from the repo root, with keys in .env):
    python scripts/download_data.py --start 2016-01-01

Output:
    data/raw/SPY_1min_<year>.parquet   raw (unadjusted) minute bars, all sessions, UTC timestamps
    data/SPY_daily_adj.parquet         daily bars adjusted for splits + dividends (buy & hold benchmark)
    data/SPY_dividends.csv             cash dividends per share by ex-date (derived from raw vs adjusted closes)

Notes:
- The free Alpaca plan gives historical SIP data (consolidated tape) from 2016 onwards,
  as long as the requested window ends more than 15 minutes in the past.
- Raw prices are used for the strategy because commissions are charged per share,
  so the real share price level matters for cost realism.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _headers() -> dict[str, str]:
    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY_ID"), os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY. Copy .env.example to .env and fill it in.")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_bars(symbol: str, timeframe: str, start: str, end: str, adjustment: str, feed: str) -> pd.DataFrame:
    """Fetch all bars between start and end, following Alpaca's pagination."""
    params = {"timeframe": timeframe, "start": start, "end": end, "limit": 10_000,
              "adjustment": adjustment, "feed": feed}
    headers, rows = _headers(), []
    while True:
        resp = requests.get(BASE_URL.format(symbol=symbol), params=params, headers=headers, timeout=60)
        if resp.status_code == 429:  # rate limit: wait and retry
            time.sleep(5)
            continue
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("bars") or [])
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                                            "c": "close", "v": "volume", "n": "trade_count", "vw": "bar_vwap"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None, help="default: now minus 20 minutes")
    parser.add_argument("--feed", default="sip", choices=["sip", "iex"])
    args = parser.parse_args()

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    start = pd.Timestamp(args.start, tz="UTC")
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for year in range(start.year, end.year + 1):
        y_start = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        y_end = min(end, pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        out = raw_dir / f"{args.symbol}_1min_{year}.parquet"
        if out.exists() and year < end.year:
            print(f"{out.name} exists, skipping")
            continue
        print(f"Downloading {args.symbol} 1-min bars for {year} ...", flush=True)
        df = fetch_bars(args.symbol, "1Min", y_start.isoformat(), y_end.isoformat(), "raw", args.feed)
        df.to_parquet(out)
        print(f"  saved {len(df):,} bars -> {out.relative_to(DATA_DIR.parent)}")

    print("Downloading daily adjusted bars (benchmark) ...")
    daily = fetch_bars(args.symbol, "1Day", start.isoformat(), end.isoformat(), "all", args.feed)
    daily.to_parquet(DATA_DIR / f"{args.symbol}_daily_adj.parquet")
    print(f"  saved {len(daily):,} daily bars.")

    # Dividends: with f_t = adj_close_t / raw_close_t, on an ex-date t the previous raw close
    # expressed in today's terms is raw_{t-1} * f_{t-1} / f_t = raw_{t-1} - D_t.
    raw = fetch_bars(args.symbol, "1Day", start.isoformat(), end.isoformat(), "split", args.feed)
    f = (daily["close"] / raw["close"]).dropna()
    prev_raw = raw["close"].shift(1).reindex(f.index)
    div = (prev_raw * (1 - f.shift(1) / f)).round(4)
    div = div[div > 0.01].rename("dividend")
    div.index = div.index.tz_convert("America/New_York").normalize().tz_localize(None)
    div.to_csv(DATA_DIR / f"{args.symbol}_dividends.csv")
    print(f"  found {len(div)} dividends -> data/{args.symbol}_dividends.csv. Done.")


if __name__ == "__main__":
    main()
