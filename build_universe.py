"""Builds universe.yaml: the N most liquid US common stocks, ranked by 30-day average dollar volume.

Source: Alpaca's asset list (NYSE + NASDAQ, active, tradable) minus ETFs/funds/trusts by name,
then 30 days of daily bars for all of them (fetched in chunks; ~2-3 minutes for ~5,000 symbols).

Run weekly (cron, Sunday) or by hand:
  python build_universe.py                 # size from config.yaml (screener.universe_size)
  python build_universe.py --size 150

watchlist.yaml still matters: its names are ALWAYS included, and its `volatile: true` tags carry over.
Any universe name not in watchlist.yaml is treated as volatile if its ATR% is above the universe median.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from trader.broker import Broker
from trader.settings import Settings

FUND_WORDS = re.compile(
    r"\b(ETF|ETN|Fund|Trust|Index|iShares|SPDR|ProShares|Direxion|VanEck|Invesco|WisdomTree|"
    r"Vanguard|Schwab|Global X|Bond|Treasury|Preferred|Warrant|Rights?|Units?|Depositary|ADR|"
    r"Acquisition|SPAC|Notes? due|Ultra|Leveraged|Bull|Bear|2X|3X|Daily)\b", re.I)


def candidate_symbols(broker: Broker) -> dict[str, str]:
    """symbol -> company name"""
    syms: dict[str, str] = {}
    for exch in (AssetExchange.NYSE, AssetExchange.NASDAQ):
        assets = broker.trading.get_all_assets(
            GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY, exchange=exch))
        for a in assets:
            if not a.tradable or not a.symbol.isalpha() or len(a.symbol) > 5:
                continue  # skip preferreds/warrants like BRK.B-style or ABC.WS, and weird tickers
            if FUND_WORDS.search(a.name or ""):
                continue
            syms[a.symbol] = a.name or ""
    return dict(sorted(syms.items()))


def liquidity_table(broker: Broker, symbols: list[str], days: int = 30, chunk: int = 200) -> pd.DataFrame:
    start = datetime.now(timezone.utc) - timedelta(days=int(days * 1.6))
    rows = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            df = broker.data.get_stock_bars(
                StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start, feed="iex")).df
        except Exception as e:
            print(f"  chunk {i//chunk}: {e}")
            time.sleep(2)
            continue
        if df.empty:
            continue
        df["dollar_vol"] = df["close"] * df["volume"]
        g = df.groupby(level="symbol")
        tr = (df["high"] - df["low"]) / df["close"]
        agg = pd.DataFrame({
            "price": g["close"].last(),
            "avg_dollar_vol": g["dollar_vol"].mean(),
            "days": g["close"].count(),
            "atr_pct": tr.groupby(level="symbol").mean(),
        })
        rows.append(agg)
        print(f"  {min(i+chunk, len(symbols))}/{len(symbols)} symbols fetched", end="\r")
        time.sleep(0.35)  # stay under 200 req/min
    print()
    return pd.concat(rows) if rows else pd.DataFrame()


def industry_tags(symbols: list[str], state_dir, key: str) -> dict[str, str]:
    """v3: Finnhub company profile 'finnhubIndustry' per symbol, cached forever in state/industries.json
    (industries rarely change). Free tier = 60 calls/min, so ~500 new names take ~9 minutes the first time."""
    import requests
    path = state_dir / "industries.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [s for s in symbols if s not in cache]
    if missing and not key:
        print(f"  [sectors] FINNHUB_KEY not set; {len(missing)} names without industry tags (risk reviewer skips them)")
        return cache
    for i, sym in enumerate(missing, 1):
        try:
            r = requests.get("https://finnhub.io/api/v1/stock/profile2", params={"symbol": sym, "token": key}, timeout=15)
            if r.status_code == 429:
                time.sleep(20); r = requests.get("https://finnhub.io/api/v1/stock/profile2", params={"symbol": sym, "token": key}, timeout=15)
            cache[sym] = (r.json() or {}).get("finnhubIndustry") or "unknown"
        except Exception as e:
            cache[sym] = "unknown"; print(f"  [sectors] {sym}: {e}")
        if i % 25 == 0:
            path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
            print(f"  [sectors] {i}/{len(missing)} tagged", end="\r")
        time.sleep(1.05)   # 60/min
    path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--no-sectors", action="store_true", help="skip Finnhub industry tagging (v3 risk reviewer uses the tags)")
    args = ap.parse_args()
    s = Settings()
    size = args.size or s.cfg["screener"].get("universe_size", 500)
    broker = Broker(s.creds("large"))

    print("listing assets ...")
    names = candidate_symbols(broker)
    syms = list(names)
    print(f"{len(syms)} candidate common stocks; fetching 30d bars ...")
    tab = liquidity_table(broker, syms)
    tab = tab[(tab["days"] >= 15) & (tab["price"] >= args.min_price)]
    tab = tab.sort_values("avg_dollar_vol", ascending=False)

    manual = {w["symbol"]: w for w in s.watchlist}
    top = list(tab.index[:size])
    for sym in manual:  # always keep hand-picked names
        if sym not in top:
            top.append(sym)
    vol_median = float(tab["atr_pct"].median())
    inds = {} if args.no_sectors else industry_tags(top, s.state_dir, os.getenv("FINNHUB_KEY", ""))
    out = []
    for sym in top:
        if sym in manual:
            out.append({"symbol": sym, "name": names.get(sym, ""), "sector": manual[sym].get("sector", "?"),
                        "industry": inds.get(sym, manual[sym].get("industry", "unknown")),
                        "volatile": bool(manual[sym].get("volatile", False))})
        else:
            atr = float(tab.loc[sym, "atr_pct"]) if sym in tab.index else vol_median
            out.append({"symbol": sym, "name": names.get(sym, ""), "sector": "auto", "industry": inds.get(sym, "unknown"),
                        "volatile": bool(atr > vol_median * 1.5)})

    path = s.root / "universe.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-built {datetime.now():%Y-%m-%d %H:%M} by build_universe.py — top {size} US stocks by 30d dollar volume.\n")
        f.write("# Do not edit by hand; edit watchlist.yaml for names you always want included / volatile tags.\n")
        yaml.safe_dump({"stocks": out}, f, sort_keys=False)
    nvol = sum(1 for o in out if o["volatile"])
    print(f"wrote {path}: {len(out)} names ({nvol} tagged volatile). Top 10: {top[:10]}")

    # wider pool for the backtest (point-in-time universe is selected from this each date)
    wide_n = s.cfg["screener"].get("wide_pool_size", 1500)
    wide = [{"symbol": sym, "name": names.get(sym, "")} for sym in tab.index[:wide_n]]
    wpath = s.root / "universe_wide.yaml"
    with open(wpath, "w", encoding="utf-8") as f:
        f.write(f"# Auto-built {datetime.now():%Y-%m-%d} — top {wide_n} by 30d dollar volume. Backtest pool only.\n")
        yaml.safe_dump({"stocks": wide}, f, sort_keys=False)
    print(f"wrote {wpath}: {len(wide)} names (backtest pool)")


if __name__ == "__main__":
    main()
