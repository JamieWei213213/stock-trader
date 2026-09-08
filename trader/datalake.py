"""Historical data lake: daily bars for the watchlist stored as Parquet under data/.

Borrowed idea: keep raw history locally once, then run any number of backtests for free.
Alpaca's free IEX feed goes back to 2016. First fetch takes ~1 min; later runs only
append the missing days.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .broker import Broker

COLS = ["open", "high", "low", "close", "volume"]


class DataLake:
    def __init__(self, root: Path):
        self.dir = root / "data"
        self.dir.mkdir(exist_ok=True)
        self.path = self.dir / "daily_bars.parquet"

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=COLS, index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]))

    def update(self, broker: Broker, symbols: list[str], years: int = 4) -> pd.DataFrame:
        have = self.load()
        full_start = datetime.now(timezone.utc) - timedelta(days=365 * years)
        known = set(have.index.get_level_values("symbol")) if not have.empty else set()
        missing = [s for s in symbols if s not in known]          # brand-new symbols need full history
        existing = [s for s in symbols if s in known]
        start = full_start
        if not have.empty:
            last = have.index.get_level_values("timestamp").max()
            start = max(full_start, pd.Timestamp(last).to_pydatetime() + timedelta(days=1))
        jobs = []
        if missing:
            jobs.append((missing, full_start))
        if existing and start.date() < datetime.now(timezone.utc).date():
            jobs.append((existing, start))
        if not jobs:
            print(f"data lake up to date ({len(have)} rows, {len(known)} symbols)")
            return have
        parts = []
        for syms, st in jobs:
            print(f"fetching daily bars for {len(syms)} symbols since {st:%Y-%m-%d} ...")
            chunk = 100
            for i in range(0, len(syms), chunk):
                try:
                    part = broker.daily_bars_since(syms[i:i + chunk], st)
                    if not part.empty:
                        parts.append(part[COLS])
                except Exception as e:
                    print(f"  chunk {i // chunk} failed: {e}")
                print(f"  {min(i + chunk, len(syms))}/{len(syms)}", end="\r")
            print()
        if not parts:
            return have
        new = pd.concat(parts)
        df = pd.concat([have, new]) if not have.empty else new
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(self.path)
        print(f"data lake: {len(df)} rows, {df.index.get_level_values('symbol').nunique()} symbols, "
              f"{df.index.get_level_values('timestamp').min():%Y-%m-%d} → {df.index.get_level_values('timestamp').max():%Y-%m-%d}")
        return df
