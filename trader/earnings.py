"""Earnings blackout: don't open a position within N trading days of a report.

Sources (tried in order, cached once per day in state/earnings.json):
  1. Finnhub free tier, if FINNHUB_KEY is set in .env
  2. Nasdaq's public earnings calendar (no key; one request per calendar day, ~14/day)
If both fail, the blackout is disabled for the day with a warning.
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests


def _bdays(a: date, b: date) -> int:
    """Signed number of weekdays from a to b (b > a positive)."""
    if b >= a:
        return int(np.busday_count(a.isoformat(), b.isoformat()))
    return -int(np.busday_count(b.isoformat(), a.isoformat()))

NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


class EarningsCalendar:
    def __init__(self, state_dir: Path, blackout_days: int = 2, horizon_days: int = 21, recent_days: int = 0):
        self.path = state_dir / "earnings.json"
        self.blackout_days = blackout_days      # v3: run_cycle passes max(config, account max_hold_days)
        self.recent_days = recent_days          # v3: also skip names that reported within the last N trading days
        self.horizon_days = horizon_days
        self.key = os.getenv("FINNHUB_KEY", "")
        self.dates: dict[str, str] = {}   # symbol -> next report date (YYYY-MM-DD)
        self.source = "none"
        self._load_or_fetch()

    # ---- sources ----
    def _finnhub(self, start: date, end: date) -> dict[str, str]:
        r = requests.get("https://finnhub.io/api/v1/calendar/earnings",
                         params={"from": start.isoformat(), "to": end.isoformat(), "token": self.key}, timeout=20)
        r.raise_for_status()
        out: dict[str, str] = {}
        for row in r.json().get("earningsCalendar", []):
            sym, d = row.get("symbol"), row.get("date")
            if sym and d and (sym not in out or d < out[sym]):
                out[sym] = d
        return out

    def _nasdaq(self, start: date, end: date) -> dict[str, str]:
        out: dict[str, str] = {}
        d = start
        while d <= end:
            if d.weekday() < 5:
                r = requests.get("https://api.nasdaq.com/api/calendar/earnings",
                                 params={"date": d.isoformat()}, headers=NASDAQ_HEADERS, timeout=20)
                r.raise_for_status()
                rows = ((r.json() or {}).get("data") or {}).get("rows") or []
                for row in rows:
                    sym = row.get("symbol")
                    if sym and sym not in out:
                        out[sym] = d.isoformat()
            d += timedelta(days=1)
        return out

    def _load_or_fetch(self):
        today = date.today()
        if self.path.exists():
            cached = json.loads(self.path.read_text(encoding="utf-8"))
            if cached.get("fetched") == today.isoformat():
                self.dates, self.source = cached["dates"], cached.get("source", "cache")
                return
        end = today + timedelta(days=self.horizon_days)
        start = today - timedelta(days=int(self.recent_days * 1.5) + 2) if self.recent_days else today   # +2 covers a weekend
        for name, fn in (("finnhub", self._finnhub), ("nasdaq", self._nasdaq)):
            if name == "finnhub" and not self.key:
                continue
            try:
                self.dates = fn(start, end)
                self.source = name
                self.path.write_text(json.dumps({"fetched": today.isoformat(), "source": name, "dates": self.dates}),
                                     encoding="utf-8")
                print(f"  [earnings] {len(self.dates)} reports in the next {self.horizon_days} days via {name} (cached)")
                return
            except Exception as e:
                print(f"  [earnings] {name} failed: {e}")
        if self.path.exists():
            self.dates = json.loads(self.path.read_text(encoding="utf-8")).get("dates", {})
            self.source = "stale-cache"
            print(f"  [earnings] using stale cache ({len(self.dates)} entries)")
        else:
            print("  [earnings] no source available — blackout disabled today")

    # ---- queries ----
    def next_report(self, symbol: str) -> str | None:
        return self.dates.get(symbol)

    def soon(self, symbol: str) -> bool:
        """True if the symbol reports within blackout_days weekdays ahead, or reported within the last
        recent_days weekdays (v3). Weekends are skipped; a Friday report is 1 day back on Monday."""
        d = self.dates.get(symbol)
        if not d:
            return False
        days = _bdays(date.today(), date.fromisoformat(d))   # trading days, weekend-aware
        if 0 <= days <= self.blackout_days:
            return True
        return self.recent_days > 0 and -self.recent_days <= days < 0
