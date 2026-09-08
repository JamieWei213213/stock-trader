"""Scorecard: does Claude's judgment add anything over the screener?

Every cycle logs each candidate with the screener rank, Claude's forecast, and what we did.
daily_summary.py later fills in the realized 1/5/10-day returns. At month end:
    python daily_summary.py --scorecard
compares forward returns of names Claude approved vs rejected vs the screener's raw top picks.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

import pandas as pd

FIELDS = ["logged", "account", "symbol", "rank", "price", "expected_5d_pct", "confidence", "bias",
          "approved", "acted", "ret_1d", "ret_5d", "ret_10d"]


class Scorecard:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "scorecard.csv"
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, FIELDS).writeheader()

    def log(self, account: str, rows: list[dict]):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            for r in rows:
                w.writerow({"logged": now, "account": account, **{k: r.get(k, "") for k in FIELDS[2:]}})

    def fill(self, broker) -> int:
        """Fill realized returns for rows old enough. Returns number of cells filled."""
        df = pd.read_csv(self.path)
        if df.empty:
            return 0
        df["day"] = pd.to_datetime(df["logged"]).dt.normalize()
        syms = sorted(df["symbol"].unique())
        bars = broker.daily_bars(syms, 40)
        if bars.empty:
            return 0
        close = bars["close"].unstack("symbol").sort_index()
        close.index = close.index.tz_convert(None).normalize()
        filled = 0
        for i, row in df.iterrows():
            sym = row["symbol"]
            if sym not in close.columns:
                continue
            series = close[sym].dropna()
            after = series[series.index >= row["day"]]
            if after.empty:
                continue
            base = after.iloc[0]
            for h in (1, 5, 10):
                col = f"ret_{h}d"
                if pd.isna(row[col]) and len(after) > h:
                    df.at[i, col] = round(float(after.iloc[h] / base - 1), 5)
                    filled += 1
        df.drop(columns=["day"]).to_csv(self.path, index=False)
        return filled

    def report(self) -> str:
        df = pd.read_csv(self.path)
        if df.empty or df["ret_5d"].notna().sum() < 5:
            return "scorecard: not enough filled rows yet"
        lines = ["=== Scorecard: 5-day forward return by group (mean, n) ==="]
        d = df[df["ret_5d"].notna()]
        groups = {
            "screener top-6 (all candidates)": d,
            "Claude approved": d[d["approved"] == True],
            "Claude rejected": d[d["approved"] == False],
            "actually bought": d[d["acted"] == True],
        }
        for name, g in groups.items():
            if len(g):
                lines.append(f"{name:>32}: {g['ret_5d'].mean()*100:+.2f}%  (n={len(g)}, hit {(g['ret_5d']>0).mean()*100:.0f}%)")
        corr = d[["expected_5d_pct", "ret_5d"]].dropna()
        if len(corr) > 10:
            lines.append(f"correlation(Claude forecast, realized 5d): {corr.corr().iloc[0,1]:+.2f}  (n={len(corr)})")
        return "\n".join(lines)
