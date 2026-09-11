"""Filings agent (v3): reads the company's last N quarters of financial statements from SEC EDGAR
and asks Haiku for the direction of the next earnings change.

Why: Kim, Muhn & Nikolaev (2024) had GPT-4 read anonymized, standardized financial statements (no text,
no names) and predict the direction of next-period earnings; it beat analysts and matched a trained ML
model. That is the only LLM-in-finance result with a real out-of-sample edge that we know of, so this
agent copies the setup as closely as free data allows: numbers only, anonymized, structured output.

Data: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json  (free, ~10 req/s allowed, needs a
User-Agent with contact info). Ticker->CIK from https://www.sec.gov/files/company_tickers.json.
Cached per symbol in state/filings/<SYM>.json for agents.filings.cache_days.

Output per symbol:
  {"earnings_direction": "up|down|flat", "confidence": 0-1, "quality_flags": [...], "summary": "..",
   "tilt_pct": +/- weight_pct x confidence}      -> the orchestrator adds tilt_pct to the news forecast.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path

import requests

UA = {"User-Agent": "StockTrader research bot (contact: jamiejwei@gmail.com)", "Accept-Encoding": "gzip, deflate"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# (label, [XBRL concepts in preference order], kind)  kind: flow = quarterly duration, stock = point-in-time
CONCEPTS = [
    ("revenue",      ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                      "RevenueFromContractWithCustomerIncludingAssessedTax"], "flow"),
    ("gross_profit", ["GrossProfit"], "flow"),
    ("op_income",    ["OperatingIncomeLoss"], "flow"),
    ("net_income",   ["NetIncomeLoss", "ProfitLoss"], "flow"),
    ("eps_diluted",  ["EarningsPerShareDiluted", "EarningsPerShareBasic"], "flow"),
    ("op_cash_flow", ["NetCashProvidedByUsedInOperatingActivities"], "flow"),
    ("capex",        ["PaymentsToAcquirePropertyPlantAndEquipment"], "flow"),
    ("cash",         ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "stock"),
    ("total_assets", ["Assets"], "stock"),
    ("total_liab",   ["Liabilities"], "stock"),
    ("equity",       ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "stock"),
    ("lt_debt",      ["LongTermDebtNoncurrent", "LongTermDebt"], "stock"),
    ("shares",       ["WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding"], "flow"),
]

FILINGS_SYSTEM = (
    "You are a financial-statement analyst. You receive up to 8 quarters of standardized, anonymized "
    "financial data for ONE company (identity hidden; judge only the numbers). Task, as in the academic "
    "'LLM reads financial statements' setup: predict whether the company's NEXT quarterly earnings (net income / "
    "EPS) will be UP or DOWN versus the same quarter one year earlier, using trend, margins, cash conversion, "
    "leverage and any accounting-quality warning signs (e.g. income growing faster than operating cash flow, "
    "rising receivables/inventory implied by cash gaps, one-off items). Base rate: earnings rise year-over-year "
    "roughly 55-60% of the time, so 'up' with low confidence is the default when the picture is mixed. "
    "Output ONLY compact JSON: {\"earnings_direction\": \"up|down|flat\", \"confidence\": <0.0-1.0>, "
    "\"quality_flags\": [\"<=6 words each\", ...], \"summary\": \"<=25 words\"}"
)


# ------------------------------------------------------------------------------------------ EDGAR
class Edgar:
    def __init__(self, state_dir: Path, session: requests.Session | None = None):
        self.dir = state_dir / "filings"
        self.dir.mkdir(exist_ok=True)
        self.s = session or requests.Session()
        self.s.headers.update(UA)
        self._ciks: dict[str, int] | None = None
        self._lock = threading.Lock()   # cik() is called from several agent threads at once

    def cik(self, symbol: str) -> int | None:
        with self._lock:
            if self._ciks is None:
                p = self.dir / "_tickers.json"
                raw = None
                if p.exists() and time.time() - p.stat().st_mtime < 7 * 86400:
                    try:
                        raw = json.loads(p.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        raw = None
                if raw is None:
                    r = self.s.get(TICKERS_URL, timeout=30); r.raise_for_status()
                    raw = r.json()
                    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(raw), encoding="utf-8"); os.replace(tmp, p)
                self._ciks = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        return self._ciks.get(symbol.upper().replace(".", "-")) or self._ciks.get(symbol.upper())

    def company_facts(self, cik: int) -> dict:
        r = self.s.get(FACTS_URL.format(cik=cik), timeout=60)
        r.raise_for_status()
        time.sleep(0.12)   # stay under SEC's 10 req/s
        return r.json()


# ------------------------------------------------------------------------------------------ extraction
def _frame_key(frame: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"CY(\d{4})Q([1-4])I?", frame or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _annual_key(frame: str) -> int | None:
    m = re.fullmatch(r"CY(\d{4})", frame or "")
    return int(m.group(1)) if m else None


NO_DERIVE = {"eps_diluted", "shares"}   # per-share / share counts can't be derived by subtraction


def extract_quarters(facts: dict, quarters: int = 8) -> dict:
    """-> {"quarters": [(y,q), ... oldest->newest], "series": {label: [val|None per quarter]}, "latest_filed": "YYYY-MM-DD"}
    EDGAR has no Q4 duration frame (the 10-K carries the annual CY#### frame), so Q4 flow values are derived as
    annual - (Q1 + Q2 + Q3) when all three exist. Concepts are merged by priority per quarter (companies switch tags)."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    series: dict[str, dict[tuple[int, int], float]] = {}
    latest_filed = ""
    for label, concepts, kind in CONCEPTS:
        by_frame: dict[tuple[int, int], float] = {}
        annual: dict[int, float] = {}
        for c in concepts:
            units = (gaap.get(c) or {}).get("units") or {}
            vals = units.get("USD") or units.get("USD/shares") or units.get("shares") or []
            for v in vals:
                frame = str(v.get("frame", ""))
                if v.get("filed", "") > latest_filed:
                    latest_filed = v["filed"]
                fk = _frame_key(frame)
                if fk:
                    if (kind == "stock") != frame.endswith("I"):
                        continue
                    by_frame.setdefault(fk, float(v["val"]))
                elif kind == "flow" and label not in NO_DERIVE:
                    ak = _annual_key(frame)
                    if ak:
                        annual.setdefault(ak, float(v["val"]))
        for y, a in annual.items():
            if (y, 4) not in by_frame and all((y, q) in by_frame for q in (1, 2, 3)):
                by_frame[(y, 4)] = a - sum(by_frame[(y, q)] for q in (1, 2, 3))
        if by_frame:
            series[label] = by_frame
    if not series:
        return {"quarters": [], "series": {}, "latest_filed": latest_filed}
    all_q = sorted(set().union(*[set(s) for s in series.values()]))
    # keep only quarters where at least revenue or net income exists
    core = set(series.get("revenue", {})) | set(series.get("net_income", {}))
    all_q = [q for q in all_q if q in core][-quarters:]
    return {"quarters": all_q,
            "series": {k: [v.get(q) for q in all_q] for k, v in series.items()},
            "latest_filed": latest_filed}


def _fmt(x: float | None, scale: float = 1e6) -> str:
    if x is None:
        return "n/a"
    if abs(x) < 100:          # per-share numbers
        return f"{x:.2f}"
    return f"{x / scale:,.0f}"


def render_table(q: dict) -> str:
    """Anonymized text table, oldest -> newest, values in $ millions (EPS in $). Adds YoY growth where possible."""
    qs = q["quarters"]
    if not qs:
        return ""
    hdr = "quarter      " + " ".join(f"{'Q-' + str(len(qs) - 1 - i) if i < len(qs) - 1 else 'Q0(latest)':>11}" for i in range(len(qs)))
    lines = [hdr]
    for label, _, _ in CONCEPTS:
        vals = q["series"].get(label)
        if not vals or all(v is None for v in vals):
            continue
        row = f"{label:<13}" + " ".join(f"{_fmt(v):>11}" for v in vals)
        lines.append(row)
    # derived
    rev, ni, ocf = q["series"].get("revenue"), q["series"].get("net_income"), q["series"].get("op_cash_flow")
    if rev and ni:
        margins = [f"{(n / r * 100):.1f}%" if (r and n is not None) else "n/a" for r, n in zip(rev, ni)]
        lines.append(f"{'net margin':<13}" + " ".join(f"{m:>11}" for m in margins))
    pos = {q: i for i, q in enumerate(qs)}

    def yoy(s):   # same quarter one year earlier, looked up by (year, quarter) — the list may have gaps
        out = []
        for (y, qq), v in zip(qs, s):
            j = pos.get((y - 1, qq))
            prev = s[j] if j is not None else None
            out.append(f"{(v / prev - 1) * 100:+.0f}%" if (v is not None and prev) else "n/a")
        return out
    if rev and any(m != "n/a" for m in yoy(rev)):
        lines.append(f"{'revenue YoY':<13}" + " ".join(f"{m:>11}" for m in yoy(rev)))
    if ni and any(m != "n/a" for m in yoy(ni)):
        lines.append(f"{'net inc YoY':<13}" + " ".join(f"{m:>11}" for m in yoy(ni)))
    if ni and ocf:
        conv = [f"{(o / n):.2f}x" if (n and o is not None and n > 0) else "n/a" for n, o in zip(ni, ocf)]
        lines.append(f"{'OCF/net inc':<13}" + " ".join(f"{m:>11}" for m in conv))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ agent
class FilingsAgent:
    def __init__(self, agents, state_dir: Path, cfg: dict, edgar: Edgar | None = None):
        """agents: trader.agents.Agents (for the Claude call + cost tracking)."""
        self.agents = agents
        self.cfg = cfg
        self.edgar = edgar or Edgar(state_dir)
        self.dir = state_dir / "filings"
        self.dir.mkdir(exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        return self.dir / f"{symbol}.json"

    def cached(self, symbol: str) -> dict | None:
        p = self._cache_path(symbol)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        age = (date.today() - date.fromisoformat(d["fetched"])).days
        if age > self.cfg.get("cache_days", 90):
            return None
        # a new 10-Q must exist ~45 days after quarter end; refetch once the cached filing is > 100 days old
        if d.get("latest_filed") and (date.today() - date.fromisoformat(d["latest_filed"])).days > 100 and age > 7:
            return None
        return d

    def analyze(self, symbol: str, force: bool = False) -> dict | None:
        if not force:
            c = self.cached(symbol)
            if c:
                out = dict(c["result"]); out["_cached"] = True; out["_cost_usd"] = 0.0
                return out
        cik = self.edgar.cik(symbol)
        if not cik:
            return {"symbol": symbol, "error": "no CIK for symbol", "tilt_pct": 0.0}
        facts = self.edgar.company_facts(cik)
        q = extract_quarters(facts, self.cfg.get("quarters", 8))
        table = render_table(q)
        if not table or len(q["quarters"]) < 4:
            return {"symbol": symbol, "error": f"only {len(q['quarters'])} usable quarters", "tilt_pct": 0.0}
        user = f"Standardized quarterly financials, $ millions except per-share (identity hidden):\n\n{table}"
        out = self.agents._call(self.agents.cfg["research_model"], FILINGS_SYSTEM, user, 350)
        if not out:
            return None
        out = self.finish(out, symbol, q["latest_filed"])
        self._cache_path(symbol).write_text(json.dumps({
            "fetched": date.today().isoformat(), "latest_filed": q["latest_filed"], "quarters": q["quarters"],
            "table": table, "result": {k: v for k, v in out.items() if not k.startswith("_")}}, indent=1, default=str),
            encoding="utf-8")
        return out

    def finish(self, out: dict, symbol: str, latest_filed: str) -> dict:
        out["symbol"] = symbol
        out["latest_filed"] = latest_filed
        d = str(out.get("earnings_direction", "flat")).lower()
        out["earnings_direction"] = d if d in ("up", "down", "flat") else "flat"
        try:
            out["confidence"] = min(max(float(out.get("confidence", 0)), 0.0), 1.0)
        except (TypeError, ValueError):
            out["confidence"] = 0.0
        w = float(self.cfg.get("weight_pct", 0.5))
        sign = {"up": 1, "down": -1, "flat": 0}[out["earnings_direction"]]
        out["tilt_pct"] = round(sign * w * out["confidence"], 3)
        return out
