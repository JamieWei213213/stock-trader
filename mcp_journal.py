"""MCP server over the trading records (v3). Read-only: it can explain, list and search; it cannot trade.

Lets Claude Desktop (or any MCP client) ask "why did we sell HOOD on Tuesday?" against your own journal.
Runs on your PC against a local copy of state/ and reports/ (sync from the VPS with scp or rsync; see README).

Install:  pip install mcp        Run by hand:  python mcp_journal.py      (stdio transport; Claude Desktop launches it)
Claude Desktop config (claude_desktop_config.json):
  {"mcpServers": {"stocktrader": {"command": "C:\\\\path\\\\to\\\\.venv\\\\Scripts\\\\python.exe",
                                  "args": ["C:\\\\path\\\\to\\\\StockTrading\\\\mcp_journal.py"]}}}
Tools: get_trade, list_trades, scorecard_summary, search_journal, cost_report, cycle_log, killswitch_status.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

from trader.journal import Journal
from trader.settings import Settings

s = Settings()
mcp = FastMCP("stocktrader-journal")


def _journal() -> Journal:
    return Journal(s.state_dir)


def _num(x, nd=3):
    try:
        return None if pd.isna(x) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


@mcp.tool()
def get_trade(symbol: str, date: str = "") -> dict:
    """Full evidence chain for one symbol: journal entries (features, the anonymized text the news agent read, each
    agent's output, the decision and reason, the order) and the trades ledger (entry, stop, target, exit, P&L, post-mortem).
    date: optional YYYY-MM-DD to narrow to one day."""
    j = _journal()
    entries = [e for e in j.entries_for(symbol.upper(), limit=30) if not date or e.get("logged", "").startswith(date)]
    trades = [t for t in j.trades if t["symbol"] == symbol.upper() and (not date or date in (t.get("entry_time", "") + str(t.get("exit_time", ""))))]
    for e in entries:   # keep replies, trim the big prompt
        if e.get("prompt_sent"):
            e["prompt_sent"] = e["prompt_sent"][:2000]
    return {"symbol": symbol.upper(), "entries": entries, "trades": trades}


@mcp.tool()
def list_trades(since: str = "", exit_reason: str = "", account: str = "", status: str = "") -> list[dict]:
    """Trades from the ledger. since: YYYY-MM-DD; exit_reason e.g. stop_loss, trail_stop, take_profit, time_stop, stale, hard_max, sold;
    account: small|large; status: open|closed."""
    out = []
    for t in _journal().trades:
        if since and t.get("entry_time", "") < since:
            continue
        if exit_reason and not str(t.get("exit_reason", "")).startswith(exit_reason):
            continue
        if account and t["account"] != account:
            continue
        if status and t["status"] != status:
            continue
        out.append({k: t.get(k) for k in ("id", "account", "symbol", "status", "entry_time", "entry_price", "qty", "stop", "target",
                                          "forecast_5d_pct", "confidence", "catalyst", "entry_reason", "exit_time", "exit_price",
                                          "exit_reason", "pnl", "pnl_pct", "post_mortem")})
    return out


@mcp.tool()
def scorecard_summary(window_days: int = 30) -> dict:
    """The evaluation numbers: forecast-vs-realized correlation, approved vs rejected 5-day returns, veto accuracy,
    per-agent ablation (news-only vs combined), for rows logged in the last window_days."""
    p = s.state_dir / "scorecard.csv"
    if not p.exists():
        return {"error": "no scorecard yet"}
    df = pd.read_csv(p)
    df["logged"] = pd.to_datetime(df["logged"], errors="coerce")
    df = df[df["logged"] >= pd.Timestamp.now() - pd.Timedelta(days=window_days)]
    if "vetoed" in df:
        df = df[~df["vetoed"].fillna("").astype(str).str.startswith("held")]
    d = df.dropna(subset=["ret_5d"]).copy()
    out = {"rows_logged": int(len(df)), "rows_with_realized_5d": int(len(d))}
    if len(d) >= 3:
        for c in ("expected_5d_pct", "news_exp", "confidence"):
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        out["corr_forecast_vs_realized"] = _num(d[["expected_5d_pct", "ret_5d"]].corr().iloc[0, 1])
        if "news_exp" in d and d["news_exp"].notna().sum() >= 3:
            out["corr_news_only_vs_realized"] = _num(d[["news_exp", "ret_5d"]].corr().iloc[0, 1])
        appr = d["approved"].astype(str).str.lower() == "true"
        out["approved_mean_5d_pct"] = _num(d[appr]["ret_5d"].mean() * 100)
        out["rejected_mean_5d_pct"] = _num(d[~appr]["ret_5d"].mean() * 100)
        out["acted_mean_5d_pct"] = _num(d[d["acted"].astype(str).str.lower() == "true"]["ret_5d"].mean() * 100)
        if "vetoed" in d:
            v = d["vetoed"].fillna("").astype(str)
            out["veto_reasons"] = v[v != ""].str.split(":").str[0].value_counts().to_dict()
    out["note"] = "n < 30 is a preview, not a result. Run eval_agents.py for the full report."
    return out


@mcp.tool()
def search_journal(query: str, limit: int = 20) -> list[dict]:
    """Case-insensitive text search across decision reasons, catalysts, risks, market notes and post-mortem lessons."""
    q = re.compile(re.escape(query), re.I)
    hits = []
    for e in _journal().entries_for(limit=5000):
        hay = json.dumps({k: e.get(k) for k in ("forecast", "manager", "market_note", "agents")}, default=str)
        if q.search(hay):
            hits.append({"logged": e.get("logged"), "account": e.get("account"), "symbol": e.get("symbol"),
                         "forecast": e.get("forecast"), "decision": e.get("manager"), "file": e.get("_file")})
            if len(hits) >= limit:
                break
    lp = s.state_dir / "lessons.jsonl"
    if lp.exists():
        for line in lp.read_text(encoding="utf-8").splitlines():
            if q.search(line):
                hits.append({"lesson": json.loads(line)})
    return hits


@mcp.tool()
def cost_report(days: int = 30) -> dict:
    """Claude API spend per day and agent reliability counters."""
    cp = s.state_dir / "costs.json"
    out = {"days": {}}
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        keys = sorted(c)[-days:]
        out["days"] = {k: c[k] for k in keys}
        out["total_usd"] = round(sum(c[k]["usd"] for k in keys), 4)
    ap = s.state_dir / "agent_stats.json"
    if ap.exists():
        out["agent_stats"] = json.loads(ap.read_text(encoding="utf-8"))
    return out


@mcp.tool()
def cycle_log(n: int = 5, account: str = "") -> list[dict]:
    """The last n cycle records: candidates, forecasts, decision, vetoes, regime, per-agent stats, executor log."""
    d = s.state_dir / "cycles"
    if not d.exists():
        return []
    files = sorted(d.glob(f"*_{account}*.json" if account else "*.json"))[-n:]
    out = []
    for p in files:
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            c.pop("forecasts", None)
            out.append({"file": p.name, **c})
        except json.JSONDecodeError:
            continue
    return out


@mcp.tool()
def killswitch_status() -> dict:
    """Whether either account's kill switch is tripped, and the equity peak / day-start it is measured against."""
    out = {}
    for name in ("small", "large"):
        p = s.state_dir / f"killswitch_{name}.json"
        out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"tripped": None}
    return out


if __name__ == "__main__":
    mcp.run()
