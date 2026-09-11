"""Ask the journal (v3): plain-English questions answered from YOUR system's own records.

  python ask.py "why did we sell CIEN"
  python ask.py "which vetoes were wrong this month"
  python ask.py "what are the lessons so far"
  python ask.py                      # interactive; blank line to quit

The agent only sees data the system wrote (journal entries, trades, scorecard, cycle records, lessons,
equity history, eval numbers) and is told to answer from it or say it isn't there. Symbols mentioned in
the question pull in that symbol's full history. One Haiku call per question (~$0.002-0.01), cost-tracked
like every other agent. It never places orders or changes config.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from trader.agents import Agents
from trader.costs import CostTracker
from trader.journal import Journal
from trader.settings import Settings

ASK_SYSTEM = (
    "You are the analyst for a small automated paper-trading system. Answer the operator's question using ONLY "
    "the records below (decision journal, trades, scorecard, cycle summaries, lessons, equity history). Quote the "
    "actual numbers, dates, forecasts and reasons from the records. If the records do not contain the answer, say "
    "exactly what is missing instead of guessing. Be concise and plain; no headers, no bullet lists unless listing "
    "trades. Never recommend real-money trading. The system's thresholds: buy needs forecast >= 1.5% and confidence "
    ">= 0.6 (v2 manager) or 'not vetoed' (v3: veto if forecast < -0.5%, bearish with conf >= 0.6, or thesis broken); "
    "stops are 2xATR; max hold 5 days (large) / 10 days (small)."
)

MAX_CHARS = 60000   # ~15k tokens of context; Haiku handles it for well under a cent of input


def _tickers(q: str, known: set[str]) -> list[str]:
    return [t for t in dict.fromkeys(re.findall(r"\b[A-Z]{1,5}\b", q)) if t in known]


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def gather(s: Settings, question: str, month: str | None = None) -> str:
    """Deterministic context assembly. Symbol-specific history first, then the system-wide records."""
    j = Journal(s.state_dir)
    parts: list[str] = [f"Today: {datetime.now():%Y-%m-%d %H:%M}"]
    known = set(s.symbols) | {t["symbol"] for t in j.trades}
    syms = _tickers(question, known)

    # --- symbol-specific: every journal entry + every trade for the named symbols
    for sym in syms[:4]:
        ents = j.entries_for(sym, limit=12)
        lines = []
        for e in ents:
            f = e.get("forecast") or {}
            m = e.get("manager") or {}
            ag = e.get("agents") or {}
            lines.append(f"- {e.get('logged')} {e.get('account')} rank {e.get('rank')}{' HELD' if e.get('held') else ''}: "
                         f"forecast {f.get('expected_5d_return_pct')}% conf {f.get('confidence')} {f.get('bias')}; "
                         f"catalyst: {f.get('catalyst')}; risk: {f.get('risk')}"
                         + (f"; filings {ag['filings'].get('earnings_direction')} ({ag['filings'].get('summary', '')})" if ag.get("filings", {}).get("earnings_direction") else "")
                         + (f"; memory {ag['memory'].get('thesis_status')} ({ag['memory'].get('note', '')})" if ag.get("memory", {}).get("thesis_status") not in (None, "none") else "")
                         + f"; decision: {m.get('action')} — {m.get('reason')}"
                         + (f"; order: {e['order']}" if e.get("order") else ""))
        trades = [t for t in j.trades if t["symbol"] == sym]
        tl = [f"- trade {t['id']} {t['account']} {t['entry_time']} buy {t['qty']} @ {t['entry_price']} stop {t['stop']} target {t['target']} "
              f"(forecast {t.get('forecast_5d_pct')}%, {t.get('catalyst')}) -> "
              + (f"{t['exit_time']} exit {t['exit_price']} by {t['exit_reason']}, P&L ${t['pnl']} ({(t['pnl_pct'] or 0) * 100:+.1f}%)"
                 if t["status"] == "closed" else "still open")
              + (f"; post-mortem [{t['post_mortem'].get('mistake_type')}]: {t['post_mortem'].get('lesson')}" if t.get("post_mortem") else "")
              for t in trades]
        parts.append(f"## {sym}: journal entries (newest first)\n" + ("\n".join(lines) or "(none)") + f"\n## {sym}: trades\n" + ("\n".join(tl) or "(none)"))
        # what the news agent actually read, for the most recent entry only
        if ents and ents[0].get("prompt_sent"):
            parts.append(f"## {sym}: text the news agent read on {ents[0].get('logged')}\n" + _trim(ents[0]["prompt_sent"], 2500))

    # --- system-wide
    ct = j.closed_trades()
    if month:
        ct = [t for t in ct if str(t.get("exit_time", ""))[:7] == month]
    parts.append("## All trades (closed)\n" + ("\n".join(
        f"- {t['symbol']} {t['account']} {t['entry_time'][:10]}->{str(t['exit_time'])[:10]} {t['exit_reason']} {(t['pnl_pct'] or 0) * 100:+.1f}% ${t['pnl']}"
        + (f" [{t['post_mortem'].get('mistake_type')}] {t['post_mortem'].get('lesson')}" if t.get("post_mortem") else "")
        for t in ct[-40:]) or "(none)"))
    parts.append("## Open trades\n" + ("\n".join(f"- {t['symbol']} {t['account']} since {t['entry_time']} @ {t['entry_price']} stop {t['stop']} target {t['target']}"
                                                  for t in j.open_trades()) or "(none)"))
    parts.append("## Trade summary\n" + j.summary())

    sc = s.state_dir / "scorecard.csv"
    if sc.exists():
        df = pd.read_csv(sc)
        if month:
            df = df[df["logged"].astype(str).str[:7] == month]
        if len(df):
            cols = [c for c in ("logged", "account", "symbol", "rank", "expected_5d_pct", "confidence", "approved", "acted", "vetoed", "ret_5d") if c in df]
            recent = df[cols].tail(60)
            parts.append("## Scorecard (last 60 rows; ret_5d = realized 5-day return, blank = not yet known)\n" + recent.to_string(index=False))
            d = df.dropna(subset=["ret_5d"]) if "ret_5d" in df else df.iloc[0:0]
            if len(d) >= 5:
                app = d[d["approved"].astype(str).str.lower() == "true"]["ret_5d"]
                rej = d[d["approved"].astype(str).str.lower() != "true"]["ret_5d"]
                parts.append(f"## Scorecard totals: n={len(d)} with realized returns; approved mean 5d {app.mean() * 100:+.2f}% (n={len(app)}), "
                             f"not approved mean 5d {rej.mean() * 100:+.2f}% (n={len(rej)})")
    cyc = sorted((s.state_dir / "cycles").glob("*.json"))[-6:] if (s.state_dir / "cycles").exists() else []
    if cyc:
        lines = []
        for p in cyc:
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
                lines.append(f"- {p.stem}: equity ${c.get('equity'):,.0f} held {c.get('held')} candidates {c.get('candidates')}; "
                             f"{(c.get('decision') or {}).get('market_note')}; vetoed {(c.get('decision') or {}).get('vetoed')}; "
                             f"regime {(c.get('summary') or {}).get('regime', {}).get('label')}; log: {' | '.join(c.get('log') or [])[:600]}")
            except Exception:
                continue
        parts.append("## Recent cycles\n" + "\n".join(lines))
    lp = s.state_dir / "lessons.jsonl"
    if lp.exists():
        parts.append("## Lessons (post-mortems)\n" + _trim(lp.read_text(encoding="utf-8"), 4000))
    eq = s.state_dir / "equity_history.csv"
    if eq.exists():
        parts.append("## Equity history (last 20 rows)\n" + pd.read_csv(eq).tail(20).to_string(index=False))
    rep = sorted(s.reports_dir.glob("eval_*.md"))
    if rep:
        parts.append(f"## Latest eval report ({rep[-1].name})\n" + _trim(rep[-1].read_text(encoding="utf-8"), 5000))
    cp = s.state_dir / "costs.json"
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        parts.append(f"## Claude spend: total ${sum(v['usd'] for v in c.values()):.3f} over {len(c)} days")
    return _trim("\n\n".join(parts), MAX_CHARS)


def ask(s: Settings, agents: Agents, question: str, month: str | None = None) -> str:
    ctx = gather(s, question, month)
    r = agents.client.messages.create(model=agents.cfg["research_model"], max_tokens=700, system=ASK_SYSTEM,
                                      messages=[{"role": "user", "content": f"{ctx}\n\n## Question\n{question}"}])
    with agents.lock:
        agents.costs.record(agents.cfg["research_model"], r.usage.input_tokens, r.usage.output_tokens)
    agents._bump("calls", model=agents.cfg["research_model"])
    return "".join(getattr(c, "text", "") for c in r.content).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--month", default=None, help="restrict trades/scorecard to YYYY-MM")
    args = ap.parse_args()
    s = Settings()
    rcfg = s.cfg["research"]
    agents = Agents(s.anthropic_key, rcfg, CostTracker(s.state_dir, s.cfg["pricing"], rcfg["daily_usd_cap"]),
                    cache_path=s.state_dir / "research_cache.json", names=s.names)
    if args.question:
        print(ask(s, agents, " ".join(args.question), args.month))
        return
    print("Ask the journal (blank line to quit)")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        if not agents.costs.can_spend():
            print("daily Claude cap reached"); break
        print(ask(s, agents, q, args.month))


if __name__ == "__main__":
    main()
