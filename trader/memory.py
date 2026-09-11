"""Memory + post-mortem agents (v3).

memory     Before forecasting a name we have seen before, show Haiku what we said about it last time(s),
           what actually happened, and any lessons from closed trades in it. It answers whether the
           standing thesis is intact or broken and may nudge the forecast a little. This is the only agent
           with state across cycles; the eval's ablation measures whether it helps at all.
post_mortem After a trade closes (daily_summary.py), Haiku reads the entry reasoning and the outcome and
           writes a two-line lesson tagged with a mistake type. Lessons feed the memory agent and the
           monthly report. Cheap (one call per closed trade).
Neither agent sees unrealized P&L of open positions.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

MEMORY_SYSTEM = (
    "You are reviewing an analyst's PAST calls on one anonymized stock before a new forecast is made. "
    "You see prior forecasts (expected 5-day return, confidence, stated catalyst), what the stock actually did "
    "over the following 5 days, and lessons from closed trades. Decide whether the standing thesis is INTACT "
    "(the catalyst played out or is still pending), BROKEN (the catalyst failed or reversed, or the analyst was "
    "repeatedly wrong on this name), or NONE (no usable thesis). Output ONLY compact JSON: "
    "{\"thesis_status\": \"intact|broken|none\", \"adjustment_pct\": <-0.5..0.5, nudge for the new forecast>, "
    "\"note\": \"<=20 words\"}. Be conservative: most past calls are noise; 'broken' needs a clear failed catalyst."
)

POST_MORTEM_SYSTEM = (
    "You write a two-line post-mortem for a closed swing trade (anonymized stock). You see the entry reasoning, "
    "the analyst forecast and catalyst, the risk noted, the exit reason (stop loss / take profit / time stop / sold) "
    "and the result. Classify the main cause and write a lesson that is actionable for the NEXT similar trade. "
    "Output ONLY compact JSON: {\"mistake_type\": \"thesis_wrong|timing|market_wide|noise|stop_too_tight|none\", "
    "\"avoidable\": true|false, \"lesson\": \"<=30 words\"}. A winning trade can still contain a mistake; a losing "
    "trade on a sound thesis is 'noise' or 'market_wide', not a mistake."
)


class MemoryAgent:
    def __init__(self, agents, journal, state_dir: Path, cfg: dict):
        self.agents = agents
        self.journal = journal
        self.cfg = cfg
        self.lessons_path = state_dir / "lessons.jsonl"
        self.scorecard_path = state_dir / "scorecard.csv"
        self._sc: pd.DataFrame | None = None

    # ---- history assembly (deterministic) ----
    def _realized(self, symbol: str, day: str) -> float | None:
        if self._sc is None:
            self._sc = pd.read_csv(self.scorecard_path) if self.scorecard_path.exists() else pd.DataFrame()
        if self._sc.empty or "ret_5d" not in self._sc:
            return None
        m = self._sc[(self._sc["symbol"] == symbol) & (self._sc["logged"].astype(str).str[:10] == day)]
        v = pd.to_numeric(m["ret_5d"], errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else None

    def history(self, symbol: str) -> list[str]:
        """Compact, anonymized lines: prior forecasts + realized, closed trades + lessons."""
        lines, seen_days = [], set()
        for e in self.journal.entries_for(symbol, limit=200):
            day = e.get("logged", "")[:10]
            f = e.get("forecast") or {}
            if not f or day in seen_days:
                continue
            seen_days.add(day)
            real = self._realized(symbol, day)
            real_s = f"{real * 100:+.1f}%" if real is not None else "pending"
            lines.append(f"- {day}: forecast {float(f.get('expected_5d_return_pct', 0)):+.1f}% conf {float(f.get('confidence', 0)):.2f}; "
                         f"catalyst: {f.get('catalyst', '')}; realized 5d: {real_s}")
            if len(lines) >= self.cfg.get("max_history", 6):
                break
        for t in self.journal.closed_trades():
            if t["symbol"] != symbol:
                continue
            pm = t.get("post_mortem") or {}
            lines.append(f"- trade {t['entry_time'][:10]} -> {str(t.get('exit_time'))[:10]}: {t.get('exit_reason')} "
                         f"{float(t.get('pnl_pct') or 0) * 100:+.1f}%; lesson: {pm.get('lesson', '(none yet)')}")
        return lines[: self.cfg.get("max_history", 6) + 4]

    def review(self, symbol: str) -> dict | None:
        hist = self.history(symbol)
        if not hist:
            return {"symbol": symbol, "thesis_status": "none", "adjustment_pct": 0.0, "note": "no history", "_cached": True, "_cost_usd": 0.0}
        user = "Past calls on the company (most recent first):\n" + "\n".join(hist)
        out = self.agents._call(self.agents.cfg["research_model"], MEMORY_SYSTEM, user, 200)
        if not out:
            return None
        st = str(out.get("thesis_status", "none")).lower()
        out["thesis_status"] = st if st in ("intact", "broken", "none") else "none"
        try:
            out["adjustment_pct"] = max(-0.5, min(0.5, float(out.get("adjustment_pct", 0))))
        except (TypeError, ValueError):
            out["adjustment_pct"] = 0.0
        out["symbol"] = symbol
        out["history_n"] = len(hist)
        return out


class PostMortemAgent:
    def __init__(self, agents, journal, state_dir: Path):
        self.agents = agents
        self.journal = journal
        self.lessons_path = state_dir / "lessons.jsonl"

    def write(self, trade: dict) -> dict | None:
        """Runs once per closed trade; stores the result on the trade record and in lessons.jsonl."""
        if trade.get("post_mortem") or trade.get("status") != "closed":
            return trade.get("post_mortem")
        snippet = ""
        jf = trade.get("journal_file")
        if jf and Path(jf).exists():
            try:
                j = json.loads(Path(jf).read_text(encoding="utf-8"))
                snippet = (j.get("prompt_sent") or "")[-700:]
                f = j.get("forecast") or {}
                risk = f.get("risk", "")
            except Exception:
                risk = ""
        else:
            risk = ""
        days = None
        try:
            days = (datetime.strptime(trade["exit_time"], "%Y-%m-%d %H:%M") - datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M")).days
        except Exception:
            pass
        user = (f"Entry reason: {trade.get('entry_reason')}\nForecast: {trade.get('forecast_5d_pct')}% conf {trade.get('confidence')}; "
                f"catalyst: {trade.get('catalyst')}; risk noted: {risk}\n"
                f"Stop {((trade['stop'] / trade['entry_price']) - 1) * 100:+.1f}%, target {((trade['target'] / trade['entry_price']) - 1) * 100:+.1f}% from entry.\n"
                f"Outcome: exit by {trade.get('exit_reason')} after {days} days, result {float(trade.get('pnl_pct') or 0) * 100:+.1f}%.\n"
                + (f"\nWhat the analyst read at entry (tail):\n{snippet}" if snippet else ""))
        out = self.agents._call(self.agents.cfg["research_model"], POST_MORTEM_SYSTEM, user, 200)
        if not out:
            return None
        pm = {"mistake_type": str(out.get("mistake_type", "none")), "avoidable": bool(out.get("avoidable", False)),
              "lesson": str(out.get("lesson", ""))[:200], "written": datetime.now().strftime("%Y-%m-%d %H:%M")}
        trade["post_mortem"] = pm
        self.journal._save()
        with open(self.lessons_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"trade_id": trade["id"], "symbol": trade["symbol"], "account": trade["account"],
                                "exit_reason": trade.get("exit_reason"), "pnl_pct": trade.get("pnl_pct"), **pm}) + "\n")
        return pm
