"""Claude research agents (v2 news agent; v3 adds filings/memory/post-mortem in filings.py and memory.py).

- research_stock(): one cheap Haiku call per candidate -> numeric forecast JSON.
  v2 changes backed by RESEARCH.md §1: the company name and ticker are hidden from the model
  (Glasserman & Lin: anonymized headlines predict better), article bodies are passed rather than
  headlines alone, the model is told not to extrapolate last week's move, and it outputs an
  expected 5-day return + confidence instead of a 1-5 "conviction" so we can score it later.
- rank_and_decide(): one Sonnet call per cycle -> buy/sell/hold list.

Every call is logged through CostTracker; a 3-hour cache skips re-analysis when nothing changed.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time

import anthropic

from .costs import CostTracker

RESEARCH_SYSTEM = (
    "You are a disciplined sell-side analyst producing a short-horizon forecast for ONE US stock. "
    "The company is anonymized as 'the company'; judge only the information given, not what you may "
    "remember about the firm. Base rates: over 5 trading days a large-cap stock moves about +/-3%; most "
    "news is already priced in by the time it is published; last week's return has little predictive "
    "value and should NOT be extrapolated. Output ONLY compact JSON: "
    "{\"expected_5d_return_pct\": <number, e.g. -1.5>, \"confidence\": <0.0-1.0>, "
    "\"bias\": \"bullish|bearish|neutral\", \"catalyst\": \"<=15 words\", \"risk\": \"<=15 words\"}. "
    "A forecast near 0 with low confidence is the correct answer when the news is routine."
)

RANK_SYSTEM = (
    "You are the portfolio manager for a small experimental paper-trading account. You receive analyst "
    "forecasts (expected 5-day return %, confidence), current positions, and account rules. "
    "Output ONLY JSON: {\"actions\":[{\"symbol\":..,\"action\":\"buy|sell|hold\",\"reason\":\"<=20 words\"}],"
    "\"market_note\":\"<=25 words\"}. Rules: 'buy' only when expected_5d_return_pct >= min_expected_return_pct "
    "AND confidence >= min_confidence; 'sell' held names when the forecast turns negative or the catalyst is gone; "
    "otherwise hold. Fewer, better trades. Never exceed max_positions. Longs only. Staying in cash is fine."
)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def anonymize(text: str, symbol: str, name: str | None) -> str:
    """Replace ticker and company name (and its first word, e.g. 'Robinhood' in 'Robinhood Markets Inc') ."""
    out = re.sub(rf"\b\$?{re.escape(symbol)}\b", "the company", text, flags=re.I)
    if name:
        clean = re.sub(r"\b(Inc|Corp|Corporation|Co|Ltd|PLC|Holdings|Group|Class [A-C]|Common Stock)\b\.?", "", name, flags=re.I)
        clean = re.sub(r"[,.]", "", clean).strip()
        if clean:
            out = re.sub(re.escape(clean), "the company", out, flags=re.I)
            first = clean.split()[0]
            if len(first) >= 4:
                out = re.sub(rf"\b{re.escape(first)}(?:'s)?\b", "the company", out, flags=re.I)
    return out


class Agents:
    def __init__(self, api_key: str, research_cfg: dict, costs: CostTracker, cache_path=None, names: dict | None = None):
        # SDK timeout < orchestrator timeout (60s) so a hung call really ends instead of running on in its thread
        self.client = anthropic.Anthropic(api_key=api_key, timeout=float(research_cfg.get("call_timeout_s", 45)), max_retries=1)
        self.cfg = research_cfg
        self.costs = costs
        self.names = names or {}
        self.cache_path = cache_path
        self.stats_path = (cache_path.parent / "agent_stats.json") if cache_path else None
        self.stats: dict = {"calls": 0, "parse_failures": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0,
                            "by_model": {}}
        if self.stats_path and self.stats_path.exists():
            try:
                self.stats.update(json.loads(self.stats_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        self.lock = threading.Lock()   # v3: agents run concurrently; stats/cache/cost files need a lock
        self.cache: dict = {}
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}

    # ---- cache ----
    def _cache_get(self, symbol: str, key: str) -> dict | None:
        hit = self.cache.get(symbol)
        ttl = self.cfg.get("cache_hours", 0) * 3600
        if hit and hit.get("key") == key and time.time() - hit.get("t", 0) < ttl:
            self._bump("cache_hits")
            out = dict(hit["result"])
            out["_cost_usd"] = 0.0
            out["_cached"] = True
            return out
        return None

    def _cache_put(self, symbol: str, key: str, result: dict):
        with self.lock:
            self.cache[symbol] = {"key": key, "t": time.time(),
                                  "result": {k: v for k, v in result.items() if not k.startswith("_")}}
            if self.cache_path:
                self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def _bump(self, key: str, n: int = 1, model: str | None = None):
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + n
            if model:
                m = self.stats["by_model"].setdefault(model, {"calls": 0, "parse_failures": 0})
                if key in m:
                    m[key] += n
            if self.stats_path:
                self.stats_path.write_text(json.dumps(self.stats), encoding="utf-8")

    # ---- calls ----
    def _call(self, model: str, system: str, user: str, max_tokens: int) -> dict | None:
        if not self.costs.can_spend():
            print(f"  [costs] daily cap ${self.costs.cap:.2f} reached — skipping Claude call")
            return None
        r = self.client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}])
        with self.lock:
            usd = self.costs.record(model, r.usage.input_tokens, r.usage.output_tokens)
            self.stats["input_tokens"] += r.usage.input_tokens
            self.stats["output_tokens"] += r.usage.output_tokens
        self._bump("calls", model=model)
        text = "".join(getattr(c, "text", "") for c in r.content)
        out = _extract_json(text)
        if out is None:
            self._bump("parse_failures", model=model)
            print(f"  [agents] could not parse JSON from {model}: {text[:120]!r}")
        if out is not None:
            out["_cost_usd"] = round(usd, 5)
            out["_prompt"] = user
            out["_raw"] = text
        return out

    def research_stock(self, symbol: str, feats: dict, news: list[dict]) -> dict | None:
        name = self.names.get(symbol)
        stats = (
            f"The company: price {feats['price']:.2f}; returns 1d {feats['ret_1d']*100:+.1f}%, 5d {feats['ret_5d']*100:+.1f}%, "
            f"20d {feats['ret_20d']*100:+.1f}%, 12-month {feats.get('mom_12_2', 0)*100:+.0f}% (ex last month); "
            f"volume {feats['vol_surge']:.1f}x average; typical daily range {feats['atr_pct']*100:.1f}%; "
            f"{feats['dist_20d_high']*100:+.1f}% from 20-day high."
        )
        items = []
        for n in news[: self.cfg.get("news_items_per_stock", 6)]:
            body = (n.get("content") or n.get("summary") or "")[: self.cfg.get("article_chars", 500)]
            items.append(anonymize(f"- [{n['time']}] {n['headline']}. {body}", symbol, name))
        news_block = "\n".join(items) or "- (no news in the window)"
        bucket = round(math.log(max(feats["price"], 0.01)) / math.log(1.03))
        key = f"v2|{bucket}|" + "|".join(sorted(n["headline"] for n in news))
        cached = self._cache_get(symbol, key)
        if cached:
            return cached
        user = f"{stats}\n\nRecent news (anonymized):\n{news_block}"
        out = self._call(self.cfg["research_model"], RESEARCH_SYSTEM, user, self.cfg["research_max_tokens"])
        if out:
            out["symbol"] = symbol
            try:
                out["expected_5d_return_pct"] = float(out.get("expected_5d_return_pct", 0))
                out["confidence"] = min(max(float(out.get("confidence", 0)), 0.0), 1.0)
            except (TypeError, ValueError):
                out["expected_5d_return_pct"], out["confidence"] = 0.0, 0.0
            self._cache_put(symbol, key, out)
        return out

    def rank_and_decide(self, account_name: str, rules: dict, snapshot: dict, research: list[dict]) -> dict | None:
        user = json.dumps(
            {
                "account": account_name,
                "rules": {
                    "max_positions": rules["max_positions"], "max_hold_days": rules["max_hold_days"],
                    "min_expected_return_pct": self.cfg.get("min_expected_return_pct", 1.5),
                    "min_confidence": self.cfg.get("min_confidence", 0.6),
                },
                "equity": snapshot["equity"], "cash": snapshot["cash"],
                # v3: never show P&L to a model (disposition effect); symbol + size only
                "positions": [{"symbol": p["symbol"], "qty": p["qty"]} for p in snapshot["positions"]],
                "analyst_forecasts": [{k: v for k, v in r.items() if not k.startswith("_")} for r in research],
            },
            separators=(",", ":"),
        )
        return self._call(self.cfg["ranking_model"], RANK_SYSTEM, user, self.cfg["ranking_max_tokens"])
