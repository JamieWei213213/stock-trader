"""v3 orchestrator: runs the agents concurrently over a typed per-cycle state and combines their outputs.

Nodes in one cycle (see README "Architecture v3"):
  screener (deterministic)  -> candidates
  regime monitor (determ.)  -> risk multiplier / buy freeze
  news agent (LLM)          -> expected_5d_return_pct, confidence, bias, catalyst, risk      [per candidate]
  filings agent (LLM)       -> earnings_direction, confidence -> tilt_pct                     [per candidate, cached 90d]
  memory agent (LLM)        -> thesis_status, adjustment_pct                                  [per candidate with history]
  forecaster (deterministic)-> combined forecast = news + filings tilt + memory nudge
  decision rule (determ.)   -> buy / hold / sell / swap; agents can only veto
  risk reviewer (determ.)   -> industry concentration + correlation-with-book checks
  executor (deterministic)  -> orders

Concurrency: asyncio + a thread pool for the blocking SDK calls, a semaphore to stay polite with the
APIs, a per-call timeout and one retry. A failed agent degrades the forecast (lower confidence or no
forecast -> veto) and is recorded in state.agent_log; it never crashes the cycle. Every agent's input
and output is journaled so the eval can run an ablation (news-only vs combined vs each agent).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import pandas as pd

from .decide import decide, veto_reason
from .regime import RegimeView


# ------------------------------------------------------------------------------------------ typed state
@dataclass
class Candidate:
    symbol: str
    rank: int
    features: dict
    held: bool = False


@dataclass
class AgentResult:
    agent: str
    symbol: str
    ok: bool
    seconds: float
    cached: bool = False
    cost_usd: float = 0.0
    error: str | None = None
    attempts: int = 1


@dataclass
class CycleState:
    account: str
    mode: str                                   # "entry" | "manage"
    snapshot: dict
    regime: RegimeView
    candidates: list[Candidate]
    news: dict[str, dict] = field(default_factory=dict)
    filings: dict[str, dict] = field(default_factory=dict)
    memory: dict[str, dict] = field(default_factory=dict)
    forecasts: dict[str, dict] = field(default_factory=dict)   # combined, keyed by symbol
    agent_log: list[AgentResult] = field(default_factory=list)
    decision: dict | None = None
    risk_review: dict[str, str] = field(default_factory=dict)  # symbol -> why the reviewer dropped a buy
    notes: list[str] = field(default_factory=list)

    @property
    def held(self) -> list[str]:
        return [c.symbol for c in self.candidates if c.held]

    def summary(self) -> dict:
        by_agent: dict[str, dict] = {}
        for r in self.agent_log:
            a = by_agent.setdefault(r.agent, {"calls": 0, "ok": 0, "cached": 0, "failed": 0, "usd": 0.0, "seconds": 0.0})
            a["calls"] += 1; a["ok"] += r.ok; a["cached"] += r.cached; a["failed"] += (not r.ok)
            a["usd"] += r.cost_usd; a["seconds"] = round(a["seconds"] + r.seconds, 1)
        return {"agents": by_agent, "regime": self.regime.as_dict(), "notes": self.notes,
                "risk_review": self.risk_review, "mode": self.mode}


# ------------------------------------------------------------------------------------------ forecaster
def combine(news: dict | None, filings: dict | None, memory: dict | None) -> dict | None:
    """Deterministic combination. News is the primary forecast; filings tilt it; memory nudges it and can
    flag a broken thesis. Every component is kept on the result so the eval can ablate."""
    if not news:
        return None
    exp = float(news.get("expected_5d_return_pct", 0))
    conf = float(news.get("confidence", 0))
    out = {"symbol": news.get("symbol"), "news_exp": exp, "news_conf": conf, "bias": news.get("bias"),
           "catalyst": news.get("catalyst"), "risk": news.get("risk")}
    tilt = 0.0
    if filings and not filings.get("error"):
        tilt = float(filings.get("tilt_pct", 0))
        out.update({"filings_dir": filings.get("earnings_direction"), "filings_conf": filings.get("confidence"),
                    "filings_tilt": tilt, "filings_flags": filings.get("quality_flags")})
        if tilt and exp:
            conf += 0.05 if np.sign(tilt) == np.sign(exp) else -0.10
    adj = 0.0
    if memory:
        adj = float(memory.get("adjustment_pct", 0))
        out.update({"thesis_status": memory.get("thesis_status", "none"), "memory_adj": adj,
                    "memory_note": memory.get("note", "")})
    exp_c = exp + tilt + adj
    out["expected_5d_return_pct"] = round(exp_c, 2)
    out["confidence"] = round(min(max(conf, 0.0), 1.0), 2)
    if (exp_c > 0.5) != (exp > 0.5) or (exp_c < -0.5) != (exp < -0.5):
        out["bias"] = "bullish" if exp_c > 0.5 else "bearish" if exp_c < -0.5 else "neutral"
    return out


# ------------------------------------------------------------------------------------------ risk reviewer
def risk_review(actions: list[dict], held: list[str], industries: dict[str, str], close: pd.DataFrame | None,
                rcfg: dict) -> tuple[list[dict], dict[str, str]]:
    """Drop buys that would concentrate the book. Deterministic; explains every drop."""
    max_ind = int(rcfg.get("max_per_industry", 3))
    max_corr = float(rcfg.get("max_corr_with_book", 0.75))
    book = list(held)
    dropped: dict[str, str] = {}
    kept = []
    rets = None
    if close is not None and len(close) > 30:
        rets = close.iloc[-60:].pct_change().dropna(how="all")
    for a in actions:
        if a.get("action") != "buy":
            kept.append(a); continue
        sym = a["symbol"]
        ind = industries.get(sym)
        if ind:
            n_same = sum(1 for h in book if industries.get(h) == ind)
            if n_same >= max_ind:
                dropped[sym] = f"industry cap: already {n_same} in '{ind}'"
                continue
        if rets is not None and sym in rets.columns:
            peers = [h for h in book if h in rets.columns]
            if len(peers) >= 2:
                cors = rets[peers].corrwith(rets[sym]).dropna()
                hi = cors[cors > max_corr]
                if len(hi) >= 2:
                    dropped[sym] = f"correlation: {len(hi)} held names > {max_corr:.2f} ({', '.join(f'{k} {v:.2f}' for k, v in hi.items())})"
                    continue
        kept.append(a)
        book.append(sym)
    return kept, dropped


# ------------------------------------------------------------------------------------------ orchestrator
class Orchestrator:
    def __init__(self, agents, cfg: dict, broker=None, filings_agent=None, memory_agent=None, news_fetch: Callable | None = None,
                 rcfg: dict | None = None):
        """agents: trader.agents.Agents; cfg: config['agents']; news_fetch(symbol) -> list[news dict]."""
        self.agents = agents
        self.cfg = cfg
        self.rcfg = rcfg or {}
        self.filings = filings_agent if cfg.get("filings", {}).get("enabled") else None
        self.memory = memory_agent if cfg.get("memory", {}).get("enabled") else None
        self.news_fetch = news_fetch or (lambda sym: broker.recent_news(sym, self.rcfg.get("news_lookback_hours", 36),
                                                                          self.rcfg.get("news_items_per_stock", 6)))
        self.sem = asyncio.Semaphore(int(cfg.get("max_concurrency", 6)))

    async def _guarded(self, state: CycleState, agent: str, symbol: str, fn: Callable) -> dict | None:
        timeout = float(self.cfg.get("timeout_s", 60))
        retries = int(self.cfg.get("retries", 1))
        t0, err = time.time(), None
        for attempt in range(1, retries + 2):
            try:
                async with self.sem:
                    out = await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
                ok = out is not None
                state.agent_log.append(AgentResult(agent, symbol, ok, round(time.time() - t0, 2),
                                                   bool(out and out.get("_cached")), float((out or {}).get("_cost_usd", 0) or 0),
                                                   None if ok else "returned None (cost cap or unparseable)", attempt))
                return out
            except asyncio.TimeoutError:
                err = f"timeout after {timeout:.0f}s"
            except Exception as e:  # noqa: BLE001 — any agent failure degrades, never crashes
                err = f"{type(e).__name__}: {e}"[:200]
            if attempt <= retries:
                await asyncio.sleep(1.0)
        state.agent_log.append(AgentResult(agent, symbol, False, round(time.time() - t0, 2), error=err, attempts=retries + 1))
        return None

    async def _one(self, state: CycleState, c: Candidate):
        sym = c.symbol
        tasks = {}
        if self.cfg.get("news", {}).get("enabled", True):
            def news_job():
                news = self.news_fetch(sym)
                return self.agents.research_stock(sym, c.features, news)
            tasks["news"] = self._guarded(state, "news", sym, news_job)
        if self.filings is not None:
            tasks["filings"] = self._guarded(state, "filings", sym, lambda: self.filings.analyze(sym))
        if self.memory is not None:
            tasks["memory"] = self._guarded(state, "memory", sym, lambda: self.memory.review(sym))
        results = await asyncio.gather(*tasks.values())
        for k, r in zip(tasks, results):
            if r is not None:
                getattr(state, k)[sym] = r
        f = combine(state.news.get(sym), state.filings.get(sym), state.memory.get(sym))
        if f:
            state.forecasts[sym] = f

    async def run_agents(self, state: CycleState):
        await asyncio.gather(*(self._one(state, c) for c in state.candidates))

    def run(self, state: CycleState) -> CycleState:
        asyncio.run(self.run_agents(state))
        return state

    # ---- decision + review (deterministic) ----
    def decide(self, state: CycleState, rules: dict, dcfg: dict, rcfg: dict, industries: dict[str, str],
               close: pd.DataFrame | None, risk_rules: dict, days_held: dict | None = None, stale: set | None = None) -> dict:
        cands = [c.symbol for c in state.candidates if not c.held]
        d = decide(cands, state.forecasts, state.held, rules, dcfg, rcfg, state.regime, days_held=days_held, stale=stale)
        if state.mode == "manage":
            d["actions"] = [a for a in d["actions"] if a["action"] != "buy"]
            for sym in cands:
                d["vetoed"].setdefault(sym, "manage-only cycle")
        kept, dropped = risk_review(d["actions"], state.held, industries, close, risk_rules)
        for old, new in d.get("swaps", []):      # a swap whose buy was dropped must not sell the old name for nothing
            if new in dropped:
                for a in kept:
                    if a["symbol"] == old and a["action"] == "sell":
                        a.update({"action": "hold", "reason": f"kept: swap partner {new} dropped by risk reviewer"})
        d["actions"] = kept
        for sym, why in dropped.items():
            d["vetoed"][sym] = f"risk reviewer: {why}"
        state.risk_review = dropped
        state.decision = d
        return d


def build_candidates(table: pd.DataFrame, cands: list[str], held: list[str]) -> list[Candidate]:
    out = []
    for rank, sym in enumerate(cands, 1):
        if sym not in table.index:
            continue
        feats = {k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v)
                 for k, v in table.loc[sym].to_dict().items()}
        out.append(Candidate(sym, rank, feats, sym in held))
    return out


def state_to_json(state: CycleState) -> dict:
    d = asdict(state)
    d["regime"] = state.regime.as_dict()
    return d
