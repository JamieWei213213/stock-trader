# Changelog

Dated entries; `monthly_report.py` pulls the ones from the current month into section 4.

## 2026-09-13 — v3: multi-agent orchestration, veto rule, regime filter
- `trader/orchestrator.py`: agents run concurrently (asyncio + thread pool), typed per-cycle state, per-call
  timeout + one retry, graceful degradation (a failed agent = no forecast = veto, never a crash), every agent's
  I/O journaled, per-cycle record in `state/cycles/`.
- New LLM agents: **filings** (SEC EDGAR XBRL, last 8 quarters, anonymized, cached 90 days; Kim/Muhn/Nikolaev setup),
  **memory** (the symbol's own past forecasts/outcomes/lessons -> thesis intact/broken), **post-mortem**
  (two-line lesson per closed trade -> `state/lessons.jsonl`). Critic agent exists only as a config flag (off).
- Deterministic nodes: **regime monitor** (SPY vs 200-day, vol spike -> halve risk / freeze buys),
  **risk reviewer** (industry cap, correlation with the book), **decision rule** (`trader/decide.py`).
- Sonnet portfolio manager removed from the default path (`decision.mode: manager` keeps it for A/B).
  No model ever sees P&L any more.
- Large account: 15 candidates, 10 positions, heat cap 8%. 6:05am cycle is manage-only (`--mode manage`).
- Stop-distance cap 8% of price; earnings blackout >= max_hold_days ahead and 1 day behind.
- `eval_agents.py` sections 6-8: veto accuracy, per-agent ablation, fleet health. `monthly_report.py` added.
- `build_universe.py` tags industries via Finnhub (`state/industries.json`) for the risk reviewer.
- `ask.py`: 'ask the journal' — plain-English questions answered from the system's own records (journal, trades,
  scorecard, cycles, lessons, eval); read-only, cost-tracked.
- Middle-path exits (large account, `hold:` block): trailing stop that ratchets the broker stop, thesis check from the
  review day, stale-slot rotation from day 10, hard ceiling at day 30. Small account keeps the hard 10-day stop.
  New exit reasons: `trail_stop`, `stale`, `hard_max`.
- Kill switch (`trader/killswitch.py`, `killswitch.py`): daily-loss and drawdown freezes on new buys; stops stay live.
- `mcp_journal.py`: read-only MCP server (7 tools) so Claude Desktop can query the journal, ledger, scorecard, cycles,
  costs and kill-switch state.
- `propose.py`: plain-English screener idea -> LLM writes a strict spec -> fair backtest judges it; proposals logged
  with a multiple-testing count.

## 2026-09-06 — v2 hotfix
- Held names are always re-forecast even when the screener rule excludes them (`keep=`).

## 2026-09-03 — v2
- Fair backtest (point-in-time universe, hold-out year, costs, t>=3 bar); evidence-based screener variants;
  ATR-scaled risk with bracket orders; anonymized article bodies; earnings blackout; decision journal +
  trades ledger; `eval_agents.py`; dashboard with candles, SPY benchmark and trade history.

## 2026-08-30 — v1
- First live paper cycles: attention screener, Haiku conviction scores, Sonnet manager, fixed-fraction sizing.
