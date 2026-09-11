"""One full cycle for one account (v3): screen -> regime -> agents (concurrent) -> decide -> review -> execute -> journal.

Usage:
  python run_cycle.py --account small                 # entry cycle (cron: 12:45pm PT)
  python run_cycle.py --account large --mode manage   # 6:05am look: sells / time stops only, no new buys
  python run_cycle.py --account large --dry-run       # everything except placing orders
  python run_cycle.py --account large --force         # run even if market is closed (testing)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from trader.agents import Agents
from trader.broker import Broker
from trader.costs import CostTracker
from trader.earnings import EarningsCalendar
from trader.executor import Executor
from trader.filings import FilingsAgent
from trader.journal import Journal
from trader.killswitch import KillSwitch
from trader.memory import MemoryAgent
from trader.orchestrator import CycleState, Orchestrator, build_candidates
from trader.regime import assess_live
from trader.scorecard import Scorecard
from trader.screener import pick_candidates, screen
from trader.settings import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", choices=["small", "large"], required=True)
    ap.add_argument("--mode", choices=["entry", "manage"], default="entry",
                    help="manage = only manage existing positions (no new buys); used for the 6:05am cycle")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even when market is closed")
    args = ap.parse_args()

    s = Settings()
    rules = s.account_cfg(args.account)
    rcfg, dcfg, acfg = s.cfg["research"], s.cfg.get("decision", {"mode": "veto"}), s.cfg.get("agents", {})
    broker = Broker(s.creds(args.account))
    costs = CostTracker(s.state_dir, s.cfg["pricing"], rcfg["daily_usd_cap"])
    agents = Agents(s.anthropic_key, rcfg, costs, cache_path=s.state_dir / "research_cache.json", names=s.names)
    journal = Journal(s.state_dir)
    now = datetime.now()
    print(f"=== {now:%Y-%m-%d %H:%M} v3 cycle '{args.account}' mode={args.mode} dry_run={args.dry_run} decision={dcfg.get('mode')} ===")

    if not broker.market_open() and not args.force:
        clock = broker.trading.get_clock()
        if clock.next_open.astimezone().date() != now.date():
            print("market closed today — skipping (use --force to override)")
            return

    snap = broker.snapshot()
    held = [p["symbol"] for p in snap.positions]
    print(f"equity ${snap.equity:,.2f} cash ${snap.cash:,.2f} positions {held} daytrades {snap.daytrade_count}")

    # 0) kill switch (deterministic): a bad day or a deep drawdown freezes NEW buys; exits and broker stops still run
    tripped, why = KillSwitch(s.state_dir, args.account, s.cfg.get("killswitch", {})).check(snap.equity)
    if tripped:
        print(f"!! KILL SWITCH: {why}")
        if args.mode == "entry":
            args.mode = "manage"

    # 1) screener (deterministic). Held names are always included so they get re-forecast.
    table, bars = screen(s, broker, keep=held, return_bars=True)
    if table.empty:
        print("screener returned nothing — aborting cycle")
        return
    close = bars["close"].unstack("symbol").sort_index()

    # 2) regime monitor (deterministic)
    regime = assess_live(broker, s.cfg.get("regime", {}))
    print(f"regime: {regime.label} — {regime.note}")

    # 3) eligibility: earnings windows (forward >= hold period, plus recently reported) and affordability
    blackout = max(int(rcfg.get("earnings_blackout_days", 2)), int(rules["max_hold_days"]))
    earnings = EarningsCalendar(s.state_dir, blackout, recent_days=int(rcfg.get("earnings_recent_days", 0)))
    max_stop_pct = s.cfg.get("risk", {}).get("max_stop_pct")

    def eligible(sym) -> bool:
        if sym in held:
            return True
        if earnings.soon(sym):
            return False
        r = table.loc[sym]
        stop_dist = rules["atr_stop_mult"] * float(r["atr"])
        if max_stop_pct and stop_dist / float(r["price"]) > max_stop_pct:
            return False
        qty = min(snap.equity * rules["risk_per_trade_pct"] * regime.risk_mult // max(stop_dist, 1e-9),
                  snap.equity * rules["max_position_pct"] // float(r["price"]))
        return qty >= 1
    keep = [sym for sym in table.index if eligible(sym)]
    dropped = len(table) - len(keep)
    table = table.loc[keep]
    n_cands = 0 if args.mode == "manage" else s.candidates_per_cycle(args.account)
    cands = pick_candidates(table, n_cands, held) if n_cands else list(held)
    print(f"screened {len(table) + dropped} names ({dropped} skipped: earnings window / stop too wide / unaffordable), "
          f"variant={s.cfg['screener'].get('variant')}, earnings source={earnings.source}")
    print("candidates:", cands)

    # 4) agents, concurrently, over typed state
    state = CycleState(args.account, args.mode, asdict(snap), regime, build_candidates(table, cands, held))
    orch = Orchestrator(agents, acfg, broker=broker, rcfg=rcfg,
                        filings_agent=FilingsAgent(agents, s.state_dir, acfg.get("filings", {})),
                        memory_agent=MemoryAgent(agents, journal, s.state_dir, acfg.get("memory", {})))
    if state.candidates:
        orch.run(state)
    for c in state.candidates:
        f = state.forecasts.get(c.symbol)
        if not f:
            print(f"  {c.symbol}: no forecast"); continue
        parts = [f"news {f['news_exp']:+.1f}%/{f['news_conf']:.2f}"]
        if "filings_dir" in f:
            parts.append(f"filings {f['filings_dir']} tilt {f['filings_tilt']:+.2f}")
        if f.get("thesis_status") not in (None, "none"):
            parts.append(f"memory {f['thesis_status']} {f.get('memory_adj', 0):+.2f}")
        print(f"  {c.symbol}{' (held)' if c.held else ''}: {f['expected_5d_return_pct']:+.1f}% conf {f['confidence']:.2f} {f.get('bias')} "
              f"[{'; '.join(parts)}] — {f.get('catalyst')}")
    failed = [r for r in state.agent_log if not r.ok]
    if failed:
        print(f"  agent failures: " + ", ".join(f"{r.agent}/{r.symbol} ({r.error})" for r in failed))

    # 5) decision (deterministic rule + agent veto) and risk review; legacy Sonnet manager kept behind config
    ex = Executor(args.account, rules, broker, s.state_dir, s.is_volatile, dry_run=args.dry_run,
                  earnings_soon=earnings.soon, journal=journal, forecasts=state.forecasts,
                  risk_mult=regime.risk_mult, max_stop_pct=max_stop_pct, manage_only=(args.mode == "manage"))
    held_map = {p["symbol"]: p for p in snap.positions}
    stale = ex.stale_positions(held_map)
    days_held = {sym: ex.days_held(sym) for sym in held}
    if stale:
        print(f"stale (flat >= {rules.get('hold', {}).get('stale_days', '?')}d, slot available to a better pick): {sorted(stale)}")
    if dcfg.get("mode") == "manager":
        research = [dict(state.news[c.symbol], symbol=c.symbol) for c in state.candidates if c.symbol in state.news]
        decision = agents.rank_and_decide(args.account, rules, asdict(snap), research) if research else None
        if decision:
            decision.setdefault("vetoed", {}); decision["mode"] = "manager"
            if args.mode == "manage":
                decision["actions"] = [a for a in decision["actions"] if a.get("action") != "buy"]
        state.decision = decision
    else:
        decision = orch.decide(state, rules, dcfg, rcfg, s.industries, close, s.cfg.get("risk", {}),
                               days_held=days_held, stale=stale)
    if decision:
        print("decision:", json.dumps({k: v for k, v in decision.items() if k in ("actions", "market_note")}))
        if decision.get("vetoed"):
            print("vetoed:", "; ".join(f"{k}: {v}" for k, v in decision["vetoed"].items()))

    # 6) execute with guards (plain Python; the models cannot override)
    prices = {c.symbol: float(c.features["price"]) for c in state.candidates}
    atrs = {c.symbol: float(c.features["atr"]) for c in state.candidates}
    try:
        prices.update({k: v for k, v in broker.latest_prices(list(prices)).items() if v > 0})
    except Exception as e:
        print(f"  latest quotes unavailable ({e}); using last close")
    actions = {a.get("symbol"): a for a in (decision or {}).get("actions", [])}
    jfiles: dict[str, str] = {}
    if not args.dry_run:
        for c in state.candidates:
            f = state.forecasts.get(c.symbol, {})
            news = state.news.get(c.symbol, {})
            jfiles[c.symbol] = str(journal.write_entry(args.account, c.symbol, {
                "rank": c.rank, "held": c.held, "mode": args.mode, "features": c.features,
                "prompt_sent": news.get("_prompt"), "raw_reply": news.get("_raw"), "cached": bool(news.get("_cached")),
                "forecast": {k: v for k, v in f.items() if k != "symbol"},
                "agents": {"filings": {k: v for k, v in state.filings.get(c.symbol, {}).items() if not k.startswith("_")},
                           "memory": {k: v for k, v in state.memory.get(c.symbol, {}).items() if not k.startswith("_")}},
                "manager": actions.get(c.symbol) or ({"action": "veto", "reason": decision["vetoed"][c.symbol]}
                                                     if decision and c.symbol in decision.get("vetoed", {}) else None),
                "market_note": (decision or {}).get("market_note"), "regime": regime.as_dict(),
            }, now))
    ex.journal_files = jfiles
    log = ex.apply(decision, snap, prices, atrs)
    if not args.dry_run:
        for sym, o in ex.orders_placed.items():
            if sym in jfiles:
                pth = Path(jfiles[sym]); d = json.loads(pth.read_text(encoding="utf-8")); d["order"] = o
                pth.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")

    # 6b) confirm with the broker what actually happened
    if not args.dry_run and ex.orders_placed:
        try:
            pending = {o["symbol"]: o for o in broker.open_orders()}
            now_held = {p["symbol"] for p in broker.snapshot().positions}
            for sym, o in ex.orders_placed.items():
                st = "FILLED" if sym in now_held else ("pending at broker" if sym in pending else f"NOT FOUND ({o.get('status')})")
                print(f"  broker check {sym}: {st}"); log.append(f"broker check {sym}: {st}")
        except Exception as e:
            print(f"  broker check failed: {e}")

    # 7) scorecard with per-agent components (the eval ablates on these)
    if not args.dry_run and args.mode != "manage":
        bought = {l.split()[2] for l in log if l.startswith("BUY ")}
        rows = []
        for c in state.candidates:
            f = state.forecasts.get(c.symbol, {})
            exp, conf = f.get("expected_5d_return_pct"), f.get("confidence")
            if c.held:   # re-forecast of a position, not a pick: tagged so the eval keeps it out of both control groups
                approved, veto = (actions.get(c.symbol) or {}).get("action") != "sell", f"held: {(actions.get(c.symbol) or {}).get('action', 'hold')}"
            else:
                approved = bool(f) and c.symbol not in (decision or {}).get("vetoed", {}) or (c.symbol in bought)
                veto = (decision or {}).get("vetoed", {}).get(c.symbol, "")
            rows.append({"symbol": c.symbol, "rank": c.rank, "price": prices.get(c.symbol), "expected_5d_pct": exp,
                         "confidence": conf, "bias": f.get("bias"), "approved": bool(approved), "acted": c.symbol in bought,
                         "news_exp": f.get("news_exp"), "news_conf": f.get("news_conf"), "filings_dir": f.get("filings_dir"),
                         "filings_tilt": f.get("filings_tilt"), "memory_status": f.get("thesis_status"), "memory_adj": f.get("memory_adj"),
                         "vetoed": veto, "regime": regime.label, "mode": dcfg.get("mode")})
        Scorecard(s.state_dir).log(args.account, rows)

    # 8) cycle record + daily report
    summ = state.summary()
    (s.state_dir / "cycles").mkdir(exist_ok=True)
    (s.state_dir / "cycles" / f"{now:%Y-%m-%d_%H%M%S}_{args.account}{'_dry' if args.dry_run else ''}.json").write_text(json.dumps(
        {"account": args.account, "mode": args.mode, "dry_run": args.dry_run, "equity": snap.equity, "held": held,
         "candidates": cands, "forecasts": state.forecasts, "decision": decision, "summary": summ, "log": log},
        indent=1, default=str), encoding="utf-8")
    rep = s.reports_dir / f"{now:%Y-%m-%d}.md"
    with open(rep, "a", encoding="utf-8") as f:
        f.write(f"\n## {now:%H:%M} — {args.account} ({args.mode}){' — DRY RUN (no orders sent)' if args.dry_run else ''}\n")
        f.write(f"Equity ${snap.equity:,.2f} | cash ${snap.cash:,.2f} | positions {held} | regime {regime.label}"
                + (f" | **KILL SWITCH: {why}**" if tripped else "") + "\n\n")
        for c in state.candidates:
            r = state.forecasts.get(c.symbol)
            if r:
                f.write(f"- **{c.symbol}** {r['expected_5d_return_pct']:+.1f}% / conf {r['confidence']:.2f} ({r.get('bias')}): "
                        f"{r.get('catalyst')} / risk: {r.get('risk')}"
                        + (f" / filings {r['filings_dir']}" if r.get('filings_dir') else "")
                        + (f" / memory {r['thesis_status']}" if r.get('thesis_status') not in (None, 'none') else "") + "\n")
        if decision:
            f.write(f"\n{decision.get('market_note', '')}\n")
            if decision.get("vetoed"):
                f.write("\nVetoed: " + "; ".join(f"{k} ({v})" for k, v in decision["vetoed"].items()) + "\n")
        f.write("\nActions:\n" + "".join(f"- {l}\n" for l in log))
        ag = ", ".join(f"{k} {v['ok']}/{v['calls']} ok, {v['cached']} cached, ${v['usd']:.4f}" for k, v in summ["agents"].items())
        f.write(f"\nAgents: {ag or 'none'}\nClaude spend today: ${costs.spent_today():.4f}\n")
    print(f"Claude spend today: ${costs.spent_today():.4f} — report: {rep}")


if __name__ == "__main__":
    main()
