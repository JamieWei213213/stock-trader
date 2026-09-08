"""One full cycle for one account: screen -> research -> decide -> execute -> log -> report.

Usage:
  python run_cycle.py --account small            # $1k account (cron: once daily pre-open)
  python run_cycle.py --account large            # $100k account (cron: 6:05am + 10:05am)
  python run_cycle.py --account small --dry-run  # everything except placing orders
  python run_cycle.py --account large --force    # run even if market is closed (testing)
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
from trader.journal import Journal
from trader.scorecard import Scorecard
from trader.screener import pick_candidates, screen
from trader.settings import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", choices=["small", "large"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even when market is closed")
    args = ap.parse_args()

    s = Settings()
    rules = s.account_cfg(args.account)
    rcfg = s.cfg["research"]
    broker = Broker(s.creds(args.account))
    costs = CostTracker(s.state_dir, s.cfg["pricing"], rcfg["daily_usd_cap"])
    agents = Agents(s.anthropic_key, rcfg, costs, cache_path=s.state_dir / "research_cache.json", names=s.names)
    now = datetime.now()
    print(f"=== {now:%Y-%m-%d %H:%M} cycle for '{args.account}' (dry_run={args.dry_run}) ===")

    if not broker.market_open() and not args.force:
        clock = broker.trading.get_clock()
        if clock.next_open.astimezone().date() != now.date():
            print("market closed today — skipping (use --force to override)")
            return

    snap = broker.snapshot()
    held = [p["symbol"] for p in snap.positions]
    print(f"equity ${snap.equity:,.2f} cash ${snap.cash:,.2f} positions {held} daytrades {snap.daytrade_count}")

    # 1) free screen over the whole universe (held names are always included so they get re-forecast)
    table = screen(s, broker, keep=held)
    if table.empty:
        print("screener returned nothing — aborting cycle")
        return
    earnings = EarningsCalendar(s.state_dir, rcfg.get("earnings_blackout_days", 2))

    # affordability: with risk-based sizing, can this account buy >= 1 share within its risk budget?
    def affordable(sym) -> bool:
        if sym in held:
            return True
        r = table.loc[sym]
        stop_dist = rules["atr_stop_mult"] * float(r["atr"])
        qty = min(snap.equity * rules["risk_per_trade_pct"] // max(stop_dist, 1e-9),
                  snap.equity * rules["max_position_pct"] // float(r["price"]))
        return qty >= 1
    eligible = [sym for sym in table.index if affordable(sym) and not earnings.soon(sym)]
    dropped = len(table) - len(eligible)
    table = table.loc[eligible]
    print(f"screened {len(table) + dropped} names ({dropped} skipped: unaffordable or earnings soon), variant={s.cfg['screener'].get('variant')}")
    cands = pick_candidates(table, s.cfg["screener"]["candidates_per_cycle"], held)
    print("candidates:", cands)

    # 2) cheap research per candidate (anonymized, article bodies, numeric forecast)
    research = []
    for sym in cands:
        if sym not in table.index:
            continue
        news = broker.recent_news(sym, rcfg["news_lookback_hours"], rcfg["news_items_per_stock"])
        r = agents.research_stock(sym, table.loc[sym].to_dict(), news)
        if r:
            research.append(r)
            tag = "cached" if r.get("_cached") else f"${r['_cost_usd']:.4f}"
            print(f"  {sym}: {r.get('expected_5d_return_pct', 0):+.1f}% conf {r.get('confidence', 0):.2f} {r.get('bias')} — {r.get('catalyst')}  ({tag})")

    # 3) one decision call
    decision = agents.rank_and_decide(args.account, rules, asdict(snap), research) if research else None
    if decision:
        print("decision:", json.dumps({k: v for k, v in decision.items() if not k.startswith('_')}))

    # 4) execute with guards
    prices = {sym: float(table.loc[sym, "price"]) for sym in cands if sym in table.index}
    atrs = {sym: float(table.loc[sym, "atr"]) for sym in cands if sym in table.index}
    try:
        prices.update({k: v for k, v in broker.latest_prices(cands).items() if v > 0})
    except Exception as e:
        print(f"  latest quotes unavailable ({e}); using last close")
    # decision journal: one file per candidate with everything Claude saw and said
    journal = Journal(s.state_dir)
    forecasts = {r["symbol"]: r for r in research}
    actions = {a.get("symbol"): a for a in (decision or {}).get("actions", [])}
    jfiles: dict[str, str] = {}
    if not args.dry_run:
        for rank, sym in enumerate(cands, 1):
            r = forecasts.get(sym, {})
            f = table.loc[sym].to_dict() if sym in table.index else {}
            jfiles[sym] = str(journal.write_entry(args.account, sym, {
                "rank": rank,
                "features": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in f.items()},
                "prompt_sent": r.get("_prompt"), "raw_reply": r.get("_raw"), "cached": bool(r.get("_cached")),
                "forecast": {k: v for k, v in r.items() if not k.startswith("_") and k != "symbol"},
                "manager": actions.get(sym), "market_note": (decision or {}).get("market_note"),
            }, now))
    ex = Executor(args.account, rules, broker, s.state_dir, s.is_volatile, dry_run=args.dry_run,
                  earnings_soon=earnings.soon, journal=journal, forecasts=forecasts, journal_files=jfiles)
    log = ex.apply(decision, snap, prices, atrs)
    if not args.dry_run:   # attach the order to the journal entry
        for sym, o in ex.orders_placed.items():
            if sym in jfiles:
                pth = Path(jfiles[sym]); d = json.loads(pth.read_text(encoding="utf-8")); d["order"] = o
                pth.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")

    # 4b) confirm with the broker what actually happened (orders can be rejected or sit pending)
    if not args.dry_run and ex.orders_placed:
        try:
            pending = {o["symbol"]: o for o in broker.open_orders()}
            now_held = {p["symbol"] for p in broker.snapshot().positions}
            for sym, o in ex.orders_placed.items():
                state = "FILLED" if sym in now_held else ("pending at broker" if sym in pending else f"NOT FOUND ({o.get('status')})")
                print(f"  broker check {sym}: {state}")
                log.append(f"broker check {sym}: {state}")
        except Exception as e:
            print(f"  broker check failed: {e}")

    # 5) scorecard: what did the screener offer, what did Claude say, what did we do
    if not args.dry_run:
        bought = {l.split()[2] for l in log if l.startswith("BUY ")}
        rows = []
        for rank, sym in enumerate(cands, 1):
            r = next((x for x in research if x.get("symbol") == sym), {})
            exp, conf = r.get("expected_5d_return_pct"), r.get("confidence")
            approved = (exp is not None and conf is not None and exp >= rcfg.get("min_expected_return_pct", 1.5)
                        and conf >= rcfg.get("min_confidence", 0.6))
            rows.append({"symbol": sym, "rank": rank, "price": prices.get(sym), "expected_5d_pct": exp,
                         "confidence": conf, "bias": r.get("bias"), "approved": approved,
                         "acted": sym in bought})
        Scorecard(s.state_dir).log(args.account, rows)

    # 6) append to daily report
    rep = s.reports_dir / f"{now:%Y-%m-%d}.md"
    with open(rep, "a", encoding="utf-8") as f:
        f.write(f"\n## {now:%H:%M} — {args.account}{' — DRY RUN (no orders sent)' if args.dry_run else ''}\n")
        f.write(f"Equity ${snap.equity:,.2f} | cash ${snap.cash:,.2f} | positions {held}\n\n")
        for r in research:
            f.write(f"- **{r['symbol']}** {r.get('expected_5d_return_pct', 0):+.1f}% / conf {r.get('confidence', 0):.2f} "
                    f"({r.get('bias')}): {r.get('catalyst')} / risk: {r.get('risk')}\n")
        if decision:
            f.write(f"\nMarket note: {decision.get('market_note','')}\n")
        f.write("\nActions:\n" + "".join(f"- {l}\n" for l in log))
        f.write(f"\nClaude spend today: ${costs.spent_today():.4f}\n")
    print(f"Claude spend today: ${costs.spent_today():.4f} — report: {rep}")


if __name__ == "__main__":
    main()
