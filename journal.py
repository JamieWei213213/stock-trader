"""Browse the decision journal.

  python journal.py HOOD            # every time HOOD was researched: forecast, manager action, order, outcome
  python journal.py HOOD --full     # ...plus the exact text Claude read and its raw reply (latest entry)
  python journal.py --recent 20     # last 20 candidate entries across all symbols
  python journal.py --trades        # all trades, entry reason -> exit reason, P&L
  python journal.py --losers        # worst closed trades first, with their original reasoning
"""
from __future__ import annotations

import argparse

from trader.journal import Journal
from trader.settings import Settings


def fmt_entry(e: dict) -> str:
    f = e.get("forecast") or {}
    m = e.get("manager") or {}
    o = e.get("order")
    line = (f"{e['logged']}  {e['account']:5s} {e['symbol']:6s} rank {e.get('rank','?')}  "
            f"forecast {f.get('expected_5d_return_pct', 0):+.1f}% conf {f.get('confidence', 0):.2f} {f.get('bias','')}"
            f"{'  (cached)' if e.get('cached') else ''}")
    line += f"\n      catalyst: {f.get('catalyst','')} | risk: {f.get('risk','')}"
    if m:
        line += f"\n      manager: {m.get('action','').upper()} — {m.get('reason','')}"
    if o:
        line += f"\n      order: {o['qty']} @ {o['price']:.2f} stop {o['stop']:.2f} target {o['target']:.2f}"
    return line


def fmt_trade(t: dict) -> str:
    s = (f"{t['entry_time']}  {t['account']:5s} {t['symbol']:6s} {t['qty']:>4} @ {t['entry_price']:.2f}  "
         f"stop {t['stop']:.2f} target {t['target']:.2f}  forecast {t.get('forecast_5d_pct') or 0:+.1f}%")
    if t["status"] == "closed":
        s += f"\n      -> {t['exit_time']} exit {t['exit_price']:.2f} ({t['exit_reason']})  P&L ${t['pnl']:+,.2f} ({t['pnl_pct']*100:+.2f}%)"
    else:
        s += "\n      -> open"
    s += f"\n      why in: {t.get('entry_reason','')} | catalyst: {t.get('catalyst','')}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--recent", type=int, default=0)
    ap.add_argument("--trades", action="store_true")
    ap.add_argument("--losers", action="store_true")
    args = ap.parse_args()
    j = Journal(Settings().state_dir)

    if args.trades or args.losers:
        trades = j.trades if args.trades else sorted(j.closed_trades(), key=lambda t: t["pnl"])
        if not trades:
            print("no trades yet")
        for t in trades:
            print(fmt_trade(t), "\n")
        print(j.summary())
        return

    entries = j.entries_for(args.symbol.upper() if args.symbol else None, limit=args.recent or 30)
    if not entries:
        print("no journal entries" + (f" for {args.symbol}" if args.symbol else ""))
        return
    for e in entries:
        print(fmt_entry(e), "\n")
    if args.full and args.symbol:
        e = entries[0]
        print("=" * 80, "\nWHAT CLAUDE READ (latest entry):\n", e.get("prompt_sent") or "(cached — text not stored)")
        print("-" * 80, "\nRAW REPLY:\n", e.get("raw_reply") or "(cached)")
    if args.symbol:
        mine = [t for t in j.trades if t["symbol"] == args.symbol.upper()]
        if mine:
            print("=" * 80, "\nTRADES:")
            for t in mine:
                print(fmt_trade(t), "\n")


if __name__ == "__main__":
    main()
