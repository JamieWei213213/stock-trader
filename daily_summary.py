"""End-of-day: snapshot both accounts -> state/equity_history.csv, fill the scorecard, print a scoreboard.
Cron this once after the close (1:15pm PT).   `python daily_summary.py --scorecard` prints the Claude-vs-screener report.
"""
import argparse
import csv
from datetime import date

from trader.agents import Agents
from trader.broker import Broker
from trader.costs import CostTracker
from trader.journal import Journal
from trader.memory import PostMortemAgent
from trader.scorecard import Scorecard
from trader.settings import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", action="store_true", help="only print the scorecard report")
    args = ap.parse_args()
    s = Settings()
    sc = Scorecard(s.state_dir)
    if args.scorecard:
        print(sc.report())
        return

    hist = s.state_dir / "equity_history.csv"
    new = not hist.exists()
    brokers = {}
    journal = Journal(s.state_dir)
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "account", "equity", "cash", "positions", "daytrades"])
        for name in ("small", "large"):
            try:
                b = Broker(s.creds(name))
                snap = b.snapshot()
                brokers[name] = b
            except Exception as e:
                print(f"{name}: FAILED {e}")
                continue
            closed = journal.reconcile(name, b, {p["symbol"] for p in snap.positions})
            if closed:
                print(f"{name}: journal closed {closed} trade(s)")
            start = s.account_cfg(name)["starting_cash"]
            pnl = snap.equity - start
            print(f"{name:5s} equity ${snap.equity:>12,.2f}  P&L ${pnl:>+10,.2f} ({pnl/start*100:+.2f}%)  "
                  f"positions {len(snap.positions)}  daytrades(5d) {snap.daytrade_count}")
            for p in snap.positions:
                print(f"       {p['symbol']:6s} {p['qty']:>6.0f} @ {p['avg_entry']:.2f} -> {p['current']:.2f} "
                      f"({p['unrealized_plpc']*100:+.1f}%)")
            w.writerow([date.today().isoformat(), name, f"{snap.equity:.2f}", f"{snap.cash:.2f}",
                        len(snap.positions), snap.daytrade_count])

    # v3 post-mortem agent: one Haiku call per newly closed trade -> lesson on the trade + state/lessons.jsonl
    newly = [t for t in journal.closed_trades() if not t.get("post_mortem")]   # retried until written (cost cap, API errors)
    if newly and s.cfg.get("agents", {}).get("post_mortem", {}).get("enabled", True):
        try:
            rcfg = s.cfg["research"]
            agents = Agents(s.anthropic_key, rcfg, CostTracker(s.state_dir, s.cfg["pricing"], rcfg["daily_usd_cap"]),
                            cache_path=s.state_dir / "research_cache.json", names=s.names)
            pm = PostMortemAgent(agents, journal, s.state_dir)
            for t in newly:
                r = pm.write(t)
                if r:
                    print(f"post-mortem {t['symbol']} ({t['exit_reason']}, {t['pnl_pct']*100:+.1f}%): [{r['mistake_type']}] {r['lesson']}")
        except Exception as e:
            print(f"post-mortem agent failed: {e}")

    b = brokers.get("large") or brokers.get("small")
    if b:
        n = sc.fill(b)
        print(f"scorecard: filled {n} realized returns")
        print(sc.report())
        print(journal.summary())


if __name__ == "__main__":
    main()
