"""End-of-day: snapshot both accounts -> state/equity_history.csv, fill the scorecard, print a scoreboard.
Cron this once after the close (1:15pm PT).   `python daily_summary.py --scorecard` prints the Claude-vs-screener report.
"""
import argparse
import csv
from datetime import date

from trader.broker import Broker
from trader.journal import Journal
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
            closed = Journal(s.state_dir).reconcile(name, b, {p["symbol"] for p in snap.positions})
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

    b = brokers.get("large") or brokers.get("small")
    if b:
        n = sc.fill(b)
        print(f"scorecard: filled {n} realized returns")
        print(sc.report())
        print(Journal(s.state_dir).summary())


if __name__ == "__main__":
    main()
