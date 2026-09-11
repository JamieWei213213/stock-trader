"""Monthly system report (v3, item 8): one page that answers "how is the whole experiment doing?"

Sections: both accounts vs SPY for the month and since start; trades and exit-reason mix; the agent
evaluation (same sections as eval_agents.py, so the numbers agree); the veto/ablation verdict in one line
each; what changed this month (CHANGELOG.md entries dated in the month); and open questions.
Writes reports/monthly_<YYYY-MM>.md. Cron runs it on the 1st; run by hand any time:
  python monthly_report.py            # current month so far
  python monthly_report.py --month 2026-09
"""
from __future__ import annotations

import argparse
import re
from datetime import date, datetime

import pandas as pd

import eval_agents as ev
from trader.journal import Journal
from trader.settings import Settings


def equity_section(s: Settings, month: str) -> list[str]:
    out = ["## 1. Accounts vs SPY"]
    p = s.state_dir / "equity_history.csv"
    if not p.exists():
        return out + ["(no equity_history.csv yet — daily_summary.py writes it after each close)"]
    h = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    spy = None
    try:
        from trader.broker import Broker
        b = Broker(s.creds("large"))
        bars = b.daily_bars(["SPY"], 400)
        spy = bars["close"].xs("SPY", level="symbol").sort_index()
        spy.index = spy.index.tz_convert(None).normalize()
    except Exception as e:
        out.append(f"(SPY unavailable: {e})")
    for name in ("small", "large"):
        g = h[h["account"] == name]
        if g.empty:
            continue
        start_eq = float(s.account_cfg(name)["starting_cash"])
        last = g.iloc[-1]
        m = g[g["date"].dt.strftime("%Y-%m") == month]
        prev = g[g["date"] < m["date"].min()] if not m.empty else pd.DataFrame()
        base_m = float(prev.iloc[-1]["equity"]) if len(prev) else start_eq
        ret_m = float(m.iloc[-1]["equity"]) / base_m - 1 if len(m) else float("nan")
        ret_all = float(last["equity"]) / start_eq - 1
        line = f"  {name:>5}: equity ${float(last['equity']):,.2f}  month {ret_m * 100:+.2f}%  since start {ret_all * 100:+.2f}%"
        if spy is not None and len(g):
            d0, d1 = g.iloc[0]["date"], last["date"]
            sp = spy[(spy.index >= d0)]
            if len(sp) > 1:
                line += f"  | SPY same period {(float(sp.iloc[-1]) / float(sp.iloc[0]) - 1) * 100:+.2f}%"
            if len(m):
                spm = spy[(spy.index >= m["date"].min()) & (spy.index <= d1)]
                if len(spm) > 1:
                    line += f", month {(float(spm.iloc[-1]) / float(spm.iloc[0]) - 1) * 100:+.2f}%"
        mdd = float((g["equity"] / g["equity"].cummax() - 1).min())
        line += f"  | max drawdown {mdd * 100:.2f}%"
        out.append(line)
    out.append("  (the index fund is the baseline; beating it after costs over a month means nothing, over a year means something)")
    return out


def trades_section(j: Journal, month: str) -> list[str]:
    out = ["## 2. Trades this month"]
    c = [t for t in j.closed_trades() if str(t.get("exit_time", ""))[:7] == month]
    o = j.open_trades()
    if not c and not o:
        return out + ["(no trades)"]
    if c:
        wins = [t for t in c if t["pnl"] > 0]
        out.append(f"  closed {len(c)}: {len(wins)} winners ({len(wins) / len(c) * 100:.0f}%), total ${sum(t['pnl'] for t in c):+,.2f}, "
                   f"avg {sum(t['pnl_pct'] for t in c) / len(c) * 100:+.2f}%, "
                   f"avg winner {sum(t['pnl_pct'] for t in wins) / len(wins) * 100 if wins else 0:+.2f}%, "
                   f"avg loser {sum(t['pnl_pct'] for t in c if t['pnl'] <= 0) / max(len(c) - len(wins), 1) * 100:+.2f}%")
        by = {}
        for t in c:
            by.setdefault(t["exit_reason"], []).append(t["pnl_pct"])
        out.append("  exits: " + ", ".join(f"{k} n={len(v)} avg {sum(v) / len(v) * 100:+.2f}%" for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))))
        worst = sorted(c, key=lambda t: t["pnl"])[:3]
        out.append("  worst: " + "; ".join(f"{t['symbol']} {t['pnl_pct'] * 100:+.1f}% ({t['exit_reason']}"
                                          + (f", lesson: {t['post_mortem']['lesson']}" if t.get("post_mortem") else "") + ")" for t in worst))
    out.append(f"  open now: {len(o)} ({', '.join(t['symbol'] for t in o) or '-'})")
    return out


def changes_section(s: Settings, month: str) -> list[str]:
    out = ["## 4. What changed this month"]
    p = s.root / "CHANGELOG.md"
    if not p.exists():
        return out + ["(no CHANGELOG.md)"]
    hits = []
    for block in re.split(r"\n(?=## )", p.read_text(encoding="utf-8")):
        m = re.match(r"## (\d{4}-\d{2}-\d{2})", block)
        if m and m.group(1)[:7] == month:
            hits.append(block.strip())
    return out + (hits or ["(nothing dated this month in CHANGELOG.md)"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=None, help="YYYY-MM (default: current month)")
    args = ap.parse_args()
    s = Settings()
    month = args.month or date.today().strftime("%Y-%m")
    j = Journal(s.state_dir)
    sc = ev.load_scorecard(s)
    lines = [f"# Stock Trader — monthly system report {month} (written {datetime.now():%Y-%m-%d %H:%M})", ""]
    lines += equity_section(s, month) + [""]
    lines += trades_section(j, month) + [""]
    lines += ["## 3. Agent evaluation (see eval_agents.py for definitions)"]
    lines += ev.section_reliability(s) + [""] + ev.section_forecasts(sc) + [""] + ev.section_selection(sc) + [""]
    lines += ev.section_veto(sc) + [""] + ev.section_ablation(sc) + [""] + ev.section_fleet(s) + [""]
    lines += changes_section(s, month) + [""]
    lines += ["## 5. Open questions for next month",
              "- Is 'combined' above 'news only' in section 7? If not after ~100 rows, disable filings/memory in config.yaml.",
              "- Is the vetoed group worse than the passed group (section 6)? If not, the veto threshold is doing nothing.",
              "- Are stop-outs the dominant exit? If > 50% of exits, widen atr_stop_mult or lower risk; check post-mortem 'stop_too_tight' count.",
              "- Are we beating SPY since start after ~3 months? If not, the honest answer is that the index fund is winning."]
    text = "\n".join(lines)
    print(text)
    p = s.reports_dir / f"monthly_{month}.md"
    p.write_text(text, encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
