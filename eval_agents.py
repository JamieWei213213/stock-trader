"""Evaluate the LLM agents themselves — not the screener.

Answers, with numbers:
  1. Reliability   how often Haiku/Sonnet returned unparseable output; cache hit rate; cost per decision
  2. Forecasts     does Haiku's expected_5d_return_pct predict anything? directional accuracy, correlation,
                   error vs a naive "0%" forecast, and calibration by confidence bucket
  3. Selection     do the names Claude approved beat the ones it rejected, and the screener's raw top-6?
  4. Manager       were the PM's buy / hold / sell calls right over the next 5 days?
  5. Trades        realized P&L by exit reason (from the journal)

Data: state/scorecard.csv (filled by daily_summary.py), state/journal/, state/trades.json, state/agent_stats.json.
Usage:
  python eval_agents.py            # report on live data (needs >= ~20 filled scorecard rows to mean anything)
  python eval_agents.py --demo     # synthetic data, to see the report layout
  python eval_agents.py --md       # also write reports/eval_<date>.md
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from trader.journal import Journal
from trader.settings import Settings

MIN_N = 10  # below this, print numbers but flag them as too early


def _t(a: pd.Series, b: pd.Series | None = None) -> float:
    """t-stat of mean(a) (or of mean(a) - mean(b) for two independent samples)."""
    if b is None:
        return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std(ddof=1) else 0.0
    if len(a) < 3 or len(b) < 3:
        return 0.0
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se else 0.0


# ------------------------------------------------------------------------------------------ sections
def section_reliability(s: Settings) -> list[str]:
    out = ["## 1. Reliability of structured outputs"]
    p = s.state_dir / "agent_stats.json"
    if not p.exists():
        return out + ["(no agent_stats.json yet — counters start with the next cycle)"]
    st = json.loads(p.read_text(encoding="utf-8"))
    calls, fails, hits = st.get("calls", 0), st.get("parse_failures", 0), st.get("cache_hits", 0)
    out.append(f"API calls: {calls}   parse failures: {fails} ({fails / calls * 100 if calls else 0:.1f}%)   "
               f"cache hits: {hits} ({hits / (calls + hits) * 100 if calls + hits else 0:.0f}% of research requests served from cache)")
    for m, v in st.get("by_model", {}).items():
        out.append(f"  {m}: {v['calls']} calls, {v['parse_failures']} parse failures")
    cp = s.state_dir / "costs.json"
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        usd = sum(v["usd"] for v in c.values())
        n_calls = sum(v["calls"] for v in c.values())
        out.append(f"total Claude spend ${usd:.3f} over {n_calls} calls = ${usd / n_calls if n_calls else 0:.4f} per call; "
                   f"tokens in/out {st.get('input_tokens', 0):,}/{st.get('output_tokens', 0):,}")
    return out


def section_forecasts(sc: pd.DataFrame) -> list[str]:
    out = ["## 2. Haiku forecast quality (expected 5-day return vs realized)"]
    d = sc.dropna(subset=["expected_5d_pct", "ret_5d"]).copy()
    if len(d) < 3:
        return out + [f"(only {len(d)} rows with realized 5-day returns — need ~{MIN_N}+; run again in a week)"]
    d["real_pct"] = d["ret_5d"] * 100
    flag = "  <-- too few rows to trust" if len(d) < MIN_N else ""
    pear = d[["expected_5d_pct", "real_pct"]].corr().iloc[0, 1]
    spear = d[["expected_5d_pct", "real_pct"]].corr(method="spearman").iloc[0, 1]
    out.append(f"n = {len(d)}{flag}")
    out.append(f"correlation forecast vs realized: Pearson {pear:+.2f}, Spearman {spear:+.2f}  (0 = no information)")
    called = d[d["expected_5d_pct"].abs() >= 0.5]
    if len(called):
        acc = (np.sign(called["expected_5d_pct"]) == np.sign(called["real_pct"])).mean()
        out.append(f"directional accuracy when |forecast| >= 0.5%: {acc * 100:.0f}% (n={len(called)}; coin flip = 50%)")
    mae_model = (d["expected_5d_pct"] - d["real_pct"]).abs().mean()
    mae_zero = d["real_pct"].abs().mean()
    out.append(f"mean abs error: model {mae_model:.2f}%  vs  naive 'always 0%' {mae_zero:.2f}%  "
               f"({'model better' if mae_model < mae_zero else 'naive better'})")
    out.append(f"forecast bias: mean forecast {d['expected_5d_pct'].mean():+.2f}% vs mean realized {d['real_pct'].mean():+.2f}%  "
               f"(a big positive gap = systematic optimism)")
    out.append("calibration by confidence:")
    bins = [(0, 0.5, "< 0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 1.01, ">= 0.7")]
    for lo, hi, label in bins:
        g = d[(d["confidence"] >= lo) & (d["confidence"] < hi)]
        gc = g[g["expected_5d_pct"].abs() >= 0.5]
        if len(g):
            acc = (np.sign(gc["expected_5d_pct"]) == np.sign(gc["real_pct"])).mean() * 100 if len(gc) else float("nan")
            out.append(f"  conf {label:>7}: n={len(g):3d}  directional acc {acc:5.0f}%  mean realized {g['real_pct'].mean():+.2f}%")
    out.append("  (calibrated = accuracy rises with confidence; if it doesn't, confidence carries no information)")
    return out


def section_selection(sc: pd.DataFrame) -> list[str]:
    out = ["## 3. Selection value: Claude-approved vs rejected vs raw screener"]
    d = sc.dropna(subset=["ret_5d"]).copy()
    if len(d) < 3:
        return out + ["(not enough filled rows yet)"]
    d["approved"] = d["approved"].astype(str).str.lower() == "true"
    d["acted"] = d["acted"].astype(str).str.lower() == "true"
    groups = {
        "screener top-6 (all candidates)": d,
        "Claude approved": d[d["approved"]],
        "Claude rejected": d[~d["approved"]],
        "actually bought": d[d["acted"]],
        "screener rank 1-3": d[d["rank"] <= 3],
        "screener rank 4-6": d[(d["rank"] > 3) & (d["rank"] <= 6)],
    }
    for name, g in groups.items():
        if len(g):
            out.append(f"  {name:>32}: mean 5d {g['ret_5d'].mean() * 100:+.2f}%  hit {(g['ret_5d'] > 0).mean() * 100:.0f}%  n={len(g)}")
    a, r = d[d["approved"]]["ret_5d"], d[~d["approved"]]["ret_5d"]
    if len(a) >= 3 and len(r) >= 3:
        out.append(f"  approved minus rejected: {(a.mean() - r.mean()) * 100:+.2f}%  t = {_t(a, r):.2f}  "
                   f"({'suggestive' if abs(_t(a, r)) >= 2 else 'not distinguishable from noise'})")
    return out


def section_manager(s: Settings, j: Journal, bars_close: pd.DataFrame | None) -> list[str]:
    out = ["## 4. Portfolio manager (Sonnet) decisions"]
    entries = j.entries_for(limit=100000)
    acts = [e for e in entries if e.get("manager")]
    if not acts:
        return out + ["(no journal entries with manager decisions yet)"]
    counts = pd.Series([e["manager"].get("action") for e in acts]).value_counts()
    out.append("decisions: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if bars_close is None:
        return out + ["(no price data available to grade hold/sell decisions — needs Alpaca keys)"]
    rows = []
    for e in acts:
        sym, act = e["symbol"], e["manager"].get("action")
        day = pd.Timestamp(e["logged"][:10])
        if sym not in bars_close.columns:
            continue
        ser = bars_close[sym].dropna()
        after = ser[ser.index >= day]
        if len(after) > 5:
            rows.append({"action": act, "fwd5": float(after.iloc[5] / after.iloc[0] - 1)})
    if not rows:
        return out + ["(decisions too recent to grade — need 5 trading days)"]
    df = pd.DataFrame(rows)
    for act, g in df.groupby("action"):
        right = (g["fwd5"] > 0).mean() if act in ("buy", "hold") else (g["fwd5"] < 0).mean()
        out.append(f"  {act:>5}: n={len(g):3d}  mean next-5d {g['fwd5'].mean() * 100:+.2f}%  "
                   f"'right' {right * 100:.0f}%  (buy/hold right if price rose; sell right if it fell)")
    return out


def section_trades(j: Journal) -> list[str]:
    return ["## 5. Realized trades", j.summary()]


# ------------------------------------------------------------------------------------------ data
def load_scorecard(s: Settings) -> pd.DataFrame:
    p = s.state_dir / "scorecard.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ("expected_5d_pct", "confidence", "ret_1d", "ret_5d", "ret_10d", "rank"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_bars_close(s: Settings, j: Journal) -> pd.DataFrame | None:
    syms = sorted({e["symbol"] for e in j.entries_for(limit=100000)})
    if not syms:
        return None
    try:
        from trader.broker import Broker
        b = Broker(s.creds("large"))
        bars = b.daily_bars(syms, 45)
        close = bars["close"].unstack("symbol").sort_index()
        close.index = close.index.tz_convert(None).normalize()
        return close
    except Exception as e:
        print(f"(price data unavailable: {e})")
        return None


def demo_data(s: Settings):
    """Synthetic scorecard + stats so the report layout can be checked without live data."""
    rng = np.random.default_rng(1)
    n = 120
    conf = rng.uniform(0.3, 0.9, n)
    exp = rng.normal(0.8, 1.8, n)
    real = 0.15 * exp + rng.normal(0, 3.0, n)      # weak but real signal
    df = pd.DataFrame({
        "logged": [(datetime.now() - timedelta(days=int(i / 6))).strftime("%Y-%m-%d %H:%M") for i in range(n)],
        "account": rng.choice(["small", "large"], n), "symbol": rng.choice(s.symbols[:25], n),
        "rank": rng.integers(1, 7, n), "price": rng.uniform(20, 300, n), "expected_5d_pct": exp.round(1),
        "confidence": conf.round(2), "bias": np.where(exp > 0.5, "bullish", np.where(exp < -0.5, "bearish", "neutral")),
        "approved": (exp >= 1.5) & (conf >= 0.6), "acted": (exp >= 1.5) & (conf >= 0.6) & (rng.random(n) < 0.6),
        "ret_1d": (real / 5 / 100).round(4), "ret_5d": (real / 100).round(4), "ret_10d": (real * 1.3 / 100).round(4),
    })
    (s.state_dir / "scorecard.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (s.state_dir / "agent_stats.json").write_text(json.dumps({"calls": 140, "parse_failures": 2, "cache_hits": 61,
        "input_tokens": 210000, "output_tokens": 18000,
        "by_model": {"claude-haiku-4-5-20251001": {"calls": 120, "parse_failures": 2}, "claude-sonnet-5": {"calls": 20, "parse_failures": 0}}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--md", action="store_true", help="also write reports/eval_<date>.md")
    args = ap.parse_args()
    s = Settings()
    if args.demo:
        demo_data(s)
    j = Journal(s.state_dir)
    sc = load_scorecard(s)
    bars = None if args.demo else load_bars_close(s, j)

    lines = [f"# Agent evaluation — {datetime.now():%Y-%m-%d %H:%M}{' (DEMO DATA)' if args.demo else ''}", ""]
    lines += section_reliability(s) + [""]
    lines += section_forecasts(sc) + [""]
    lines += section_selection(sc) + [""]
    lines += section_manager(s, j, bars) + [""]
    lines += section_trades(j)
    lines += ["", "How to read this: the screener backtest (backtest.py) tests the rules; THIS tests the LLM layer on top of them. "
              "Section 2 asks whether the forecasts carry information at all, section 3 whether acting on them beats not acting, "
              "section 4 whether the manager's calls were right. Treat anything with n < 30 as a preview."]
    text = "\n".join(lines)
    print(text)
    if args.md:
        p = s.reports_dir / f"eval_{date.today()}.md"
        p.write_text(text, encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
