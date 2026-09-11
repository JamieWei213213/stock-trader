"""Evaluate the LLM agents themselves — not the screener.

Answers, with numbers:
  1. Reliability   how often Haiku/Sonnet returned unparseable output; cache hit rate; cost per decision
  2. Forecasts     does Haiku's expected_5d_return_pct predict anything? directional accuracy, correlation,
                   error vs a naive "0%" forecast, and calibration by confidence bucket
  3. Selection     do the names Claude approved beat the ones it rejected, and the screener's raw top-6?
  4. Manager       were the PM's buy / hold / sell calls right over the next 5 days?
  5. Trades        realized P&L by exit reason (from the journal)
  v3 additions:
  6. Veto          were the names the agents vetoed worse than the ones they passed? (by veto reason)
  7. Ablation      forecast quality of news-only vs combined (news + filings + memory); each agent's marginal value
  8. Fleet health  per-agent success / cache / cost / latency from state/cycles, and post-mortem lesson mix

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
from trader.scorecard import FIELDS
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


# ------------------------------------------------------------------------------------------ v3 sections
def _corr(d: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    x = d[[a, b]].dropna()
    if len(x) < 3:
        return float("nan"), float("nan"), len(x)
    return float(x.corr().iloc[0, 1]), float(x.corr(method="spearman").iloc[0, 1]), len(x)


def section_veto(sc: pd.DataFrame) -> list[str]:
    out = ["## 6. Veto accuracy (v3 decision rule): passed vs vetoed screener picks"]
    if "vetoed" not in sc or sc.empty:
        return out + ["(no v3 rows yet)"]
    d = sc[sc["mode"].astype(str) == "veto"].dropna(subset=["ret_5d"]).copy()   # held rows already removed in load_scorecard
    if len(d) < 3:
        return out + [f"(only {len(d)} v3 rows with realized returns)"]
    d["vetoed"] = d["vetoed"].fillna("").astype(str)
    d["is_veto"] = d["vetoed"].str.len() > 0
    d["kind"] = np.select(
        [d["vetoed"] == "", d["vetoed"].str.startswith("forecast"), d["vetoed"].str.startswith("bearish"),
         d["vetoed"].str.startswith("memory"), d["vetoed"].str.startswith("risk reviewer"), d["vetoed"].str.startswith("regime"),
         d["vetoed"].str.startswith("no slot"), d["vetoed"].str.startswith("no forecast")],
        ["passed", "veto: forecast low", "veto: bearish", "veto: thesis broken", "risk reviewer", "regime freeze", "no slot", "agent failed"],
        "other")
    for kind, g in d.groupby("kind"):
        out.append(f"  {kind:>22}: mean 5d {g['ret_5d'].mean() * 100:+.2f}%  hit {(g['ret_5d'] > 0).mean() * 100:.0f}%  n={len(g)}")
    p, v = d[~d["is_veto"]]["ret_5d"], d[d["is_veto"] & d["kind"].str.startswith("veto")]["ret_5d"]
    if len(p) >= 3 and len(v) >= 3:
        out.append(f"  passed minus agent-vetoed: {(p.mean() - v.mean()) * 100:+.2f}%  t = {_t(p, v):.2f}  "
                   f"({'the veto is earning its keep' if _t(p, v) >= 2 else 'not distinguishable from noise yet'})")
    out.append("  (a useful veto = the vetoed group does WORSE than the passed group; 'no slot' rows are a free control group)")
    return out


def section_ablation(sc: pd.DataFrame) -> list[str]:
    out = ["## 7. Ablation: what each agent adds to forecast quality"]
    if "news_exp" not in sc:
        return out + ["(no v3 rows yet)"]
    d = sc.dropna(subset=["ret_5d"]).copy()
    for c in ("news_exp", "news_conf", "filings_tilt", "memory_adj", "expected_5d_pct"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["news_exp"])
    if len(d) < 3:
        return out + [f"(only {len(d)} rows with per-agent components)"]
    d["real"] = d["ret_5d"] * 100
    rows = [("news only", "news_exp"), ("news + filings + memory (combined)", "expected_5d_pct")]
    if d["filings_tilt"].notna().any():
        d["news_plus_filings"] = d["news_exp"] + d["filings_tilt"].fillna(0)
        rows.insert(1, ("news + filings", "news_plus_filings"))
    if d["memory_adj"].notna().any():
        d["news_plus_memory"] = d["news_exp"] + d["memory_adj"].fillna(0)
        rows.insert(-1, ("news + memory", "news_plus_memory"))
    out.append(f"n = {len(d)}   (correlation with realized 5-day return; higher = more information; differences < 0.05 are noise at this n)")
    for label, col in rows:
        pe, sp, n = _corr(d, col, "real")
        called = d[d[col].abs() >= 0.5]
        acc = (np.sign(called[col]) == np.sign(called["real"])).mean() * 100 if len(called) else float("nan")
        out.append(f"  {label:>36}: Pearson {pe:+.2f}  Spearman {sp:+.2f}  directional {acc:.0f}%")
    # filings agent on its own: does 'up' beat 'down'?
    if "filings_dir" in d and d["filings_dir"].notna().any():
        g = d.dropna(subset=["filings_dir"]).groupby("filings_dir")["real"]
        out.append("  filings agent alone, mean 5d by earnings_direction: " +
                   ", ".join(f"{k} {v.mean():+.2f}% (n={len(v)})" for k, v in g))
        out.append("  (the filings signal is about the NEXT QUARTER; a 5-day read is an early, noisy check — the long-horizon score is in monthly_report.py)")
    if "memory_status" in d and d["memory_status"].notna().any():
        g = d.dropna(subset=["memory_status"]).groupby("memory_status")["real"]
        out.append("  memory agent, mean 5d by thesis_status: " + ", ".join(f"{k} {v.mean():+.2f}% (n={len(v)})" for k, v in g))
    out.append("  Read: if 'combined' is not above 'news only', the extra agents are cost without value — turn them off in config.yaml.")
    return out


def section_fleet(s: Settings) -> list[str]:
    out = ["## 8. Fleet health (per agent, from state/cycles/*.json)"]
    cyc = sorted((s.state_dir / "cycles").glob("*.json")) if (s.state_dir / "cycles").exists() else []
    if not cyc:
        return out + ["(no cycle records yet)"]
    agg: dict[str, dict] = {}
    for p in cyc[-200:]:
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for k, v in (c.get("summary") or {}).get("agents", {}).items():
            a = agg.setdefault(k, {"calls": 0, "ok": 0, "cached": 0, "failed": 0, "usd": 0.0, "seconds": 0.0})
            for f in a:
                a[f] += v.get(f, 0)
    out.append(f"cycles: {len(cyc)}")
    for k, a in agg.items():
        n = a["calls"] or 1
        out.append(f"  {k:>10}: {a['calls']:4d} calls  ok {a['ok'] / n * 100:5.1f}%  cached {a['cached'] / n * 100:4.0f}%  "
                   f"failed {a['failed']:3d}  ${a['usd']:.3f}  avg {a['seconds'] / n:.1f}s")
    lp = s.state_dir / "lessons.jsonl"
    if lp.exists():
        les = [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if les:
            mix = pd.Series([l.get("mistake_type") for l in les]).value_counts()
            out.append(f"post-mortems: {len(les)} — " + ", ".join(f"{k} {v}" for k, v in mix.items()) +
                       f"; avoidable {sum(bool(l.get('avoidable')) for l in les)}")
            for l in les[-3:]:
                out.append(f"  latest: {l['symbol']} {l.get('exit_reason')} {float(l.get('pnl_pct') or 0) * 100:+.1f}% [{l.get('mistake_type')}] {l.get('lesson')}")
    return out


# ------------------------------------------------------------------------------------------ data
def load_scorecard(s: Settings) -> pd.DataFrame:
    p = s.state_dir / "scorecard.csv"
    if not p.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(p)
    for c in FIELDS:
        if c not in df:
            df[c] = np.nan
    if "vetoed" in df:   # v3: rows for names already held are re-forecasts, not picks — keep them out of the pick groups
        df = df[~df["vetoed"].fillna("").astype(str).str.startswith("held")]
    for c in ("expected_5d_pct", "confidence", "ret_1d", "ret_5d", "ret_10d", "rank", "news_exp", "news_conf", "filings_tilt", "memory_adj"):
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
    # v3 components: news forecast + a filings tilt that carries a little extra signal + a memory nudge that carries none
    tilt = np.where(rng.random(n) < 0.7, np.sign(real + rng.normal(0, 4, n)) * 0.3, 0.0)
    df["news_exp"] = (exp - tilt).round(2); df["news_conf"] = conf.round(2)
    df["filings_dir"] = np.where(tilt > 0, "up", np.where(tilt < 0, "down", "flat")); df["filings_tilt"] = tilt.round(2)
    df["memory_status"] = rng.choice(["none", "intact", "broken"], n, p=[0.6, 0.3, 0.1]); df["memory_adj"] = 0.0
    df["vetoed"] = np.where(exp < -0.5, "forecast " + exp.round(1).astype(str) + "% < -0.5%",
                            np.where(rng.random(n) < 0.2, "no slot (positions full)", ""))
    df["regime"] = "risk-on"; df["mode"] = "veto"
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
    lines += section_trades(j) + [""]
    lines += section_veto(sc) + [""]
    lines += section_ablation(sc) + [""]
    lines += section_fleet(s)
    lines += ["", "How to read this: the screener backtest (backtest.py) tests the rules; THIS tests the LLM layer on top of them. "
              "Section 2 asks whether the forecasts carry information at all, section 3 whether acting on them beats not acting, "
              "section 4 whether the manager's calls were right, section 6 whether the v3 veto helps, section 7 which agents earn "
              "their cost. Treat anything with n < 30 as a preview."]
    text = "\n".join(lines)
    print(text)
    if args.md:
        p = s.reports_dir / f"eval_{date.today()}.md"
        p.write_text(text, encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
