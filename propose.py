"""Propose a screener rule in plain English -> Haiku turns it into a strict spec -> the FAIR backtest runs it
-> a deterministic verdict says whether it passes the bar. (v3; the honest version of "describe a strategy
and the AI backtests it".)

  python propose.py "buy last week's biggest losers that are still above their 200-day trend"
  python propose.py --spec my_rule.json            # skip the LLM, test a hand-written spec
  python propose.py --list                         # every proposal tried so far and how it did

The LLM only translates English into a spec; it cannot write code, choose the data, or touch the scoring.
The spec is a weighted sum of z-scored features with optional filters (below). The backtest is backtest.py
unchanged: point-in-time universe, hold-out year, costs, random baseline. The verdict is a rule: train t >= 3
AND positive hold-out edge, both entry timings shown. Every proposal is logged to state/proposals.jsonl and
the multiple-testing count is printed, because trying 20 ideas and keeping the best is how people fool themselves.
Runs on your PC (needs data/daily_bars.parquet from `python backtest.py --sweep`); not for the VPS.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import backtest as bt
from trader.agents import Agents, _extract_json
from trader.costs import CostTracker
from trader.screener import VARIANTS, _z
from trader.settings import Settings

FEATURES = {
    "ret_1d":        "return over the last 1 trading day",
    "ret_5d":        "return over the last 5 trading days (1 week)",
    "ret_20d":       "return over the last 20 trading days (1 month)",
    "mom_12_2":      "return from 12 months ago to 1 month ago (classic momentum, skips the last month)",
    "vol_surge":     "today's volume / 20-day average volume",
    "atr_pct":       "14-day average true range / price (volatility)",
    "avg_dollar_vol": "20-day average dollar volume (liquidity)",
    "dist_20d_high": "price / 20-day high - 1 (0 = at the high, negative = below it)",
}
OPS = {">=", "<=", ">", "<"}

PROPOSE_SYSTEM = (
    "You translate a trader's plain-English screening idea into a strict JSON spec. Available per-stock features "
    "(each is z-scored across the universe on the day): " + "; ".join(f"{k} = {v}" for k, v in FEATURES.items()) + ". "
    "Spec format: {\"name\": \"<snake_case, <=24 chars>\", \"terms\": [{\"feature\": <name>, \"weight\": <number; positive = "
    "prefer high values, negative = prefer low>}], \"filters\": [{\"feature\": <name>, \"op\": \">=|<=|>|<\", "
    "\"value\": \"median\" | <number in raw units, e.g. 0.05 for 5%>}], \"rationale\": \"<=30 words\"}. "
    "Stocks are ranked by the weighted sum; filters exclude stocks first. Use ONLY listed features. Keep it to 1-3 "
    "terms and 0-2 filters; simple rules overfit less. Evidence you may lean on: 12-2 month momentum is positive; "
    "1-week and 1-month returns REVERSE (losers bounce); 20-day momentum is reversal territory, not momentum. "
    "If the idea cannot be expressed with these features, return {\"error\": \"<why>\"}. Output ONLY JSON."
)


# ----------------------------------------------------------------------------- spec -> ranking rule
def validate(spec: dict) -> dict:
    if "error" in spec:
        raise ValueError(f"agent could not express the idea: {spec['error']}")
    name = str(spec.get("name", "proposal")).strip().replace(" ", "_")[:24] or "proposal"
    terms = spec.get("terms") or []
    if not terms:
        raise ValueError("spec has no terms")
    for t in terms:
        if t.get("feature") not in FEATURES:
            raise ValueError(f"unknown feature {t.get('feature')!r}; allowed: {list(FEATURES)}")
        t["weight"] = float(t.get("weight", 1.0))
    for f in spec.get("filters") or []:
        if f.get("feature") not in FEATURES or f.get("op") not in OPS:
            raise ValueError(f"bad filter {f}")
        if f.get("value") != "median":
            f["value"] = float(f["value"])
    return {"name": name, "terms": terms, "filters": spec.get("filters") or [], "rationale": str(spec.get("rationale", ""))[:200]}


def compile_rule(spec: dict):
    """-> callable(per-symbol DataFrame) -> Series of scores (excluded names get -inf), like VARIANTS entries."""
    terms, filters = spec["terms"], spec["filters"]

    def rule(d: pd.DataFrame) -> pd.Series:
        score = sum(t["weight"] * _z(d[t["feature"]]) for t in terms)
        mask = pd.Series(True, index=d.index)
        for f in filters:
            col = d[f["feature"]]
            thr = col.median() if f["value"] == "median" else f["value"]
            mask &= {">=": col >= thr, "<=": col <= thr, ">": col > thr, "<": col < thr}[f["op"]]
        return score.where(mask, -np.inf)
    return rule


def describe(spec: dict) -> str:
    parts = [f"{t['weight']:+g} x z({t['feature']})" for t in spec["terms"]]
    s = "score = " + " ".join(parts)
    if spec["filters"]:
        s += "  where " + " and ".join(f"{f['feature']} {f['op']} {f['value']}" for f in spec["filters"])
    return s


# ----------------------------------------------------------------------------- verdict (deterministic)
def verdict(res: pd.DataFrame, name: str, baseline: str, split_date) -> dict:
    out = {"name": name, "timings": {}}
    train, test = res[res.date < split_date], res[res.date >= split_date]
    for timing in ("oo", "cc"):
        g_tr = train[(train.variant == name) & (train.timing == timing)]
        g_te = test[(test.variant == name) & (test.timing == timing)]
        b_tr = train[(train.variant == baseline) & (train.timing == timing)]
        b_te = test[(test.variant == baseline) & (test.timing == timing)]
        r_te = test[(test.variant == "random") & (test.timing == timing)]
        if g_tr.empty:
            continue
        e1, t1, gr1, dd1 = bt._stats(g_tr)
        e2, t2, gr2, _ = bt._stats(g_te) if not g_te.empty else (0.0, 0.0, 1.0, 0.0)
        be1, bt1, _, _ = bt._stats(b_tr) if not b_tr.empty else (0.0, 0.0, 1.0, 0.0)
        be2 = bt._stats(b_te)[0] if not b_te.empty else 0.0
        re2 = bt._stats(r_te)[0] if not r_te.empty else 0.0
        out["timings"][timing] = {"train_edge": e1, "train_t": t1, "train_growth": gr1, "train_maxdd": dd1,
                                  "test_edge": e2, "test_t": t2, "test_growth": gr2,
                                  "baseline_train_edge": be1, "baseline_train_t": bt1, "baseline_test_edge": be2,
                                  "random_test_edge": re2, "passes": bool(t1 >= 3 and e2 > 0),
                                  "beats_baseline": bool(e1 > be1 and e2 > be2)}
    out["passes_any"] = any(v["passes"] for v in out["timings"].values())
    return out


def print_verdict(v: dict, spec: dict, baseline: str, n_tried: int):
    print(f"\n=== Proposal '{v['name']}': {describe(spec)}")
    if spec.get("rationale"):
        print(f"    rationale: {spec['rationale']}")
    print(f"{'timing':>6} | {'train edge':>10} {'t':>6} {'growth':>7} {'maxDD':>7} | {'test edge':>9} {'t':>6} | {'baseline tr/te':>15} {'random te':>9}")
    for timing, r in v["timings"].items():
        flag = "  PASSES BAR" if r["passes"] else ""
        print(f"{timing:>6} | {r['train_edge']*100:>+9.2f}% {r['train_t']:>6.2f} {r['train_growth']:>6.2f}x {r['train_maxdd']*100:>6.1f}% | "
              f"{r['test_edge']*100:>+8.2f}% {r['test_t']:>6.2f} | {r['baseline_train_edge']*100:>+6.2f}%/{r['baseline_test_edge']*100:>+6.2f}% "
              f"{r['random_test_edge']*100:>+8.2f}%{flag}")
    exp_max_t = math.sqrt(2 * math.log(max(n_tried, 2)))
    print(f"\nBar: train t >= 3 AND positive hold-out edge. Baseline = current variant '{baseline}'. "
          f"This is proposal #{n_tried}; with {n_tried} tries and NO real edge, the best train t you'd expect by luck is about {exp_max_t:.1f}.")
    if v["passes_any"]:
        print("Verdict: passes on at least one timing. Re-run with --cost-bps 20 and check it still passes before believing it; "
              "then set screener.variant only after a second look next month (edges that survive a month of new data are rarer than ones that pass once).")
    else:
        print("Verdict: does not pass. That is the normal outcome; most ideas are noise after costs. Logged so you don't re-test it by accident.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="*", help="the screening idea in plain English")
    ap.add_argument("--spec", default=None, help="JSON file with a hand-written spec (skips the LLM)")
    ap.add_argument("--list", action="store_true", help="show all proposals tried so far")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--online", action="store_true", help="refresh the data lake first (needs Alpaca keys)")
    args = ap.parse_args()
    s = Settings()
    log = s.state_dir / "proposals.jsonl"
    tried = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()] if log.exists() else []

    if args.list:
        if not tried:
            print("no proposals yet"); return
        print(f"{'when':>16} {'name':>24} {'train t':>8} {'test edge':>10} {'pass':>5}  rule")
        for p in tried:
            best = max(p["verdict"]["timings"].values(), key=lambda r: r["train_t"], default={})
            print(f"{p['when']:>16} {p['spec']['name']:>24} {best.get('train_t', 0):>8.2f} {best.get('test_edge', 0)*100:>+9.2f}% "
                  f"{'yes' if p['verdict']['passes_any'] else 'no':>5}  {describe(p['spec'])}")
        return

    if args.spec:
        spec = validate(json.load(open(args.spec, encoding="utf-8")))
        idea = f"(spec file {args.spec})"
    else:
        idea = " ".join(args.idea).strip()
        if not idea:
            ap.error("give an idea in plain English, or --spec file.json, or --list")
        rcfg = s.cfg["research"]
        agents = Agents(s.anthropic_key, rcfg, CostTracker(s.state_dir, s.cfg["pricing"], rcfg["daily_usd_cap"]),
                        cache_path=s.state_dir / "research_cache.json", names=s.names)
        prior = ("Already tried (do not repeat exactly): " + "; ".join(describe(p["spec"]) for p in tried[-10:])) if tried else ""
        raw = agents._call(rcfg["research_model"], PROPOSE_SYSTEM, f"Idea: {idea}\n{prior}", 300)
        if not raw:
            print("the agent returned nothing usable"); return
        spec = validate({k: v for k, v in raw.items() if not k.startswith("_")})
    print(f"idea: {idea}\nspec: {describe(spec)}")

    lake = bt.DataLake(s.root)
    if args.online:
        from trader.broker import Broker
        bars = lake.update(Broker(s.creds("large")), bt.load_symbols(s), 5)
    else:
        bars = lake.load()
    if bars.empty:
        print("no data lake — run `python backtest.py --sweep` once (or --online here) first"); return
    baseline = s.cfg["screener"].get("variant", "pullback_in_uptrend")
    name = spec["name"]
    if name in VARIANTS and name != baseline:
        name = spec["name"] = name + "_p"
    VARIANTS[name] = compile_rule(spec)
    top = args.top or s.candidates_per_cycle("large")
    res = bt.run(bars, s, top, [name, baseline], args.cost_bps, s.cfg["screener"].get("universe_size", 500))
    if res.empty:
        print("not enough history"); return
    dates = sorted(res["date"].unique())
    split_date = pd.Timestamp(dates[max(0, len(dates) - bt.TEST_DAYS // bt.HORIZON)])
    v = verdict(res, name, baseline, split_date)
    print_verdict(v, spec, baseline, len(tried) + 1)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"when": datetime.now().strftime("%Y-%m-%d %H:%M"), "idea": idea, "spec": spec, "top": top,
                            "cost_bps": args.cost_bps, "verdict": v}, default=float) + "\n")


if __name__ == "__main__":
    main()
