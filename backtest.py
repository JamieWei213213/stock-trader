"""Fair backtest of the screener (v2). Answers: do the top-N picks beat the universe, honestly?

What makes it fair (see RESEARCH.md §2):
  * point-in-time universe: at each rebalance date the eligible set is the top `universe_size`
    names by trailing 20-day dollar volume AS OF THAT DATE, drawn from a wider pool
    (universe_wide.yaml if present) — not today's winners.
  * hold-out: the last 12 months are TEST and are reported separately; tune only on TRAIN.
  * costs: `--cost-bps` round-trip (default 10) charged on every rebalance of the picks.
  * bar: with many variants tried, a t-stat >= 3 on TRAIN that also holds sign on TEST is the bar.
  * entry timing: open-to-open (our 6:05am orders) and close-to-close (a 12:45pm cycle) both reported.

Usage:
  python backtest.py --sweep                 # all variants, both timings, train + test
  python backtest.py --sweep --offline       # reuse data/daily_bars.parquet
  python backtest.py --variant reversal_5d   # one variant in detail
Run this on your PC, not the $4 VPS (500+ symbols x 4 years needs ~1GB RAM).
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
import yaml

from trader.datalake import DataLake
from trader.screener import VARIANTS, features_panel, score_at
from trader.settings import Settings

HORIZON = 5          # trading days held per rebalance (= step, so returns don't overlap)
TEST_DAYS = 252      # last year is hold-out


def run(bars: pd.DataFrame, s: Settings, top: int, variants: list[str], cost_bps: float,
        universe_n: int) -> pd.DataFrame:
    feats = features_panel(bars)
    close, opn = feats["price"], feats["open"]
    dates = close.index
    rng = np.random.default_rng(0)
    first = 260  # need 252 days for mom_12_2
    rows = []
    sizes: list[int] = []
    for i in range(first, len(dates) - HORIZON - 1, HORIZON):
        asof = dates[i]
        tables = {v: score_at(feats, asof, s.is_volatile, s.cfg["screener"]["min_avg_dollar_volume"], v, universe_n)
                  for v in variants}
        base = tables[variants[0]]
        if base.empty or len(base) < top * 2:
            continue
        universe = list(base.index)
        picks_by = {v: list(t.index[:top]) for v, t in tables.items() if not t.empty}
        sizes.append(len(universe))
        picks_by["random"] = list(rng.choice(universe, size=top, replace=False))
        # close-to-close: buy at today's close, sell at close HORIZON days later
        fwd_cc = close.iloc[i + HORIZON] / close.iloc[i] - 1
        # open-to-open: buy at TOMORROW's open (we decide pre-open), sell at open HORIZON days after that
        fwd_oo = opn.iloc[i + 1 + HORIZON] / opn.iloc[i + 1] - 1
        for v, picks in picks_by.items():
            for timing, fwd in (("cc", fwd_cc), ("oo", fwd_oo)):
                pr = fwd[picks].dropna()
                if pr.empty:
                    continue
                rows.append({
                    "variant": v, "timing": timing, "date": asof,
                    "picks_ret": float(pr.mean()) - cost_bps / 1e4,
                    "universe_ret": float(fwd[universe].dropna().mean()),
                    "hit_rate": float((pr > 0).mean()),
                })
    out = pd.DataFrame(rows)
    if sizes:
        print(f"eligible names per rebalance: median {int(np.median(sizes))}, min {min(sizes)}, max {max(sizes)} "
              f"(if this is far below universe_size, the liquidity filter is too strict)")
    return out


def _stats(g: pd.DataFrame) -> tuple[float, float, float, float]:
    edge = g["picks_ret"] - g["universe_ret"]
    t = edge.mean() / (edge.std() / np.sqrt(len(edge))) if len(edge) > 2 and edge.std() else 0.0
    curve = (1 + g["picks_ret"]).cumprod()
    dd = float((curve / curve.cummax() - 1).min()) if len(curve) else 0.0
    return float(edge.mean()), float(t), float(curve.iloc[-1]) if len(curve) else 1.0, dd


def summarize_sweep(res: pd.DataFrame, split_date) -> None:
    res = res.sort_values("date")
    train, test = res[res.date < split_date], res[res.date >= split_date]
    n_tr, n_te = train["date"].nunique(), test["date"].nunique()
    print(f"\n=== Fair sweep: top picks vs point-in-time universe, {HORIZON}-day holds, costs included ===")
    print(f"TRAIN {train.date.min().date()} → {split_date.date()} ({n_tr} rebalances)   "
          f"TEST {split_date.date()} → {test.date.max().date()} ({n_te} rebalances, hold-out)")
    print(f"\n{'variant':>20} {'timing':>6} | {'train edge':>10} {'t':>6} {'growth':>7} {'maxDD':>7} | {'test edge':>9} {'t':>6} {'growth':>7}")
    for v in list(dict.fromkeys(res["variant"])):
        for timing in ("oo", "cc"):
            g_tr = train[(train.variant == v) & (train.timing == timing)]
            g_te = test[(test.variant == v) & (test.timing == timing)]
            if g_tr.empty:
                continue
            e1, t1, gr1, dd1 = _stats(g_tr)
            e2, t2, gr2, _ = _stats(g_te) if not g_te.empty else (0, 0, 1, 0)
            flag = " <-- passes bar" if t1 >= 3 and e2 > 0 else ""
            print(f"{v:>20} {timing:>6} | {e1*100:>+9.2f}% {t1:>6.2f} {gr1:>6.2f}x {dd1*100:>6.1f}% | "
                  f"{e2*100:>+8.2f}% {t2:>6.2f} {gr2:>6.2f}x{flag}")
    print("\nedge = picks minus universe per 5-day hold, after costs. oo = buy next open (6:05am cycle); "
          "cc = buy at close (12:45pm cycle).\nBar: train t >= 3 AND positive test edge. 'random' is what luck produces. "
          "Remaining bias: names delisted before today are missing from the pool.")


def summarize_one(res: pd.DataFrame, variant: str, split_date) -> None:
    for timing in ("oo", "cc"):
        g = res[(res.variant == variant) & (res.timing == timing)].sort_values("date")
        if g.empty:
            continue
        print(f"\n--- {variant} / {timing} ---")
        for name, part in (("train", g[g.date < split_date]), ("test", g[g.date >= split_date])):
            if part.empty:
                continue
            e, t, gr, dd = _stats(part)
            u = (1 + part["universe_ret"]).cumprod().iloc[-1]
            print(f"{name:>5}: n={len(part):3d} edge {e*100:+.2f}%/hold  t={t:.2f}  hit {part.hit_rate.mean()*100:.0f}%  "
                  f"picks {gr:.2f}x vs universe {u:.2f}x  maxDD {dd*100:.1f}%")


def load_symbols(s: Settings) -> list[str]:
    wide = s.root / "universe_wide.yaml"
    syms = list(s.symbols)
    if wide.exists():
        with open(wide, encoding="utf-8") as f:
            syms += [x["symbol"] for x in yaml.safe_load(f)["stocks"]]
        print(f"using wide pool from universe_wide.yaml")
    return sorted(set(syms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--cost-bps", type=float, default=10.0, help="round-trip cost per rebalance")
    args = ap.parse_args()

    s = Settings()
    lake = DataLake(s.root)
    symbols = load_symbols(s)
    if args.offline:
        bars = lake.load()
    else:
        from trader.broker import Broker
        bars = lake.update(Broker(s.creds("large")), symbols, args.years)
    if bars.empty:
        print("no data — run without --offline first (needs Alpaca keys in .env)")
        return
    top = args.top or s.cfg["screener"]["candidates_per_cycle"]
    universe_n = s.cfg["screener"].get("universe_size", 500)
    variants = list(VARIANTS) if args.sweep else [args.variant or s.cfg["screener"].get("variant", "pullback_in_uptrend")]
    print(f"{bars.index.get_level_values('symbol').nunique()} symbols, top {top}, point-in-time universe {universe_n}, "
          f"cost {args.cost_bps} bps, variants {variants}")
    res = run(bars, s, top, variants, args.cost_bps, universe_n)
    if res.empty:
        print("not enough history (need ~260 trading days before the first rebalance)")
        return
    dates = sorted(res["date"].unique())
    split_date = pd.Timestamp(dates[max(0, len(dates) - TEST_DAYS // HORIZON)])
    if args.sweep:
        summarize_sweep(res, split_date)
    else:
        summarize_one(res, variants[0], split_date)
    out = s.reports_dir / f"backtest_{date.today()}.csv"
    res.to_csv(out, index=False)
    print(f"details: {out}")


if __name__ == "__main__":
    main()
