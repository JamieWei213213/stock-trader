"""Offline tests (v2): fake broker + fake Claude, real screener/executor/scorecard logic.
Run: python -m tests.test_offline
"""
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from trader.agents import Agents, anonymize
from trader.broker import Snapshot
from trader.costs import CostTracker
from trader.executor import Executor
from trader.scorecard import Scorecard
from trader.screener import VARIANTS, pick_candidates, score_table
from trader.settings import Settings


def fake_bars(symbols, n=320, seed=1):
    rng = np.random.default_rng(seed)
    frames = []
    idx = pd.date_range(end="2026-09-02", periods=n, freq="B", tz="UTC")
    for s in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
        df = pd.DataFrame({"open": close * (1 + rng.normal(0, 0.003, n)), "high": close * 1.01, "low": close * 0.99,
                           "close": close, "volume": rng.integers(5e6, 5e7, n)}, index=idx)
        df.index.name = "timestamp"
        df["symbol"] = s
        frames.append(df.reset_index().set_index(["symbol", "timestamp"]))
    return pd.concat(frames)


class FakeBroker:
    def __init__(self, bars):
        self.bars, self.orders = bars, []
    def daily_bars(self, symbols, days):
        return self.bars[self.bars.index.get_level_values("symbol").isin(symbols)]
    def buy_bracket(self, sym, qty, stop, target): self.orders.append(("BUY", sym, qty, stop, target))
    def close_position(self, sym): self.orders.append(("SELL", sym))


def test_all():
    s = Settings()
    syms = s.symbols[:25]
    bars = fake_bars(syms)
    fb = FakeBroker(bars)

    # --- screener: every variant produces a ranked table with atr ---
    for v in VARIANTS:
        table = score_table(bars, syms, s.is_volatile, 1e7, v)
        assert len(table) > 10 and "atr" in table and table["score"].is_monotonic_decreasing, v
    table = score_table(bars, syms, s.is_volatile, 1e7, "pullback_in_uptrend")
    full_median = score_table(bars, syms, s.is_volatile, 1e7, "mom_12_2")["mom_12_2"].median()
    assert (table["mom_12_2"] >= full_median - 1e-9).all(), "pullback_in_uptrend must only rank top-half momentum"
    cands = pick_candidates(table, 6, ["AAPL"])
    assert "AAPL" in cands
    cands = [c for c in cands if c in table.index]  # a held name may not be in the ranked table
    print("screener ok:", cands)

    # --- executor: ATR sizing, heat cap, PDT, hold days, earnings blackout ---
    tmp = Path(tempfile.mkdtemp())
    rules = s.account_cfg("large")
    snap = Snapshot(equity=100000, cash=100000, buying_power=400000, daytrade_count=0, pattern_day_trader=False, positions=[])
    prices = {c: float(table.loc[c, "price"]) for c in cands}
    atrs = {c: float(table.loc[c, "atr"]) for c in cands}
    ex = Executor("large", rules, fb, tmp, s.is_volatile, earnings_soon=lambda sym: sym == cands[0])
    ex.apply({"actions": [{"symbol": c, "action": "buy", "reason": "t"} for c in cands]}, snap, prices, atrs)
    buys = [o for o in fb.orders if o[0] == "BUY"]
    assert buys and all(o[1] != cands[0] for o in buys), "earnings blackout failed"
    for _, sym, qty, stop, target in buys:
        p, a = prices[sym], atrs[sym]
        assert abs(stop - (p - 2 * a)) < 0.02 and abs(target - (p + 4 * a)) < 0.02, "stop/target not ATR-based"
        assert qty * (p - stop) <= 100000 * rules["risk_per_trade_pct"] + p, "risk per trade exceeded"
        assert qty * p <= 100000 * rules["max_position_pct"] + p, "position cap exceeded"
    total_risk = sum(qty * (prices[sym] - stop) for _, sym, qty, stop, _ in buys)
    assert total_risk <= 100000 * rules["max_heat_pct"] + 1, "heat cap exceeded"
    print(f"executor ok: {len(buys)} buys, total open risk ${total_risk:,.0f} (cap ${100000*rules['max_heat_pct']:,.0f})")

    # small account: PDT guard + hold-days exit
    rules_s = s.account_cfg("small")
    fb.orders.clear()
    snap_s = Snapshot(equity=1000, cash=1000, buying_power=1000, daytrade_count=0, pattern_day_trader=False, positions=[])
    Executor("small", rules_s, fb, tmp, s.is_volatile).apply(
        {"actions": [{"symbol": c, "action": "buy", "reason": "t"} for c in cands]}, snap_s, prices, atrs)
    sbuys = [o for o in fb.orders if o[0] == "BUY"]
    assert sbuys, "small account should afford at least one name"
    held_sym = sbuys[0][1]
    pos = [{"symbol": held_sym, "qty": sbuys[0][2], "avg_entry": prices[held_sym], "current": prices[held_sym],
            "market_value": 1, "unrealized_pl": 0, "unrealized_plpc": 0}]
    snap2 = Snapshot(equity=1000, cash=500, buying_power=500, daytrade_count=0, pattern_day_trader=False, positions=pos)
    fb.orders.clear()
    Executor("small", rules_s, fb, tmp, s.is_volatile).apply(
        {"actions": [{"symbol": held_sym, "action": "sell", "reason": "x"}]}, snap2, prices, atrs)
    assert not fb.orders, "PDT guard failed"
    st = json.loads((tmp / "positions_small.json").read_text())
    st[held_sym]["date"] = (date.today() - timedelta(days=30)).isoformat()
    (tmp / "positions_small.json").write_text(json.dumps(st))
    Executor("small", rules_s, fb, tmp, s.is_volatile).apply(None, snap2, prices, atrs)
    assert ("SELL", held_sym) in fb.orders
    print("PDT guard + max_hold_days ok")

    # --- agents: anonymization + cache + numeric output ---
    txt = anonymize("Robinhood Markets Inc (HOOD) jumped as Robinhood's app grew; $HOOD up", "HOOD", "Robinhood Markets, Inc. Class A Common Stock")
    assert "HOOD" not in txt and "Robinhood" not in txt, txt
    costs = CostTracker(tmp, s.cfg["pricing"], 1.0)
    a = Agents("x", s.cfg["research"], costs, cache_path=tmp / "c.json", names={"AAA": "Alpha Corp"})
    resp = MagicMock(); resp.usage.input_tokens = 100; resp.usage.output_tokens = 50
    resp.content = [MagicMock(text='{"expected_5d_return_pct": "2.5", "confidence": 0.7, "bias":"bullish","catalyst":"x","risk":"y"}')]
    a.client.messages.create = MagicMock(return_value=resp)
    feats = table.loc[cands[0]].to_dict()
    news = [{"time": "t", "headline": "Alpha Corp beats", "content": "Alpha said...", "source": "s"}]
    r1 = a.research_stock("AAA", feats, news); r2 = a.research_stock("AAA", feats, news)
    assert r1["expected_5d_return_pct"] == 2.5 and r2["_cached"] and a.client.messages.create.call_count == 1
    sent = a.client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Alpha" not in sent and "AAA" not in sent, sent
    print("agents ok: anonymized, cached, numeric")

    # --- scorecard ---
    sc = Scorecard(tmp)
    sc.log("large", [{"symbol": syms[0], "rank": 1, "price": 100, "expected_5d_pct": 2.0, "confidence": 0.8, "bias": "bullish", "approved": True, "acted": True},
                     {"symbol": syms[1], "rank": 2, "price": 50, "expected_5d_pct": -1.0, "confidence": 0.5, "bias": "neutral", "approved": False, "acted": False}])
    df = pd.read_csv(tmp / "scorecard.csv")
    assert len(df) == 2 and df["ret_5d"].isna().all()
    print("scorecard ok")


if __name__ == "__main__":
    test_all()
    print("\nALL OFFLINE TESTS PASSED")
