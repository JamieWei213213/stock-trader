"""Offline tests for v3: regime monitor, decision rule, risk reviewer, orchestrator (fake agents with
timeouts/failures), filings extraction on a synthetic EDGAR payload, memory/post-mortem agents, executor
v3 guards, scorecard migration.   Run: python -m tests.test_v3
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from tests.test_offline import FakeBroker, fake_bars
from trader.agents import Agents
from trader.broker import Snapshot
from trader.costs import CostTracker
from trader.decide import decide, veto_reason
from trader.executor import Executor
from trader.filings import FilingsAgent, extract_quarters, render_table
from trader.journal import Journal
from trader.memory import MemoryAgent, PostMortemAgent
from trader.orchestrator import Candidate, CycleState, Orchestrator, build_candidates, combine, risk_review
from trader.regime import RegimeView, assess
from trader.scorecard import FIELDS, Scorecard
from trader.screener import pick_candidates, score_table
from trader.settings import Settings


def fake_agents(tmp, s, reply='{"expected_5d_return_pct": 2.0, "confidence": 0.7, "bias":"bullish","catalyst":"c","risk":"r"}'):
    costs = CostTracker(tmp, s.cfg["pricing"], 5.0)
    a = Agents("x", s.cfg["research"], costs, cache_path=tmp / "cache.json", names={})
    resp = MagicMock(); resp.usage.input_tokens = 100; resp.usage.output_tokens = 40
    resp.content = [MagicMock(text=reply)]
    a.client.messages.create = MagicMock(return_value=resp)
    return a


def test_regime():
    cfg = {"ma_days": 200, "vol_window_days": 20, "risk_mult_below_ma": 0.5, "vol_spike_mult": 2.0, "symbol": "SPY"}
    idx = pd.date_range(end="2026-09-01", periods=400, freq="B")
    up = pd.Series(np.linspace(100, 200, 400), index=idx)
    r = assess(up, cfg); assert r.label == "risk-on" and r.risk_mult == 1.0 and r.allow_new_buys, r
    down = pd.Series(np.linspace(200, 100, 400), index=idx)
    r = assess(down, cfg); assert r.label == "risk-off" and r.risk_mult == 0.5 and r.allow_new_buys, r
    rng = np.random.default_rng(0)
    crash = pd.Series(np.r_[np.linspace(150, 160, 380), 160 * np.cumprod(1 + rng.normal(-0.03, 0.06, 20))], index=idx)
    r = assess(crash, cfg); assert r.label == "crisis" and not r.allow_new_buys, r
    assert assess(up[:100], cfg).label == "unknown"
    print("regime ok")


def test_decide():
    dcfg = {"veto_below_pct": -0.5, "veto_bearish_conf": 0.6, "sell_below_pct": -1.5, "rotation_min_edge_pct": 1.0, "max_swaps_per_cycle": 1}
    rcfg = {"min_confidence": 0.6}
    f = {"A": {"expected_5d_return_pct": 2.0, "confidence": 0.7, "bias": "bullish"},
         "B": {"expected_5d_return_pct": -1.0, "confidence": 0.5, "bias": "bearish"},        # vetoed: below -0.5
         "C": {"expected_5d_return_pct": 0.0, "confidence": 0.8, "bias": "bearish"},         # vetoed: bearish + conf
         "D": {"expected_5d_return_pct": 0.3, "confidence": 0.4, "bias": "neutral"},         # passes (screener pick, weak but not vetoed)
         "H1": {"expected_5d_return_pct": -2.0, "confidence": 0.7, "bias": "bearish"},       # held -> sell
         "H2": {"expected_5d_return_pct": 0.5, "confidence": 0.5, "bias": "neutral"},        # held -> hold
         "H3": {"expected_5d_return_pct": 1.0, "confidence": 0.6, "thesis_status": "broken", "memory_note": "catalyst failed"}}
    d = decide(["A", "B", "C", "D", "E"], f, ["H1", "H2", "H3"], {"max_positions": 3}, dcfg, rcfg)
    acts = {a["symbol"]: a["action"] for a in d["actions"]}
    assert acts["H1"] == "sell" and acts["H2"] == "hold" and acts["H3"] == "sell", acts
    assert acts["A"] == "buy" and acts["D"] == "buy" and "B" not in acts and "C" not in acts, acts   # 2 slots freed
    assert "E" in d["vetoed"] and "no forecast" in d["vetoed"]["E"]
    # rotation: full book, newcomer must beat weakest held by >= 1%
    f2 = {"N": {"expected_5d_return_pct": 2.5, "confidence": 0.7, "bias": "bullish"},
          "W": {"expected_5d_return_pct": 1.0, "confidence": 0.6}, "S": {"expected_5d_return_pct": 3.0, "confidence": 0.6}}
    d = decide(["N"], f2, ["W", "S"], {"max_positions": 2}, dcfg, rcfg)
    acts = {a["symbol"]: a["action"] for a in d["actions"]}
    assert acts == {"W": "sell", "S": "hold", "N": "buy"}, acts
    f2["N"]["expected_5d_return_pct"] = 1.5
    d = decide(["N"], f2, ["W", "S"], {"max_positions": 2}, dcfg, rcfg)
    assert all(a["action"] != "buy" for a in d["actions"]) and "N" in d["vetoed"]
    # rank_by: forecast orders passers by the agents' number instead of screener rank
    f3 = {"W1": {"expected_5d_return_pct": 0.2, "confidence": 0.3}, "S1": {"expected_5d_return_pct": 3.0, "confidence": 0.7}}
    d = decide(["W1", "S1"], f3, [], {"max_positions": 1}, dcfg, rcfg)
    assert [a["symbol"] for a in d["actions"] if a["action"] == "buy"] == ["W1"]
    d = decide(["W1", "S1"], f3, [], {"max_positions": 1}, dict(dcfg, rank_by="forecast"), rcfg)
    assert [a["symbol"] for a in d["actions"] if a["action"] == "buy"] == ["S1"]
    # regime freeze
    d = decide(["A"], f, [], {"max_positions": 3}, dcfg, rcfg, regime=RegimeView("crisis", 0.5, False))
    assert not d["actions"] and "regime" in d["vetoed"]["A"]
    assert veto_reason(None, dcfg)
    print("decision rule ok")


def test_combine_and_review():
    news = {"symbol": "X", "expected_5d_return_pct": 2.0, "confidence": 0.6, "bias": "bullish", "catalyst": "c", "risk": "r"}
    c = combine(news, {"earnings_direction": "up", "confidence": 0.8, "tilt_pct": 0.4}, {"thesis_status": "intact", "adjustment_pct": 0.2, "note": "n"})
    assert abs(c["expected_5d_return_pct"] - 2.6) < 1e-9 and c["confidence"] == 0.65 and c["thesis_status"] == "intact", c
    c = combine(news, {"earnings_direction": "down", "confidence": 0.8, "tilt_pct": -0.4}, None)
    assert c["confidence"] == 0.5 and c["news_exp"] == 2.0
    assert combine(None, {"tilt_pct": 1}, None) is None
    # bias flips when the combined number crosses the neutral band
    c = combine({"expected_5d_return_pct": 0.4, "confidence": 0.6, "bias": "neutral"}, {"tilt_pct": 0.5}, None)
    assert c["bias"] == "bullish"
    # risk reviewer: industry cap + correlation
    idx = pd.date_range(end="2026-09-01", periods=80, freq="B")
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.02, 80)
    close = pd.DataFrame({k: 100 * np.cumprod(1 + base + rng.normal(0, 0.003, 80)) for k in ("H1", "H2", "T")}, index=idx)
    close["U"] = 100 * np.cumprod(1 + rng.normal(0, 0.02, 80))
    close["I"] = 100 * np.cumprod(1 + rng.normal(0, 0.02, 80))
    inds = {"H1": "semis", "H2": "semis", "H3": "semis", "I": "semis", "U": "banks"}
    acts = [{"symbol": x, "action": "buy"} for x in ("T", "U", "I")]
    kept, dropped = risk_review(acts, ["H1", "H2", "H3"], inds, close, {"max_per_industry": 3, "max_corr_with_book": 0.75})
    assert "T" in dropped and "correlation" in dropped["T"], dropped
    assert "I" in dropped and "industry cap" in dropped["I"], dropped
    assert [a["symbol"] for a in kept] == ["U"]
    print("forecaster + risk reviewer ok")


def test_orchestrator():
    tmp = Path(tempfile.mkdtemp())
    s = Settings()
    a = fake_agents(tmp, s)
    feats = {"price": 100.0, "ret_1d": 0.0, "ret_5d": -0.03, "ret_20d": 0.05, "mom_12_2": 0.3, "vol_surge": 1.1,
             "atr_pct": 0.02, "atr": 2.0, "avg_dollar_vol": 5e7, "dist_20d_high": -0.04}
    calls = {"news": 0, "fil": 0, "mem": 0}

    def news_fetch(sym):
        calls["news"] += 1
        if sym == "SLOW":
            time.sleep(3)          # -> timeout
        if sym == "BOOM":
            raise RuntimeError("news api down")
        return [{"time": "t", "headline": f"{sym} news", "content": "body", "source": "s"}]

    class FakeFilings:
        def analyze(self, sym):
            calls["fil"] += 1
            if sym == "NOFIL":
                return {"symbol": sym, "error": "no CIK", "tilt_pct": 0.0}
            return {"symbol": sym, "earnings_direction": "up", "confidence": 0.8, "tilt_pct": 0.4, "_cached": sym == "CACHED", "_cost_usd": 0.001}

    class FakeMemory:
        def review(self, sym):
            calls["mem"] += 1
            if sym == "BROKEN":
                return {"symbol": sym, "thesis_status": "broken", "adjustment_pct": -0.5, "note": "catalyst failed"}
            return {"symbol": sym, "thesis_status": "none", "adjustment_pct": 0.0, "note": "no history", "_cached": True}

    cfg = {"timeout_s": 1, "retries": 1, "news": {"enabled": True}, "filings": {"enabled": True}, "memory": {"enabled": True}}
    orch = Orchestrator(a, cfg, filings_agent=FakeFilings(), memory_agent=FakeMemory(), news_fetch=news_fetch, rcfg=s.cfg["research"])
    syms = ["GOOD", "SLOW", "BOOM", "NOFIL", "CACHED", "BROKEN"]
    state = CycleState("large", "entry", {}, RegimeView("risk-on", 1.0, True),
                       [Candidate(x, i + 1, dict(feats), held=(x == "BROKEN")) for i, x in enumerate(syms)])
    t0 = time.time(); orch.run(state); dt = time.time() - t0
    assert dt < 6, f"agents did not run concurrently ({dt:.1f}s)"   # SLOW alone = 2 attempts x 1s timeout + retry sleep
    assert "GOOD" in state.forecasts and abs(state.forecasts["GOOD"]["expected_5d_return_pct"] - 2.4) < 1e-9
    assert "SLOW" not in state.forecasts and "BOOM" not in state.forecasts, "failed news must mean no forecast"
    fails = [r for r in state.agent_log if not r.ok]
    assert {(r.agent, r.symbol) for r in fails} == {("news", "SLOW"), ("news", "BOOM")}, fails
    assert all(r.attempts == 2 for r in fails), "should retry once"
    assert state.forecasts["NOFIL"].get("filings_dir") is None and state.forecasts["NOFIL"]["expected_5d_return_pct"] == 2.0
    assert state.forecasts["BROKEN"]["thesis_status"] == "broken"
    summ = state.summary()
    assert summ["agents"]["news"]["failed"] == 2 and summ["agents"]["filings"]["cached"] == 1, summ
    # decision + review on top
    d = orch.decide(state, {"max_positions": 12}, s.cfg["decision"], s.cfg["research"], {}, None, s.cfg["risk"])
    acts = {x["symbol"]: x["action"] for x in d["actions"]}
    assert acts["BROKEN"] == "sell" and acts["GOOD"] == "buy" and "SLOW" in d["vetoed"] and "BOOM" in d["vetoed"], (acts, d["vetoed"])
    # manage mode: no buys, ever
    state.mode = "manage"
    d = orch.decide(state, {"max_positions": 12}, s.cfg["decision"], s.cfg["research"], {}, None, s.cfg["risk"])
    assert all(x["action"] != "buy" for x in d["actions"]) and d["vetoed"]["GOOD"] == "manage-only cycle"
    print(f"orchestrator ok: {len(syms)} candidates x 3 agents in {dt:.1f}s, {len(fails)} failures degraded gracefully")


def synthetic_facts(n=10, growth=0.05):
    """EDGAR companyfacts-shaped payload, realistic: 10-Qs carry CY####Q1-3 duration frames, the 10-K carries the
    ANNUAL CY#### frame (no Q4 duration frame), instants carry CY####Q#I. Revenue switches tag halfway (as real filers do)."""
    def q_entries(vals, instant=False, only=None):
        out, annual = [], {}
        for i, v in enumerate(vals):
            y, q = 2024 + i // 4, i % 4 + 1
            annual[y] = annual.get(y, 0) + v
            if q == 4 and not instant:
                if i == len(vals) - 1 or (i // 4) != ((i + 1) // 4):
                    out.append({"val": annual[y], "filed": f"{y + 1}-02-15", "frame": f"CY{y}", "form": "10-K"})
                continue
            if only and not only(y):
                continue
            out.append({"end": f"{y}-{q*3:02d}-28", "val": v, "accn": "x", "fy": y, "fp": f"Q{q}", "form": "10-Q",
                        "filed": f"{y}-{min(q*3+1, 12):02d}-15", "frame": f"CY{y}Q{q}{'I' if instant else ''}"})
        return out
    rev = [1000e6 * (1 + growth) ** i for i in range(n)]
    ni = [100e6 * (1 + growth * 1.5) ** i for i in range(n)]
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": q_entries(rev, only=lambda y: y <= 2024)}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": q_entries(rev, only=lambda y: y >= 2025)}},
        "NetIncomeLoss": {"units": {"USD": q_entries(ni)}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": q_entries([x / 50e6 for x in ni])}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": q_entries([x * 1.2 for x in ni])}},
        "Assets": {"units": {"USD": q_entries([5000e6] * n, instant=True) + [{"frame": "CY2025", "val": 1, "filed": "2026-02-01"}]}},
    }}}


def test_earnings_windows():
    from datetime import date, timedelta
    from unittest.mock import patch
    from trader.earnings import EarningsCalendar, _bdays
    assert _bdays(date(2026, 9, 7), date(2026, 9, 4)) == -1, "Friday is one trading day before Monday"
    assert _bdays(date(2026, 9, 7), date(2026, 9, 14)) == 5
    tmp = Path(tempfile.mkdtemp())
    cal = EarningsCalendar.__new__(EarningsCalendar)
    cal.dates, cal.blackout_days, cal.recent_days = {"FRI": "2026-09-04", "NEXT": "2026-09-14", "FAR": "2026-09-15", "OLD": "2026-09-01"}, 5, 1
    with patch("trader.earnings.date") as d:
        d.today.return_value = date(2026, 9, 7); d.fromisoformat = date.fromisoformat
        assert cal.soon("FRI") and cal.soon("NEXT") and not cal.soon("FAR") and not cal.soon("OLD")
    print("earnings windows ok (weekend-aware)")


def test_filings():
    tmp = Path(tempfile.mkdtemp())
    s = Settings()
    q = extract_quarters(synthetic_facts(), quarters=8)
    assert q["quarters"] == [(2024, 3), (2024, 4), (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2)], q["quarters"]
    rev = q["series"]["revenue"]
    assert all(v is not None for v in rev), "revenue must merge across the tag switch"
    assert abs(rev[1] - 1000e6 * 1.05 ** 3) < 1 and abs(rev[5] - 1000e6 * 1.05 ** 7) < 1, "Q4 must be derived as annual - Q1..Q3"
    assert q["series"]["eps_diluted"][1] is None, "EPS Q4 is not derivable"
    assert q["series"]["total_assets"][0] == 5000e6
    table = render_table(q)
    yoy = [l for l in table.split("\n") if l.startswith("revenue YoY")][0]
    assert yoy.split()[-1] == "+22%" and yoy.count("n/a") == 4, yoy     # 1.05^4 - 1 = 21.6%, first 4 quarters have no prior year
    assert "OCF/net inc" in table and "Q0(latest)" in table, table
    assert "2024" not in table.split("\n")[0], "no dates in the anonymized table"
    a = fake_agents(tmp, s, reply='{"earnings_direction": "Up", "confidence": 0.7, "quality_flags": ["ocf > ni"], "summary": "steady growth"}')
    edgar = MagicMock(); edgar.cik.return_value = 123; edgar.company_facts.return_value = synthetic_facts()
    fa = FilingsAgent(a, tmp, {"cache_days": 90, "quarters": 8, "weight_pct": 0.5}, edgar=edgar)
    (tmp / "filings").mkdir(exist_ok=True)
    r1 = fa.analyze("ZZZ"); r2 = fa.analyze("ZZZ")
    assert r1["earnings_direction"] == "up" and abs(r1["tilt_pct"] - 0.35) < 1e-9, r1
    assert r2["_cached"] and edgar.company_facts.call_count == 1, "second call must hit the cache"
    sent = a.client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "ZZZ" not in sent and "revenue" in sent
    edgar.cik.return_value = None
    assert fa.analyze("NOPE")["error"]
    print("filings agent ok:\n" + "\n".join("    " + l for l in table.split("\n")[:4]))


def test_memory_and_postmortem():
    tmp = Path(tempfile.mkdtemp())
    s = Settings()
    j = Journal(tmp)
    assert MemoryAgent(fake_agents(tmp, s), j, tmp, {"max_history": 6}).review("NEW")["thesis_status"] == "none"
    # seed a journal entry + scorecard row + closed trade
    from datetime import datetime
    jf = j.write_entry("large", "ABC", {"forecast": {"expected_5d_return_pct": 3.0, "confidence": 0.8, "catalyst": "guidance raise", "risk": "macro"},
                                       "prompt_sent": "the company raised guidance ..."}, datetime(2026, 8, 20, 12, 45))
    (tmp / "scorecard.csv").write_text("logged,account,symbol,rank,price,expected_5d_pct,confidence,bias,approved,acted,ret_1d,ret_5d,ret_10d\n"
                                       "2026-08-20 12:45,large,ABC,1,100,3.0,0.8,bullish,True,True,0.01,-0.04,-0.05\n")
    t = j.open_trade("large", "ABC", 10, 100.0, 96.0, 108.0, "screener pick", {"expected_5d_return_pct": 3.0, "confidence": 0.8, "catalyst": "guidance raise"}, str(jf), "oid")
    t["entry_time"] = "2026-08-20 12:45"
    j.close_trade(t, 96.0, "2026-08-24 10:00", "stop_loss")
    a = fake_agents(tmp, s, reply='{"mistake_type": "thesis_wrong", "avoidable": true, "lesson": "guidance was priced in; do not chase"}')
    pm = PostMortemAgent(a, j, tmp).write(t)
    assert pm["mistake_type"] == "thesis_wrong" and j.trades[0]["post_mortem"]["lesson"].startswith("guidance")
    assert (tmp / "lessons.jsonl").exists() and a.client.messages.create.call_count == 1
    assert PostMortemAgent(a, j, tmp).write(t) == pm and a.client.messages.create.call_count == 1, "post-mortem must run once per trade"
    a2 = fake_agents(tmp, s, reply='{"thesis_status": "broken", "adjustment_pct": -2, "note": "guidance catalyst failed"}')
    m = MemoryAgent(a2, j, tmp, {"max_history": 6})
    hist = m.history("ABC")
    assert any("realized 5d: -4.0%" in h for h in hist) and any("lesson: guidance" in h for h in hist), hist
    r = m.review("ABC")
    assert r["thesis_status"] == "broken" and r["adjustment_pct"] == -0.5 and r["history_n"] == 2, r
    print("memory + post-mortem ok")


def test_ask():
    import shutil
    from datetime import datetime
    import ask as A
    tmp = Path(tempfile.mkdtemp()); shutil.copy("config.yaml", tmp); shutil.copy("watchlist.yaml", tmp)
    s = Settings(tmp); j = Journal(s.state_dir)
    jf = j.write_entry("large", "CIEN", {"rank": 2, "forecast": {"expected_5d_return_pct": 2.4, "confidence": 0.7, "catalyst": "optical demand"},
                                        "manager": {"action": "buy", "reason": "pass"}, "prompt_sent": "strong optical orders"}, datetime(2026, 9, 4, 6, 5))
    t = j.open_trade("large", "CIEN", 30, 105.0, 96.0, 123.0, "pick", {"expected_5d_return_pct": 2.4}, str(jf), "o1")
    j.close_trade(t, 106.75, "2026-09-04 12:45", "sold: forecast flipped")
    ctx = A.gather(s, "why did we sell CIEN")
    assert "forecast flipped" in ctx and "strong optical" in ctx and "## Open trades" in ctx
    a = fake_agents(tmp, s, reply="Sold on the forecast flip; +$52.50.")
    assert A.ask(s, a, "why did we sell CIEN").startswith("Sold") and a.costs.spent_today() > 0
    print("ask the journal ok")


def test_propose():
    import propose as P
    import backtest as bt
    s = Settings()
    spec = P.validate({"name": "loser bounce", "terms": [{"feature": "ret_5d", "weight": -1}],
                       "filters": [{"feature": "mom_12_2", "op": ">=", "value": "median"}], "rationale": "r"})
    try:
        P.validate({"terms": [{"feature": "rsi", "weight": 1}]}); raise AssertionError("bad feature accepted")
    except ValueError:
        pass
    from trader.screener import VARIANTS
    VARIANTS["_test_prop"] = P.compile_rule(spec)
    bars = fake_bars(s.symbols[:30], n=560)
    res = bt.run(bars, s, 5, ["_test_prop", "pullback_in_uptrend"], 10.0, 25)
    dates = sorted(res["date"].unique()); split = pd.Timestamp(dates[max(0, len(dates) - bt.TEST_DAYS // bt.HORIZON)])
    v = P.verdict(res, "_test_prop", "pullback_in_uptrend", split)
    assert set(v["timings"]) == {"oo", "cc"} and "passes" in v["timings"]["cc"] and "random_test_edge" in v["timings"]["cc"]
    VARIANTS.pop("_test_prop")
    print("propose (English -> spec -> fair backtest -> verdict) ok")


def test_middle_path_and_killswitch():
    from datetime import date, timedelta
    from trader.killswitch import KillSwitch
    tmp = Path(tempfile.mkdtemp())
    s = Settings()
    rules = dict(s.account_cfg("large"))
    assert rules.get("hold"), "large account must carry a hold block"
    hold = rules["hold"]

    class FB(FakeBroker):
        def __init__(self): super().__init__(None); self.stops = {"AAA": ("o1", 94.0)}; self.replaced = []
        def stop_leg(self, sym): return self.stops.get(sym)
        def replace_stop(self, oid, new): self.replaced.append((oid, new)); self.stops["AAA"] = (oid, new)
    fb = FB()
    # position AAA: entry 100, stop 94 (dist 6), bought 3 days ago; price now 107 => up 7 >= 1.0 x 6 -> trail activates
    (tmp / "positions_large.json").write_text(json.dumps({"AAA": {"date": (date.today() - timedelta(days=3)).isoformat(),
                                                                   "stop": 94.0, "target": 112.0, "entry": 100.0, "risk_usd": 300}}))
    pos = [{"symbol": "AAA", "qty": 50, "avg_entry": 100.0, "current": 107.0, "market_value": 5350, "unrealized_pl": 350, "unrealized_plpc": 0.07}]
    snap = Snapshot(equity=100000, cash=94650, buying_power=4e5, daytrade_count=0, pattern_day_trader=False, positions=pos)
    ex = Executor("large", rules, fb, tmp, s.is_volatile, manage_only=True)
    ex.apply({"actions": [{"symbol": "AAA", "action": "hold", "reason": "x"}]}, snap, {"AAA": 107.0}, {"AAA": 2.0})
    assert fb.replaced and abs(fb.replaced[0][1] - (107 - 1.5 * 2.0)) < 1e-9, fb.replaced      # stop -> high - 1.5 ATR = 104
    st = json.loads((tmp / "positions_large.json").read_text())
    assert st["AAA"]["trail_active"] and st["AAA"]["stop"] == 104.0 and st["AAA"]["high"] == 107.0
    # price falls back: stop must NOT move down; no time stop at day 6 (hard ceiling is 30)
    st["AAA"]["date"] = (date.today() - timedelta(days=6)).isoformat(); (tmp / "positions_large.json").write_text(json.dumps(st))
    pos[0]["current"] = 105.0; fb.replaced.clear(); fb.orders.clear()
    Executor("large", rules, fb, tmp, s.is_volatile, manage_only=True).apply({"actions": []}, snap, {"AAA": 105.0}, {"AAA": 2.0})
    assert not fb.replaced and not fb.orders, "stop moved down or time-stop fired"
    # stale detection: flat within 1% at day 12; hard ceiling at day 30
    st["AAA"]["date"] = (date.today() - timedelta(days=12)).isoformat(); (tmp / "positions_large.json").write_text(json.dumps(st))
    pos[0]["current"] = 100.5
    ex = Executor("large", rules, fb, tmp, s.is_volatile)
    assert ex.stale_positions({"AAA": pos[0]}) == {"AAA"} and ex.days_held("AAA") == 12
    pos[0]["current"] = 103.0
    assert Executor("large", rules, fb, tmp, s.is_volatile).stale_positions({"AAA": pos[0]}) == set()
    st["AAA"]["date"] = (date.today() - timedelta(days=int(hold["hard_max_days"]))).isoformat(); (tmp / "positions_large.json").write_text(json.dumps(st))
    Executor("large", rules, fb, tmp, s.is_volatile, manage_only=True).apply({"actions": []}, snap, {"AAA": 103.0}, {"AAA": 2.0})
    assert ("SELL", "AAA") in fb.orders, "hard ceiling did not fire"
    # small account keeps the classic hard time stop (no hold block)
    assert not s.account_cfg("small").get("hold") and Executor("small", s.account_cfg("small"), fb, tmp, s.is_volatile).hold is None
    # decide(): review day gates the forecast sell; stale name hands its slot over without the rotation edge
    dcfg, rcfg = s.cfg["decision"], s.cfg["research"]
    f = {"OLD": {"expected_5d_return_pct": 0.2, "confidence": 0.6}, "NEG": {"expected_5d_return_pct": -3.0, "confidence": 0.8},
         "NEW": {"expected_5d_return_pct": 0.6, "confidence": 0.6, "bias": "neutral"}}
    d = decide(["NEW"], f, ["OLD", "NEG"], {"max_positions": 2, "hold": hold}, dcfg, rcfg, days_held={"OLD": 12, "NEG": 2}, stale={"OLD"})
    acts = {a["symbol"]: a["action"] for a in d["actions"]}
    assert acts == {"OLD": "sell", "NEG": "hold", "NEW": "buy"}, acts          # NEG negative but only day 2 (< review day 5)
    d = decide(["NEW"], f, ["OLD", "NEG"], {"max_positions": 2, "hold": hold}, dcfg, rcfg, days_held={"OLD": 12, "NEG": 6}, stale=set())
    acts = {a["symbol"]: a["action"] for a in d["actions"]}
    assert acts["NEG"] == "sell" and acts["NEW"] == "buy" and acts["OLD"] == "hold", acts
    # kill switch: daily loss trips and auto-clears next day; drawdown trips and stays
    ks = KillSwitch(tmp, "large", {"enabled": True, "daily_loss_pct": 0.03, "drawdown_pct": 0.08})
    assert ks.check(100000) == (False, "")
    trip, why = ks.check(96500); assert trip and "today" in why, why
    ks.st["day"] = "2000-01-01"; ks._save()
    assert not ks.check(96500)[0], "daily trip must clear on a new day"
    trip, why = ks.check(91000); assert trip and "peak" in why
    assert ks.check(99000)[0], "drawdown trip must persist until reset"
    ks.reset(); assert not ks.check(99000)[0]
    # swapped-in newcomer must not also be vetoed; swaps are reported as pairs
    d = decide(["NEW"], f, ["OLD", "NEG"], {"max_positions": 2, "hold": hold}, dcfg, rcfg, days_held={"OLD": 12, "NEG": 2}, stale={"OLD"})
    assert "NEW" not in d["vetoed"] and d["swaps"] == [("OLD", "NEW")], (d["vetoed"], d["swaps"])
    # risk reviewer drops the buy half of a swap -> the sell half is reverted
    from trader.orchestrator import Orchestrator, CycleState, Candidate
    feats = {"price": 100.0, "atr": 2.0}
    st_ = CycleState("large", "entry", {}, RegimeView("risk-on", 1.0, True), [Candidate("OLD", 1, feats, True), Candidate("NEG", 2, feats, True), Candidate("NEW", 3, feats)])
    st_.forecasts = dict(f)
    orch = Orchestrator(fake_agents(tmp, s), {"news": {"enabled": True}}, news_fetch=lambda x: [], rcfg=rcfg)
    d = orch.decide(st_, {"max_positions": 2, "hold": hold}, dcfg, rcfg, {"NEW": "semis", "OLD": "semis", "NEG": "semis", "X": "semis"}, None,
                    {"max_per_industry": 1, "max_corr_with_book": 0.75}, days_held={"OLD": 12, "NEG": 2}, stale={"OLD"})
    acts = {a["symbol"]: a["action"] for a in d["actions"]}
    assert acts["OLD"] == "hold" and "NEW" not in acts and "risk reviewer" in d["vetoed"]["NEW"], (acts, d["vetoed"])
    # trailing stop is capped below the current price (a retrace between cycles must not fire the stop instantly)
    (tmp / "positions_large.json").write_text(json.dumps({"AAA": {"date": (date.today() - timedelta(days=3)).isoformat(),
                                                                   "stop": 94.0, "entry": 100.0, "high": 120.0, "init_stop_dist": 6.0, "trail_active": True}}))
    fb.replaced.clear(); pos[0]["current"] = 110.0
    Executor("large", rules, fb, tmp, s.is_volatile, manage_only=True).apply({"actions": []}, snap, {"AAA": 110.0}, {"AAA": 2.0})
    assert fb.replaced and fb.replaced[0][1] <= 110 * 0.995 + 1e-9, fb.replaced
    # earnings ahead after the review day -> exit (hold mode only)
    fb.orders.clear()
    st2 = json.loads((tmp / "positions_large.json").read_text()); st2["AAA"]["date"] = (date.today() - timedelta(days=6)).isoformat()
    (tmp / "positions_large.json").write_text(json.dumps(st2))
    Executor("large", rules, fb, tmp, s.is_volatile, manage_only=True, earnings_soon=lambda sym: True).apply({"actions": []}, snap, {"AAA": 110.0}, {"AAA": 2.0})
    assert ("SELL", "AAA") in fb.orders, "earnings-ahead exit did not fire"
    print("middle-path exits (trail / stale / hard ceiling / review day / earnings) + swap safety + kill switch ok")


def test_executor_v3_and_scorecard():
    tmp = Path(tempfile.mkdtemp())
    s = Settings()
    syms = s.symbols[:25]
    bars = fake_bars(syms); fb = FakeBroker(bars)
    table = score_table(bars, syms, s.is_volatile, 1e7, "pullback_in_uptrend")
    cands = [c for c in pick_candidates(table, 6, []) if c in table.index]
    prices = {c: float(table.loc[c, "price"]) for c in cands}
    atrs = {c: float(table.loc[c, "atr"]) for c in cands}
    rules = dict(s.account_cfg("large"))
    snap = Snapshot(equity=100000, cash=100000, buying_power=400000, daytrade_count=0, pattern_day_trader=False, positions=[])
    dec = {"actions": [{"symbol": c, "action": "buy", "reason": "t"} for c in cands]}
    # manage-only: no buys
    ex = Executor("large", rules, fb, tmp, s.is_volatile, manage_only=True); ex.apply(dec, snap, prices, atrs)
    assert not fb.orders and any("manage-only" in l for l in ex.log)
    # risk multiplier halves size; stop cap skips wide names
    ex = Executor("large", rules, fb, tmp, s.is_volatile, risk_mult=1.0); ex.apply(dec, snap, prices, atrs)
    full = {o[1]: o[2] for o in fb.orders}; fb.orders.clear()
    Path(tmp / "positions_large.json").unlink()
    ex = Executor("large", rules, fb, tmp, s.is_volatile, risk_mult=0.5); ex.apply(dec, snap, prices, atrs)
    half = {o[1]: o[2] for o in fb.orders}; fb.orders.clear()
    assert all(abs(half[k] - full[k] // 2) <= 1 for k in half), (full, half)
    Path(tmp / "positions_large.json").unlink()
    wide = cands[0]; atrs2 = dict(atrs); atrs2[wide] = prices[wide] * 0.05   # 2xATR = 10% > 8% cap
    ex = Executor("large", rules, fb, tmp, s.is_volatile, max_stop_pct=0.08); ex.apply(dec, snap, prices, atrs2)
    assert all(o[1] != wide for o in fb.orders) and any("cap 8%" in l for l in ex.log)
    # scorecard migration from a v2 header
    (tmp / "scorecard.csv").write_text("logged,account,symbol,rank,price,expected_5d_pct,confidence,bias,approved,acted,ret_1d,ret_5d,ret_10d\n"
                                       "2026-08-20 12:45,large,ABC,1,100,3.0,0.8,bullish,True,True,0.01,-0.04,-0.05\n")
    sc = Scorecard(tmp)
    sc.log("large", [{"symbol": "X", "rank": 1, "price": 1, "expected_5d_pct": 1, "confidence": 0.5, "bias": "b", "approved": True,
                      "acted": False, "news_exp": 0.5, "filings_tilt": 0.5, "vetoed": "", "regime": "risk-on", "mode": "veto"}])
    df = pd.read_csv(tmp / "scorecard.csv")
    assert list(df.columns) == FIELDS and len(df) == 2 and df.iloc[0]["ret_5d"] == -0.04 and df.iloc[1]["news_exp"] == 0.5
    print("executor v3 guards + scorecard migration ok")


if __name__ == "__main__":
    test_regime(); test_decide(); test_combine_and_review(); test_orchestrator(); test_filings(); test_earnings_windows()
    test_memory_and_postmortem(); test_ask(); test_propose(); test_middle_path_and_killswitch(); test_executor_v3_and_scorecard()
    print("\nALL V3 TESTS PASSED")
