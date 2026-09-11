"""v3 decision rule (deterministic). Replaces the Sonnet portfolio-manager call.

The screener picks, the agents can only VETO, and the rules below turn that into actions:
  buy   screener candidates in rank order that are not vetoed, until max_positions is reached
  veto  combined forecast < veto_below_pct, or bias bearish with confidence >= veto_bearish_conf,
        or the memory agent says the thesis is broken
  sell  held names whose FRESH forecast < sell_below_pct with confidence >= min_confidence,
        or whose thesis the memory agent marks broken
  swap  when full: at most max_swaps_per_cycle, and only if the best vetoed-free newcomer beats the weakest
        held name's fresh forecast by rotation_min_edge_pct (prevents churn on noise)
Nothing here sees P&L. Output has the same shape as the old manager reply so the Executor is unchanged.
"""
from __future__ import annotations


def _num(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def veto_reason(f: dict | None, dcfg: dict) -> str | None:
    """Why the agents veto this name, or None if it passes."""
    if not f:
        return "no forecast (agent failed or cost cap)"
    exp, conf = _num(f.get("expected_5d_return_pct")), _num(f.get("confidence"))
    if f.get("thesis_status") == "broken":
        return f"memory: thesis broken ({f.get('memory_note', '')[:60]})"
    if exp < dcfg.get("veto_below_pct", -0.5):
        return f"forecast {exp:+.1f}% < {dcfg.get('veto_below_pct', -0.5)}%"
    if f.get("bias") == "bearish" and conf >= dcfg.get("veto_bearish_conf", 0.6):
        return f"bearish with confidence {conf:.2f}"
    return None


def decide(candidates: list[str], forecasts: dict[str, dict], held: list[str], rules: dict, dcfg: dict,
           rcfg: dict, regime=None, days_held: dict[str, int] | None = None, stale: set[str] | None = None) -> dict:
    """days_held / stale: v3 middle-path inputs from the executor. With a `hold` block on the account, the thesis check
    (sell on a clearly negative fresh forecast) applies from hold.review_after_days; a broken thesis sells at any age.
    Stale positions give up their slot to any non-vetoed newcomer without needing the rotation edge."""
    actions, vetoed = [], {}
    held_set = set(held)
    min_conf = rcfg.get("min_confidence", 0.6)
    hold = rules.get("hold") or {}
    review_day = int(hold.get("review_after_days", 0)) if hold else 0
    days_held, stale = days_held or {}, stale or set()

    # 1) sells among held names (fresh forecast only; never P&L)
    sells = []
    for sym in held:
        f = forecasts.get(sym)
        if not f:
            actions.append({"symbol": sym, "action": "hold", "reason": "no fresh forecast; keep (stop/target live at broker)"})
            continue
        exp, conf = _num(f.get("expected_5d_return_pct")), _num(f.get("confidence"))
        age = days_held.get(sym, 0)
        if f.get("thesis_status") == "broken":
            sells.append(sym); actions.append({"symbol": sym, "action": "sell", "reason": f"memory: thesis broken — {f.get('memory_note', '')[:50]}"})
        elif exp < dcfg.get("sell_below_pct", -1.5) and conf >= min_conf and age >= review_day:
            sells.append(sym); actions.append({"symbol": sym, "action": "sell", "reason": f"fresh forecast {exp:+.1f}% (conf {conf:.2f}), day {age}"})
        else:
            tag = " [stale]" if sym in stale else (f" [review from day {review_day}]" if hold and age < review_day and exp < dcfg.get("sell_below_pct", -1.5) else "")
            actions.append({"symbol": sym, "action": "hold", "reason": f"forecast {exp:+.1f}% conf {conf:.2f}{tag}"})
    remaining = [h for h in held if h not in sells]

    # 2) buys: screener order, agents veto
    passers = []
    for sym in candidates:
        if sym in held_set:
            continue
        why = veto_reason(forecasts.get(sym), dcfg)
        if why:
            vetoed[sym] = why
        else:
            passers.append(sym)
    if regime is not None and not getattr(regime, "allow_new_buys", True):
        for sym in passers:
            vetoed[sym] = f"regime: {getattr(regime, 'label', '?')} — no new buys"
        passers = []
    if dcfg.get("rank_by", "screener") == "forecast":   # opt-in: let the agents ORDER the passers (flip after the eval says forecasts carry information)
        passers.sort(key=lambda s_: -_num(forecasts[s_].get("expected_5d_return_pct")))
    slots = max(rules["max_positions"] - len(remaining), 0)
    for sym in passers[:slots]:
        f = forecasts[sym]
        actions.append({"symbol": sym, "action": "buy",
                        "reason": f"screener pick, agents pass ({_num(f.get('expected_5d_return_pct')):+.1f}% conf {_num(f.get('confidence')):.2f})"})
    leftover = passers[slots:]

    pairs: list[tuple[str, str]] = []   # (old, new) swaps, so the risk reviewer can undo the sell if it drops the buy
    # 3a) v3 stale rotation: flat, old positions hand their slot to any newcomer that passed (no edge needed)
    stale_left = [h for h in remaining if h in stale]
    while leftover and stale_left:
        old, new = stale_left.pop(0), leftover.pop(0)
        for a in actions:
            if a["symbol"] == old:
                a.update({"action": "sell", "reason": f"stale: flat for {days_held.get(old, '?')} days, slot needed for {new}"})
        actions.append({"symbol": new, "action": "buy", "reason": f"takes stale slot of {old} ({_num(forecasts[new].get('expected_5d_return_pct')):+.1f}%)"})
        remaining = [h for h in remaining if h != old]; sells.append(old); pairs.append((old, new))

    # 3b) rotation when full: swap the weakest held name for a clearly better newcomer (max N per cycle)
    swaps = 0
    edge = dcfg.get("rotation_min_edge_pct", 1.0)
    if leftover and remaining:
        scored = sorted(((_num(forecasts.get(h, {}).get("expected_5d_return_pct")), h) for h in remaining if h in forecasts))
        swapped = set()
        for new in leftover:
            if swaps >= dcfg.get("max_swaps_per_cycle", 1) or not scored:
                break
            w_exp, w_sym = scored[0]
            n_exp = _num(forecasts[new].get("expected_5d_return_pct"))
            if n_exp - w_exp >= edge:
                for a in actions:
                    if a["symbol"] == w_sym:
                        a.update({"action": "sell", "reason": f"rotation: {new} forecast {n_exp:+.1f}% beats {w_exp:+.1f}% by >= {edge}%"})
                actions.append({"symbol": new, "action": "buy", "reason": f"rotation in for {w_sym} ({n_exp:+.1f}%)"})
                scored.pop(0); swaps += 1; swapped.add(new); pairs.append((w_sym, new))
            else:
                vetoed[new] = f"no slot; edge over weakest held ({w_sym} {w_exp:+.1f}%) < {edge}%"
        for new in leftover:
            if new not in swapped:
                vetoed.setdefault(new, "no slot (positions full)")
    else:
        for new in leftover:
            vetoed[new] = "no slot (positions full)"

    n_buy = sum(a["action"] == "buy" for a in actions)
    note = f"rule: {n_buy} buy, {len(sells) + swaps} sell, {len(vetoed)} vetoed/no-slot"
    if regime is not None:
        note += f"; regime {getattr(regime, 'label', '?')}"
    return {"actions": actions, "market_note": note, "vetoed": vetoed, "mode": "veto", "swaps": pairs}
