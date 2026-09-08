"""Turns Claude's decisions into orders, enforcing the account's risk rules (v2: volatility-scaled).

Guards (plain Python, Claude cannot override them):
- stop = entry - atr_stop_mult x ATR(14); target = entry + reward_mult x stop distance
- position size = (equity x risk_per_trade_pct) / stop distance, capped by max_position_pct and cash
- portfolio heat: sum of open risk (qty x stop distance) must stay <= max_heat_pct of equity
- max_positions; max_hold_days forced exit; earnings blackout (no buys within N days of a report)
- PDT: an account under $25k never sells something bought today and never makes a 4th day trade
- every buy is a bracket order so the stop and target live at the broker, not in this process
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .broker import Broker, Snapshot


class Executor:
    def __init__(self, name: str, rules: dict, broker: Broker, state_dir: Path, is_volatile,
                 dry_run: bool = False, earnings_soon=None, journal=None, forecasts: dict | None = None,
                 journal_files: dict | None = None):
        self.name = name
        self.journal = journal                      # trader.journal.Journal or None
        self.forecasts = forecasts or {}            # symbol -> parsed Haiku forecast
        self.journal_files = journal_files or {}    # symbol -> journal entry path
        self.orders_placed: dict[str, dict] = {}    # symbol -> {qty, price, stop, target, reason}
        self.rules = rules
        self.broker = broker
        self.is_volatile = is_volatile
        self.dry_run = dry_run
        self.earnings_soon = earnings_soon or (lambda sym: False)
        self.path = state_dir / f"positions_{name}.json"
        raw = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        # migrate v1 format {"SYM": "2026-09-03"} -> {"SYM": {"date":..., "stop":..., "risk_usd":...}}
        self.entries: dict[str, dict] = {k: (v if isinstance(v, dict) else {"date": v}) for k, v in raw.items()}
        self.log: list[str] = []

    def _save(self):
        self.path.write_text(json.dumps(self.entries, indent=1), encoding="utf-8")

    def _say(self, msg: str):
        print(f"  [{self.name}] {msg}")
        self.log.append(msg)

    # ---- helpers ----
    def _entry_date(self, sym: str) -> date | None:
        d = self.entries.get(sym, {}).get("date")
        return date.fromisoformat(d) if d else None

    def _pdt_restricted(self, snap: Snapshot) -> bool:
        return snap.equity < 25000 and not self.rules.get("allow_day_trades", False)

    def _open_risk(self, held: dict) -> float:
        """Sum of (qty x stop distance) for open positions we know the stop of."""
        total = 0.0
        for sym, p in held.items():
            e = self.entries.get(sym, {})
            if e.get("stop"):
                total += p["qty"] * max(p["current"] - e["stop"], 0)
        return total

    def size(self, equity: float, cash: float, price: float, atr: float) -> tuple[int, float, float]:
        """Returns (qty, stop_price, target_price)."""
        stop_dist = self.rules["atr_stop_mult"] * atr
        if stop_dist <= 0 or price <= 0:
            return 0, 0.0, 0.0
        risk_usd = equity * self.rules["risk_per_trade_pct"]
        qty = int(risk_usd // stop_dist)
        qty = min(qty, int(equity * self.rules["max_position_pct"] // price), int(cash * 0.98 // price))
        stop = round(price - stop_dist, 2)
        target = round(price + self.rules["reward_mult"] * stop_dist, 2)
        return max(qty, 0), stop, target

    # ---- main ----
    def apply(self, decision: dict | None, snap: Snapshot, prices: dict[str, float], atrs: dict[str, float]) -> list[str]:
        held = {p["symbol"]: p for p in snap.positions}
        today = date.today()

        for sym in held:
            self.entries.setdefault(sym, {"date": today.isoformat()})
        for sym in list(self.entries):
            if sym not in held:
                del self.entries[sym]

        # 1) forced exits by age (independent of Claude)
        for sym in list(held):
            ed = self._entry_date(sym)
            if ed and (today - ed).days >= self.rules["max_hold_days"]:
                self._sell(sym, "max_hold_days reached")
                held.pop(sym, None)

        if not decision:
            self._say("no decision this cycle (cost cap or model error) — holding")
            self._save()
            return self.log

        actions = decision.get("actions", [])
        sells = [a for a in actions if a.get("action") == "sell" and a.get("symbol") in held]
        buys = [a for a in actions if a.get("action") == "buy" and a.get("symbol") not in held]

        # 2) sells
        for a in sells:
            sym = a["symbol"]
            if self._pdt_restricted(snap) and self._entry_date(sym) == today:
                self._say(f"skip SELL {sym}: bought today and account is PDT-restricted")
                continue
            self._sell(sym, a.get("reason", ""))
            held.pop(sym, None)

        # 3) buys
        slots = self.rules["max_positions"] - len(held)
        if self._pdt_restricted(snap) and snap.daytrade_count >= 3:
            self._say("skip all buys: already at 3 day trades in 5 days (PDT protection)")
            slots = 0
        cash = snap.cash
        heat = self._open_risk(held)
        heat_cap = snap.equity * self.rules["max_heat_pct"]
        for a in buys:
            if slots <= 0:
                break
            sym = a["symbol"]
            price, atr = prices.get(sym, 0), atrs.get(sym, 0)
            if price <= 0 or atr <= 0:
                self._say(f"skip BUY {sym}: no price/ATR")
                continue
            if self.earnings_soon(sym):
                self._say(f"skip BUY {sym}: earnings within blackout window")
                continue
            qty, stop, target = self.size(snap.equity, cash, price, atr)
            if qty < 1:
                self._say(f"skip BUY {sym}: risk budget ${snap.equity*self.rules['risk_per_trade_pct']:.0f} "
                          f"buys <1 share at ${price:.2f} with stop {self.rules['atr_stop_mult']}xATR=${self.rules['atr_stop_mult']*atr:.2f}")
                continue
            risk_usd = qty * (price - stop)
            if heat + risk_usd > heat_cap:
                self._say(f"skip BUY {sym}: portfolio heat ${heat:.0f}+${risk_usd:.0f} would exceed cap ${heat_cap:.0f}")
                continue
            self._buy(sym, qty, price, stop, target, a.get("reason", ""))
            cash -= qty * price
            heat += risk_usd
            slots -= 1
            self.entries[sym] = {"date": today.isoformat(), "stop": stop, "target": target,
                                 "entry": price, "risk_usd": round(risk_usd, 2)}

        for a in actions:
            if a.get("action") == "hold" and a.get("symbol") in held:
                self._say(f"HOLD {a['symbol']}: {a.get('reason','')}")

        self._save()
        return self.log

    def _buy(self, sym, qty, price, stop, target, reason):
        self._say(f"BUY {qty} {sym} @~{price:.2f} (${qty*price:,.0f}) stop {stop:.2f} ({(stop/price-1)*100:+.1f}%) "
                  f"target {target:.2f} ({(target/price-1)*100:+.1f}%) risk ${qty*(price-stop):,.0f} — {reason}")
        self.orders_placed[sym] = {"qty": qty, "price": price, "stop": stop, "target": target, "reason": reason}
        if self.dry_run:
            self.log[-1] = "(dry run, not sent) " + self.log[-1]
            return
        try:
            o = self.broker.buy_bracket(sym, qty, stop, target)
            oid, status = (str(o.id), str(o.status).split(".")[-1].lower()) if o is not None else (None, "not_submitted")
        except Exception as e:
            oid, status = None, f"REJECTED: {e}"
            self._say(f"  !! order for {sym} rejected by broker: {e}")
        self.orders_placed[sym].update({"order_id": oid, "status": status})
        self.log[-1] += f"  [order {oid or '-'} {status}]"
        if self.journal and oid:
            self.journal.open_trade(self.name, sym, qty, price, stop, target, reason,
                                    self.forecasts.get(sym), self.journal_files.get(sym), order_id=oid)

    def _sell(self, sym, reason):
        self._say(f"SELL {sym} — {reason}")
        if self.dry_run:
            self.log[-1] = "(dry run, not sent) " + self.log[-1]
        else:
            if self.journal:
                self.journal.mark_exit_reason(self.name, sym, "time_stop" if "max_hold_days" in reason else f"sold: {reason}")
            try:
                o = self.broker.close_position(sym)
                self.log[-1] += f"  [order {getattr(o, 'id', '-')} {str(getattr(o, 'status', '')).split('.')[-1].lower()}]"
            except Exception as e:
                self._say(f"  !! close order for {sym} rejected by broker: {e}")
        self.entries.pop(sym, None)
