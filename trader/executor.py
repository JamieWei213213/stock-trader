"""Turns Claude's decisions into orders, enforcing the account's risk rules (v2: volatility-scaled).

Guards (plain Python, Claude cannot override them):
- stop = entry - atr_stop_mult x ATR(14); target = entry + reward_mult x stop distance
- position size = (equity x risk_per_trade_pct) / stop distance, capped by max_position_pct and cash
- portfolio heat: sum of open risk (qty x stop distance) must stay <= max_heat_pct of equity
- max_positions; max_hold_days forced exit; earnings blackout (no buys within N days of a report)
- v3 "middle path" (accounts with a `hold:` block): no hard time stop. Instead a trailing stop that ratchets the
  broker's stop up once the trade is ahead by one stop distance, a thesis check via the fresh forecast (decide.py),
  a 'stale' flag from day N for flat positions so a better candidate can take the slot, and a hard ceiling in days.
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
                 journal_files: dict | None = None, risk_mult: float = 1.0, max_stop_pct: float | None = None,
                 manage_only: bool = False):
        self.name = name
        self.risk_mult = risk_mult                  # v3: regime monitor scales risk per trade (1.0 = normal)
        self.max_stop_pct = max_stop_pct            # v3: skip names whose ATR stop would be wider than this fraction of price
        self.manage_only = manage_only              # v3: 6:05am cycle — sells/time stops only, never buys
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
        self.hold: dict | None = rules.get("hold")   # v3 middle-path exits; None = classic hard max_hold_days
        self.stale: set[str] = set()

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

    def days_held(self, sym: str) -> int:
        ed = self._entry_date(sym)
        return (date.today() - ed).days if ed else 0

    def stale_positions(self, held: dict) -> set[str]:
        """v3: held names that are old enough and flat enough to give up their slot to a better candidate."""
        if not self.hold:
            return set()
        band = float(self.hold.get("stale_band_pct", 1.0)) / 100
        out = set()
        for sym, p in held.items():
            e = self.entries.get(sym, {})
            entry = e.get("entry") or p.get("avg_entry") or 0
            if entry and self.days_held(sym) >= int(self.hold.get("stale_days", 10)) and abs(p["current"] / entry - 1) <= band:
                out.add(sym)
        self.stale = out
        return out

    def _trail(self, held: dict, prices: dict, atrs: dict):
        """v3 trailing stop: once price is ahead by trail_activate_mult x the initial stop distance, move the broker's
        stop to (high since entry - trail_atr_mult x ATR); never lower it."""
        for sym, p in held.items():
            e = self.entries.get(sym)
            if not e or not e.get("stop") or not e.get("entry"):
                continue
            px = float(prices.get(sym) or p["current"])
            high = max(float(e.get("high", e["entry"])), px)
            if not self.dry_run:
                e["high"] = high
            stop_dist = e["entry"] - e["stop"] if not e.get("trail_active") else e.get("init_stop_dist", 0)
            e.setdefault("init_stop_dist", stop_dist)
            atr = atrs.get(sym) or 0
            if atr <= 0 or e["init_stop_dist"] <= 0:
                continue
            if not e.get("trail_active") and px - e["entry"] < float(self.hold["trail_activate_mult"]) * e["init_stop_dist"]:
                continue
            new_stop = round(min(high - float(self.hold["trail_atr_mult"]) * atr, px * 0.995), 2)   # never at/above the price
            if new_stop <= e["stop"]:
                continue
            msg = f"TRAIL {sym}: stop {e['stop']:.2f} -> {new_stop:.2f} (high {high:.2f}, {(new_stop / e['entry'] - 1) * 100:+.1f}% vs entry)"
            if self.dry_run:
                self._say("(dry run, not sent) " + msg); continue
            try:
                leg = self.broker.stop_leg(sym)
                if leg:
                    self.broker.replace_stop(leg[0], new_stop)
                    e["stop"], e["trail_active"] = new_stop, True
                    self._say(msg)
                    if self.journal:
                        self.journal.mark_trailing(self.name, sym, new_stop)
                else:
                    self._say(f"  !! no resting stop order found for {sym}; cannot trail")
            except Exception as ex:
                self._say(f"  !! trail update for {sym} rejected: {ex}")

    def size(self, equity: float, cash: float, price: float, atr: float) -> tuple[int, float, float]:
        """Returns (qty, stop_price, target_price)."""
        stop_dist = self.rules["atr_stop_mult"] * atr
        if stop_dist <= 0 or price <= 0:
            return 0, 0.0, 0.0
        risk_usd = equity * self.rules["risk_per_trade_pct"] * self.risk_mult
        qty = int(risk_usd // stop_dist)
        intended = min(qty, int(equity * self.rules["max_position_pct"] // price))
        qty = min(intended, int(cash * 0.98 // price))
        if intended >= 1 and qty < intended * 0.5:   # v3: don't buy crumbs when cash runs out; wait for a slot
            return 0, stop_dist, -intended
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

        # 1) forced exits by age (independent of Claude): classic hard time stop, or (v3 hold block) the hard ceiling
        limit = int(self.hold["hard_max_days"]) if self.hold else int(self.rules["max_hold_days"])
        for sym in list(held):
            ed = self._entry_date(sym)
            if ed and (today - ed).days >= limit:
                self._sell(sym, "hard_max_days reached" if self.hold else "max_hold_days reached")
                held.pop(sym, None)
        # 1a) v3 hold mode: don't carry a position through an earnings report once it is past the review day
        if self.hold:
            for sym in list(held):
                if self.earnings_soon(sym) and self.days_held(sym) >= int(self.hold.get("review_after_days", 5)):
                    self._sell(sym, "earnings within blackout window")
                    held.pop(sym, None)
        # 1b) v3 trailing stop ratchet (every cycle, including manage-only); skip names the rule is selling anyway
        if self.hold:
            selling = {a.get("symbol") for a in (decision or {}).get("actions", []) if a.get("action") == "sell"}
            self._trail({k: v for k, v in held.items() if k not in selling}, prices, atrs)

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
        if self.manage_only and buys:
            self._say(f"manage-only cycle: {len(buys)} buy(s) deferred to the entry cycle")
            slots = 0
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
            if self.max_stop_pct and self.rules["atr_stop_mult"] * atr / price > self.max_stop_pct:
                self._say(f"skip BUY {sym}: stop {self.rules['atr_stop_mult']}xATR = {self.rules['atr_stop_mult']*atr/price*100:.1f}% of price > cap {self.max_stop_pct*100:.0f}%")
                continue
            qty, stop, target = self.size(snap.equity, cash, price, atr)
            if qty == 0 and target < 0:
                self._say(f"skip BUY {sym}: cash ${cash:,.0f} buys < half the intended {-int(target)} shares — waiting for a slot")
                continue
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
                self.journal.mark_exit_reason(self.name, sym, "hard_max" if "hard_max_days" in reason else "time_stop" if "max_hold_days" in reason
                                              else "stale" if reason.startswith("stale") else "earnings_exit" if reason.startswith("earnings")
                                              else f"sold: {reason}")
            try:
                o = self.broker.close_position(sym)
                self.log[-1] += f"  [order {getattr(o, 'id', '-')} {str(getattr(o, 'status', '')).split('.')[-1].lower()}]"
            except Exception as e:
                self._say(f"  !! close order for {sym} rejected by broker: {e}")
        self.entries.pop(sym, None)
