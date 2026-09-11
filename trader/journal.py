"""Decision journal: the full evidence trail behind every candidate and every trade.

Two stores under state/:
  journal/YYYY-MM-DD/HHMM_<account>_<SYMBOL>.json   one file per candidate per cycle:
      screener features, the exact anonymized text sent to Haiku, Haiku's raw reply, the parsed
      forecast, the manager's action + reason for this symbol, and the order (if any).
  trades.json                                       one record per trade, entry -> exit:
      entry date/price/qty/stop/target, the manager's reason, the forecast, a pointer to the
      journal file, and after the position closes: exit date/price/reason and realized P&L.

`python journal.py SYMBOL`, `python journal.py --losers`, `python journal.py --recent 20`
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path


class Journal:
    def __init__(self, state_dir: Path):
        self.dir = state_dir / "journal"
        self.dir.mkdir(exist_ok=True)
        self.trades_path = state_dir / "trades.json"
        self.trades: list[dict] = json.loads(self.trades_path.read_text(encoding="utf-8")) if self.trades_path.exists() else []

    # ---------------------------------------------------------------- per-candidate entries
    def write_entry(self, account: str, symbol: str, entry: dict, when: datetime | None = None) -> Path:
        when = when or datetime.now()
        day = self.dir / when.strftime("%Y-%m-%d")
        day.mkdir(exist_ok=True)
        path = day / f"{when:%H%M}_{account}_{symbol}.json"
        entry = {"logged": when.strftime("%Y-%m-%d %H:%M"), "account": account, "symbol": symbol, **entry}
        path.write_text(json.dumps(entry, indent=1, default=str), encoding="utf-8")
        return path

    def entries_for(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        files = sorted(self.dir.glob("*/*.json"), reverse=True)
        out = []
        for f in files:
            if symbol and not f.stem.endswith(f"_{symbol}"):
                continue
            try:
                out.append({"_file": str(f), **json.loads(f.read_text(encoding="utf-8"))})
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    # ---------------------------------------------------------------- trades
    def _save(self):
        self.trades_path.write_text(json.dumps(self.trades, indent=1, default=str), encoding="utf-8")

    def open_trade(self, account: str, symbol: str, qty: int, price: float, stop: float, target: float,
                   reason: str, forecast: dict | None, journal_file: str | None, order_id: str | None = None) -> dict:
        t = {
            "id": uuid.uuid4().hex[:8], "account": account, "symbol": symbol, "status": "open",
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M"), "qty": qty, "entry_price": round(price, 2),
            "stop": stop, "target": target, "entry_reason": reason,
            "forecast_5d_pct": (forecast or {}).get("expected_5d_return_pct"),
            "confidence": (forecast or {}).get("confidence"), "catalyst": (forecast or {}).get("catalyst"),
            "journal_file": journal_file, "order_id": order_id, "exit_time": None, "exit_price": None, "exit_reason": None,
            "pnl": None, "pnl_pct": None,
        }
        self.trades.append(t)
        self._save()
        return t

    def mark_exit_reason(self, account: str, symbol: str, reason: str):
        """Called by the executor when IT decides to sell (time stop or manager sell)."""
        for t in self.trades:
            if t["account"] == account and t["symbol"] == symbol and t["status"] == "open":
                t["pending_exit_reason"] = reason
        self._save()

    def mark_trailing(self, account: str, symbol: str, stop: float):
        """v3: the executor ratcheted the stop; a later stop fill is a 'trail_stop', not the initial stop."""
        for t in self.trades:
            if t["account"] == account and t["symbol"] == symbol and t["status"] == "open":
                t["trailing"] = True
                t["trail_stop"] = stop           # t["stop"] stays the entry stop (post-mortem / journal CLI use it)
        self._save()

    def open_trades(self, account: str | None = None) -> list[dict]:
        return [t for t in self.trades if t["status"] == "open" and (account is None or t["account"] == account)]

    def close_trade(self, t: dict, exit_price: float, exit_time: str, reason: str):
        t.update({
            "status": "closed", "exit_price": round(exit_price, 2), "exit_time": exit_time, "exit_reason": reason,
            "pnl": round((exit_price - t["entry_price"]) * t["qty"], 2),
            "pnl_pct": round(exit_price / t["entry_price"] - 1, 4),
        })
        t.pop("pending_exit_reason", None)
        self._save()

    def reconcile(self, account: str, broker, held_symbols: set[str]) -> int:
        """Close journal trades whose position is gone, using the broker's filled sell orders.
        Exit reason: executor-marked (time stop / manager sell) else inferred from the fill's order type
        (stop -> 'stop_loss', limit -> 'take_profit')."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        closed = 0
        for t in self.open_trades(account):
            if t["symbol"] in held_symbols:
                continue
            try:
                orders = broker.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[t["symbol"]], limit=20))
            except Exception as e:
                print(f"  [journal] could not fetch orders for {t['symbol']}: {e}")
                continue
            sells = [o for o in orders if str(o.side).endswith("SELL") and o.filled_at and o.filled_avg_price]
            sells.sort(key=lambda o: o.filled_at, reverse=True)
            if not sells:
                continue
            o = sells[0]
            otype = str(o.type).split(".")[-1].lower()
            reason = t.get("pending_exit_reason") or {"stop": "trail_stop" if t.get("trailing") else "stop_loss",
                                                       "limit": "take_profit", "market": "sold"}.get(otype, otype)
            self.close_trade(t, float(o.filled_avg_price), o.filled_at.astimezone().strftime("%Y-%m-%d %H:%M"), reason)
            closed += 1
        return closed

    # ---------------------------------------------------------------- reporting
    def closed_trades(self, account: str | None = None) -> list[dict]:
        return [t for t in self.trades if t["status"] == "closed" and (account is None or t["account"] == account)]

    def summary(self) -> str:
        c = self.closed_trades()
        if not c:
            return "journal: no closed trades yet"
        wins = [t for t in c if t["pnl"] > 0]
        by_reason: dict[str, list] = {}
        for t in c:
            by_reason.setdefault(t["exit_reason"], []).append(t["pnl_pct"])
        lines = [f"=== Trades: {len(c)} closed, {len(wins)} winners ({len(wins)/len(c)*100:.0f}%), "
                 f"total P&L ${sum(t['pnl'] for t in c):+,.2f}, avg {sum(t['pnl_pct'] for t in c)/len(c)*100:+.2f}% ==="]
        for r, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"  exit by {r:<12} n={len(v):3d}  avg {sum(v)/len(v)*100:+.2f}%")
        return "\n".join(lines)
