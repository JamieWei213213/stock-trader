"""Thin wrapper around alpaca-py for one paper account."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from .settings import AccountCreds


@dataclass
class Snapshot:
    equity: float
    cash: float
    buying_power: float
    daytrade_count: int
    pattern_day_trader: bool
    positions: list[dict]


class Broker:
    def __init__(self, creds: AccountCreds):
        self.name = creds.name
        self.trading = TradingClient(creds.key, creds.secret, paper=True)
        self.data = StockHistoricalDataClient(creds.key, creds.secret)
        self.news = NewsClient(creds.key, creds.secret)

    # ---------- account ----------
    def snapshot(self) -> Snapshot:
        acct = self.trading.get_account()
        positions = []
        for p in self.trading.get_all_positions():
            positions.append(
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry": float(p.avg_entry_price),
                    "current": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
            )
        return Snapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            daytrade_count=int(acct.daytrade_count or 0),
            pattern_day_trader=bool(acct.pattern_day_trader),
            positions=positions,
        )

    def market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    # ---------- market data ----------
    def daily_bars(self, symbols: list[str], days: int = 60) -> pd.DataFrame:
        """Returns a multi-index (symbol, timestamp) DataFrame of daily OHLCV."""
        start = datetime.now(timezone.utc) - timedelta(days=int(days * 1.5) + 5)
        parts = []
        for i in range(0, len(symbols), 100):  # keep request URLs short; SDK paginates within a chunk
            req = StockBarsRequest(symbol_or_symbols=symbols[i:i + 100], timeframe=TimeFrame.Day, start=start, feed="iex")
            df = self.data.get_stock_bars(req).df
            if not df.empty:
                parts.append(df)
        return pd.concat(parts) if parts else pd.DataFrame()

    def daily_bars_since(self, symbols: list[str], start: datetime) -> pd.DataFrame:
        """Full history from `start` (used by the data lake). Alpaca pages automatically."""
        req = StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, feed="iex"
        )
        return self.data.get_stock_bars(req).df

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed="iex")
        quotes = self.data.get_stock_latest_quote(req)
        out = {}
        for sym, q in quotes.items():
            mid = (float(q.ask_price) + float(q.bid_price)) / 2 if q.ask_price and q.bid_price else 0.0
            out[sym] = mid or float(q.ask_price or q.bid_price or 0)
        return out

    def recent_news(self, symbol: str, hours: int = 36, limit: int = 8) -> list[dict]:
        start = datetime.now(timezone.utc) - timedelta(hours=hours)
        req = NewsRequest(symbols=symbol, start=start, limit=limit, include_content=True)
        res = self.news.get_news(req)
        items = []
        for n in res.data.get("news", []):
            content = re.sub(r"<[^>]+>", " ", n.content or "")          # strip HTML tags
            content = re.sub(r"\s+", " ", content).strip()
            items.append(
                {
                    "time": str(n.created_at)[:16],
                    "headline": n.headline,
                    "summary": (n.summary or "")[:300],
                    "content": content[:1500],
                    "source": n.source,
                }
            )
        return items

    # ---------- orders ----------
    def buy_bracket(self, symbol: str, qty: float, stop_price: float, target_price: float):
        """Market buy with attached stop-loss and take-profit (bracket order) at absolute prices."""
        qty = int(qty)  # bracket orders require whole shares
        if qty < 1:
            return None
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(target_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
        )
        return self.trading.submit_order(req)

    def close_position(self, symbol: str):
        # cancel any open bracket legs first, then liquidate
        for o in self.trading.get_orders():
            if o.symbol == symbol:
                try:
                    self.trading.cancel_order_by_id(o.id)
                except Exception:
                    pass
        return self.trading.close_position(symbol)

    def stop_leg(self, symbol: str):
        """The open SELL stop order protecting a position (bracket child leg) -> (order_id, stop_price) or None."""
        # After the bracket parent fills, the stop leg sits in status 'held' (OCO sibling of the limit leg) and is NOT
        # returned by status=open; and nested legs hang off the (filled) parent. So query ALL and pick by the leg's state.
        from alpaca.trading.enums import QueryOrderStatus
        live = {"new", "held", "accepted", "pending_new", "partially_filled", "pending_replace"}
        for o in self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[symbol], nested=True, limit=50)):
            for leg in [o] + list(o.legs or []):
                if (leg.symbol == symbol and str(leg.side).endswith("SELL") and "stop" in str(leg.type).lower()
                        and leg.stop_price and str(leg.status).split(".")[-1].lower() in live):
                    return str(leg.id), float(leg.stop_price)
        return None

    def replace_stop(self, order_id: str, new_stop: float):
        """Ratchet a resting stop order to a new price (v3 trailing stop)."""
        return self.trading.replace_order_by_id(order_id, ReplaceOrderRequest(stop_price=round(new_stop, 2)))

    def open_orders(self) -> list[dict]:
        return [
            {"symbol": o.symbol, "side": str(o.side), "qty": o.qty, "type": str(o.type), "status": str(o.status)}
            for o in self.trading.get_orders()
        ]
