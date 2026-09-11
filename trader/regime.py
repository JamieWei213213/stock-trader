"""Regime monitor (v3, deterministic node): scales risk by market state. It never picks stocks.

Rules (config.yaml -> regime):
  - SPY below its 200-day average           -> risk_per_trade multiplied by risk_mult_below_ma (default 0.5)
  - ...AND 20-day realized vol > vol_spike_mult x its 1-year median -> no new buys at all
Evidence: trend filters on the index do not raise returns much but reliably cut drawdowns (Faber 2007,
Moskowitz/Ooi/Pedersen 2012 time-series momentum). Cheap insurance, so it is on by default.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class RegimeView:
    label: str                 # "risk-on" | "risk-off" | "crisis" | "unknown"
    risk_mult: float           # multiply risk_per_trade_pct by this
    allow_new_buys: bool
    spy_close: float = 0.0
    spy_ma: float = 0.0
    vol_20d: float = 0.0       # annualized
    vol_median_1y: float = 0.0
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def assess(close: pd.Series, cfg: dict) -> RegimeView:
    """close: daily closes of the index proxy, oldest first (needs >= ma_days + vol window)."""
    close = close.dropna().astype(float)
    ma_n = int(cfg.get("ma_days", 200))
    vw = int(cfg.get("vol_window_days", 20))
    if len(close) < ma_n + vw:
        return RegimeView("unknown", 1.0, True, note=f"only {len(close)} bars, need {ma_n + vw}; regime filter off")
    ma = float(close.rolling(ma_n).mean().iloc[-1])
    last = float(close.iloc[-1])
    rets = close.pct_change().dropna()
    vol = rets.rolling(vw).std() * np.sqrt(252)
    v_now = float(vol.iloc[-1])
    v_med = float(vol.iloc[-252:].median())
    below = last < ma
    spike = v_med > 0 and v_now > cfg.get("vol_spike_mult", 2.0) * v_med
    if below and spike:
        return RegimeView("crisis", float(cfg.get("risk_mult_below_ma", 0.5)), False, last, ma, v_now, v_med,
                          f"{cfg.get('symbol','SPY')} {last:.0f} < MA{ma_n} {ma:.0f} and vol {v_now:.0%} > {cfg.get('vol_spike_mult',2.0)}x median {v_med:.0%}: no new buys")
    if below:
        return RegimeView("risk-off", float(cfg.get("risk_mult_below_ma", 0.5)), True, last, ma, v_now, v_med,
                          f"{cfg.get('symbol','SPY')} {last:.0f} < MA{ma_n} {ma:.0f}: risk x{cfg.get('risk_mult_below_ma', 0.5)}")
    return RegimeView("risk-on", 1.0, True, last, ma, v_now, v_med,
                      f"{cfg.get('symbol','SPY')} {last:.0f} > MA{ma_n} {ma:.0f}, vol {v_now:.0%}")


def assess_live(broker, cfg: dict) -> RegimeView:
    if not cfg.get("enabled", True):
        return RegimeView("off", 1.0, True, note="regime filter disabled in config")
    sym = cfg.get("symbol", "SPY")
    try:
        bars = broker.daily_bars([sym], int(cfg.get("ma_days", 200)) + 300)
        close = bars["close"].xs(sym, level="symbol").sort_index()
        return assess(close, cfg)
    except Exception as e:
        return RegimeView("unknown", 1.0, True, note=f"could not load {sym} bars ({e}); regime filter off")
