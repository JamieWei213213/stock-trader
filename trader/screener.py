"""Free quantitative pre-screen: ranks the universe so Claude only researches the top N.

v2: features are computed as panels (date x symbol) so the exact same code serves the live
screen (last row) and the backtest (any row), and 500 symbols x 4 years runs in seconds.

Signals with real evidence for a 1-10 day hold (see RESEARCH.md):
  reversal_5d          buy last week's losers (short-term reversal)
  mom_12_2             classic momentum: months 12..2, skipping the last month
  pullback_in_uptrend  last week's losers among top-half 12-2 momentum names (the combination)
Legacy variants are kept so the sweep shows them side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .broker import Broker
from .settings import Settings

LOOKBACK_DAYS = 300  # trading days of history needed for 12-2 momentum (252) + warmup


# ----------------------------------------------------------------------------- features
def features_panel(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """bars: MultiIndex (symbol, timestamp) OHLCV -> dict of wide DataFrames (timestamp x symbol)."""
    close = bars["close"].unstack("symbol").sort_index()
    high = bars["high"].unstack("symbol").reindex_like(close)
    low = bars["low"].unstack("symbol").reindex_like(close)
    vol = bars["volume"].unstack("symbol").reindex_like(close)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    atr14 = tr.rolling(14).mean()
    return {
        "price": close,
        "open": bars["open"].unstack("symbol").reindex_like(close),
        "ret_1d": close / close.shift(1) - 1,
        "ret_5d": close / close.shift(5) - 1,
        "ret_20d": close / close.shift(20) - 1,
        "mom_12_2": close.shift(21) / close.shift(252) - 1,
        "vol_surge": vol / vol.rolling(20).mean().shift(1),
        "atr_pct": atr14 / close,
        "atr": atr14,
        "avg_dollar_vol": (close * vol).rolling(20).mean(),
        "dist_20d_high": close / close.rolling(20).max() - 1,
    }


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0


# Ranking rules operate on a per-symbol DataFrame (one row per symbol, columns = features).
VARIANTS = {
    # --- legacy (v1) ---
    "attention":   lambda d: 1.0 * _z(d["ret_5d"].abs()) + 0.5 * _z(d["ret_20d"]) + 1.0 * _z(d["vol_surge"]) + 0.5 * _z(d["atr_pct"]),
    "momentum":    lambda d: _z(d["ret_20d"]) + 0.5 * _z(d["dist_20d_high"]),   # NOTE: 20d window = reversal territory
    "pullback":    lambda d: -_z(d["ret_5d"]) + 0.5 * _z(d["ret_20d"]),
    "reversal_1d": lambda d: -_z(d["ret_1d"]),
    "low_vol":     lambda d: -_z(d["atr_pct"]),
    "volume":      lambda d: _z(d["vol_surge"]),
    # --- v2, evidence-based ---
    "reversal_5d": lambda d: -_z(d["ret_5d"]),
    "mom_12_2":    lambda d: _z(d["mom_12_2"]),
    "pullback_in_uptrend": lambda d: (-_z(d["ret_5d"])).where(d["mom_12_2"] >= d["mom_12_2"].median(), -np.inf),
}


def score_at(feats: dict[str, pd.DataFrame], when, is_volatile, min_dollar_vol: float,
             variant: str = "pullback_in_uptrend", universe_n: int | None = None,
             keep: list[str] | None = None) -> pd.DataFrame:
    """Per-symbol table as of one date. universe_n: keep only the top-N by trailing dollar volume
    AS OF THAT DATE (point-in-time universe, used by the backtest to limit survivorship bias).
    keep: symbols (held positions) that must stay in the table even if the rule would exclude them —
    they get score = -inf and sort last, but keep their features so they can be re-researched."""
    cols = {k: v.loc[when] for k, v in feats.items()}
    full = pd.DataFrame(cols).dropna(subset=["price", "ret_5d", "atr_pct", "avg_dollar_vol"])
    if full.empty:
        return full
    df = full.nlargest(universe_n, "avg_dollar_vol") if universe_n else full
    df = df[df["avg_dollar_vol"] >= min_dollar_vol]
    if df.empty:
        return df
    if variant in ("mom_12_2", "pullback_in_uptrend"):
        df = df.dropna(subset=["mom_12_2"])
    df = df.copy()
    df["volatile"] = [bool(is_volatile(s)) for s in df.index]
    df["score"] = VARIANTS[variant](df)
    df = df[np.isfinite(df["score"])]
    df = df.sort_values("score", ascending=False)
    for sym in (keep or []):
        if sym not in df.index and sym in full.index:
            row = full.loc[[sym]].copy()
            row["volatile"] = bool(is_volatile(sym))
            row["score"] = -np.inf
            df = pd.concat([df, row])
    return df


def score_table(bars: pd.DataFrame, symbols: list[str], is_volatile, min_dollar_vol: float,
                variant: str = "pullback_in_uptrend", keep: list[str] | None = None) -> pd.DataFrame:
    """Live screen: score every symbol from a (symbol, timestamp) bars frame as of the last bar."""
    bars = bars[bars.index.get_level_values("symbol").isin(list(symbols) + list(keep or []))]
    if bars.empty:
        return pd.DataFrame()
    feats = features_panel(bars)
    return score_at(feats, feats["price"].index[-1], is_volatile, min_dollar_vol, variant, keep=keep)


def screen(settings: Settings, broker: Broker, keep: list[str] | None = None, return_bars: bool = False):
    """keep = currently held symbols: always included (and fetched) so they get re-researched every cycle.
    return_bars=True also returns the raw bars (v3 risk reviewer computes correlations from them)."""
    cfg = settings.cfg["screener"]
    symbols = list(dict.fromkeys(list(settings.symbols) + list(keep or [])))
    bars = broker.daily_bars(symbols, cfg.get("lookback_days", LOOKBACK_DAYS))
    table = score_table(bars, symbols, settings.is_volatile, cfg["min_avg_dollar_volume"],
                        cfg.get("variant", "pullback_in_uptrend"), keep=keep)
    return (table, bars) if return_bars else table


def pick_candidates(df: pd.DataFrame, n: int, held: list[str]) -> list[str]:
    """Top-N by score, always including anything currently held (so it gets re-evaluated)."""
    top = [s for s in df.index[:n]]
    for h in held:
        if h not in top:
            top.append(h)
    return top
