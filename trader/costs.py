"""Tracks Claude API spend per day and enforces the daily USD cap."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path


class CostTracker:
    def __init__(self, state_dir: Path, pricing: dict, daily_cap: float):
        self.path = state_dir / "costs.json"
        self.pricing = pricing
        self.cap = daily_cap
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _today(self) -> dict:
        return self.data.setdefault(date.today().isoformat(), {"usd": 0.0, "calls": 0, "in": 0, "out": 0})

    def spent_today(self) -> float:
        return self._today()["usd"]

    def can_spend(self) -> bool:
        return self.spent_today() < self.cap

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.pricing.get(model, {"input": 3.0, "output": 15.0})
        usd = input_tokens / 1e6 * p["input"] + output_tokens / 1e6 * p["output"]
        t = self._today()
        t["usd"] += usd
        t["calls"] += 1
        t["in"] += input_tokens
        t["out"] += output_tokens
        self.path.write_text(json.dumps(self.data, indent=1))
        return usd
