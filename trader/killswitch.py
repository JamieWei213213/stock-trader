"""Kill switch (v3, deterministic, no model). Freezes NEW buys when the account is bleeding:
  - daily:    equity down `daily_loss_pct` from the day's first cycle  -> no new buys for the rest of the day (auto-clears tomorrow)
  - drawdown: equity down `drawdown_pct` from its all-time peak        -> no new buys until `python killswitch.py --reset <account>`
Existing positions keep their bracket stops/targets at the broker; the executor's exits still run.
State: state/killswitch_<account>.json {peak, day, day_start, tripped, reason, since}.
Why not liquidate everything: in a fast drop the stops are already doing that job at the broker, and a
market sell of the whole book at the worst moment is how paper losses become real ones. The switch's job is to
stop adding risk and make you look, not to trade.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


class KillSwitch:
    def __init__(self, state_dir: Path, account: str, cfg: dict):
        self.path = state_dir / f"killswitch_{account}.json"
        self.cfg = cfg or {}
        self.account = account
        self.st = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _save(self):
        self.path.write_text(json.dumps(self.st, indent=1), encoding="utf-8")

    def check(self, equity: float) -> tuple[bool, str]:
        """Update peak / day-start with this cycle's equity and return (tripped, reason)."""
        if not self.cfg.get("enabled", True):
            return False, ""
        today = date.today().isoformat()
        st = self.st
        st["peak"] = max(float(st.get("peak", 0) or 0), equity)
        if st.get("day") != today:
            st["day"], st["day_start"] = today, equity
            if st.get("tripped") == "daily":          # daily trips auto-clear on a new day
                st.update({"tripped": None, "reason": None, "since": None})
        day_loss = 1 - equity / float(st["day_start"]) if st.get("day_start") else 0.0
        dd = 1 - equity / st["peak"] if st["peak"] else 0.0
        if not st.get("tripped"):
            if dd >= float(self.cfg.get("drawdown_pct", 0.08)):
                st.update({"tripped": "drawdown", "since": datetime.now().strftime("%Y-%m-%d %H:%M"),
                           "reason": f"equity ${equity:,.0f} is {dd * 100:.1f}% below peak ${st['peak']:,.0f} (cap {self.cfg.get('drawdown_pct', 0.08) * 100:.0f}%); "
                                     f"run `python killswitch.py --reset {self.account}` after you've looked (add --peak if equity is still far below the old peak, or it re-trips next cycle)"})
            elif day_loss >= float(self.cfg.get("daily_loss_pct", 0.03)):
                st.update({"tripped": "daily", "since": datetime.now().strftime("%Y-%m-%d %H:%M"),
                           "reason": f"equity ${equity:,.0f} is {day_loss * 100:.1f}% below today's start ${st['day_start']:,.0f} (cap {self.cfg.get('daily_loss_pct', 0.03) * 100:.0f}%); clears tomorrow"})
        self._save()
        return bool(st.get("tripped")), st.get("reason") or ""

    def reset(self):
        self.st.update({"tripped": None, "reason": None, "since": None, "peak": self.st.get("peak", 0)})
        self._save()

    def status(self) -> dict:
        return dict(self.st)
