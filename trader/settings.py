"""Loads config.yaml, watchlist.yaml and .env into one Settings object."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class AccountCreds:
    name: str
    key: str
    secret: str


class Settings:
    def __init__(self, root: Path = ROOT):
        self.root = root
        with open(root / "config.yaml", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        with open(root / "watchlist.yaml", encoding="utf-8") as f:
            self.watchlist = yaml.safe_load(f)["stocks"]
        # universe.yaml (built weekly by build_universe.py) widens the tradeable set; watchlist.yaml wins on tags
        self.universe_file = root / "universe.yaml"
        if self.universe_file.exists():
            with open(self.universe_file, encoding="utf-8") as f:
                auto = yaml.safe_load(f)["stocks"]
            manual = {w["symbol"]: w for w in self.watchlist}
            merged = list(self.watchlist)
            for a in auto:
                if a["symbol"] not in manual:
                    merged.append(a)
                elif a.get("name"):
                    manual[a["symbol"]].setdefault("name", a["name"])
            self.watchlist = merged
        self.names: dict[str, str] = {w["symbol"]: w.get("name", "") for w in self.watchlist if w.get("name")}
        self.state_dir = root / self.cfg["paths"]["state_dir"]
        self.reports_dir = root / self.cfg["paths"]["reports_dir"]
        self.state_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

    # ---- accounts ----
    def account_cfg(self, name: str) -> dict:
        return self.cfg["accounts"][name]

    def creds(self, name: str) -> AccountCreds:
        prefix = f"ALPACA_{name.upper()}"
        key = os.getenv(f"{prefix}_KEY", "")
        secret = os.getenv(f"{prefix}_SECRET", "")
        if not key or not secret:
            raise RuntimeError(
                f"Missing {prefix}_KEY / {prefix}_SECRET in .env (copy .env.example to .env)"
            )
        return AccountCreds(name, key, secret)

    @property
    def anthropic_key(self) -> str:
        k = os.getenv("ANTHROPIC_API_KEY", "")
        if not k:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in .env")
        return k

    # ---- watchlist helpers ----
    @property
    def symbols(self) -> list[str]:
        return [s["symbol"] for s in self.watchlist]

    def is_volatile(self, symbol: str) -> bool:
        for s in self.watchlist:
            if s["symbol"] == symbol:
                return bool(s.get("volatile", False))
        return True  # unknown names are treated as risky
