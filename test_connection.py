"""Step 1 sanity check: can we reach both Alpaca paper accounts, market data, news, and Claude?

Run:  python test_connection.py
"""
import sys

from trader.settings import Settings
from trader.broker import Broker


def main():
    s = Settings()
    ok = True
    for name in ("small", "large"):
        print(f"\n=== Account '{name}' ===")
        try:
            b = Broker(s.creds(name))
            snap = b.snapshot()
            print(f"  equity=${snap.equity:,.2f}  cash=${snap.cash:,.2f}  buying_power=${snap.buying_power:,.2f}")
            print(f"  day-trades(5d)={snap.daytrade_count}  PDT flag={snap.pattern_day_trader}")
            print(f"  open positions={len(snap.positions)}  market open now={b.market_open()}")
            expected = s.account_cfg(name)["starting_cash"]
            if abs(snap.equity - expected) > expected * 0.5:
                print(f"  WARNING: equity is far from configured starting_cash={expected}. "
                      f"Did you attach the right keys to this account?")
        except Exception as e:
            ok = False
            print(f"  FAILED: {e}")

    print("\n=== Market data (via 'large' keys) ===")
    try:
        b = Broker(s.creds("large"))
        bars = b.daily_bars(["AAPL", "NVDA"], 10)
        print(f"  got {len(bars)} daily bars for AAPL/NVDA")
        news = b.recent_news("NVDA", hours=72, limit=3)
        print(f"  got {len(news)} news items for NVDA; first: {news[0]['headline'] if news else '-'}")
    except Exception as e:
        ok = False
        print(f"  FAILED: {e}")

    print("\n=== Claude API ===")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=s.anthropic_key)
        r = client.messages.create(
            model=s.cfg["research"]["research_model"], max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the word OK."}],
        )
        print(f"  {s.cfg['research']['research_model']} replied: {r.content[0].text.strip()}")
    except Exception as e:
        ok = False
        print(f"  FAILED: {e}")

    print("\n=== SEC EDGAR (v3 filings agent) ===")
    try:
        from trader.filings import Edgar, extract_quarters
        e = Edgar(s.state_dir)
        cik = e.cik("AAPL")
        q = extract_quarters(e.company_facts(cik), 8)
        print(f"  AAPL CIK {cik}: {len(q['quarters'])} quarters, latest filed {q['latest_filed']}")
    except Exception as e:
        print(f"  WARNING: EDGAR unreachable ({e}) — the filings agent will be skipped (news-only forecasts)")

    print("\nALL GOOD" if ok else "\nSomething failed — fix the items above before running cycles.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
