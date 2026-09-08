# Stock Trader — an honest experiment in LLM-assisted systematic trading

A small, fully automated **paper-trading** system: a quantitative screener over ~500 liquid US stocks,
Claude (Haiku) research agents that read anonymized news and emit numeric forecasts, a volatility-scaled
risk layer that the model cannot override, and — the part most "AI trading bot" repos skip — a fair
backtester and an evaluation harness that measure whether the LLM adds anything at all.

Two Alpaca paper accounts ($1k and $100k) run it live from a $4 VPS. **Nothing here touches real money.**

**What the evidence says so far** (see `RESEARCH.md`): on a point-in-time universe with costs and a
hold-out year, no price-based screening rule reaches statistical significance; the reversal family leans
the right way but weakly. The realistic ceiling for this class of system on large caps is roughly
break-even after costs with lower drawdown. The live experiment's real output is the scorecard/eval,
which answers "does an LLM reading news beat the screener alone?" — not P&L.

> This is a research project, not investment advice. Don't run it with real money.

## Architecture

```
universe.yaml (500 names, weekly)
   └─> screener.py  — pullback-in-uptrend / reversal / momentum variants, ATR, liquidity (free IEX data)
         └─> agents.py — Haiku: anonymized article bodies -> {expected_5d_return, confidence, catalyst, risk}
               └─> Sonnet manager (or v3 veto rule) -> buy / hold / sell
                     └─> executor.py — ATR stops, risk-per-trade sizing, heat cap, PDT guard, earnings blackout,
                                        bracket orders at the broker; Claude cannot override any of it
                           └─> journal.py + scorecard.py — every input, output, order and outcome recorded
backtest.py  — point-in-time universe, hold-out year, costs, t>=3 bar, entry-timing test
eval_agents.py — forecast accuracy, calibration, selection lift vs screener, manager decisions, parse-failure rate
dashboard.py — self-contained HTML: equity vs SPY, positions with candles, trade history
```

## What you need to do (once)

### 1. Alpaca (free)
1. Sign up at alpaca.markets, then open the dashboard and switch to **Paper Trading**.
2. The default paper account is $100k. Click "View API Keys" → generate. Put them in `.env` as `ALPACA_LARGE_KEY/SECRET`.
3. Create a **second** paper account (the paper account dropdown → create new) with **$1,000**. Generate its keys → `ALPACA_SMALL_KEY/SECRET`.
   If your dashboard only offers "reset" instead of "create", reset the balance to $1,000 on one account and sign up a second Alpaca login for the $100k one.

### 2. Anthropic API key
console.anthropic.com → API Keys → create. Add $10 of credit; that should last the month. Put it in `.env` as `ANTHROPIC_API_KEY`.

### 3. Test locally first (on your PC)
```bash
cd StockTrader
python -m venv .venv && .venv\Scripts\activate      # Windows   (mac/linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                                # then edit .env with your keys
python test_connection.py                             # must print ALL GOOD
python run_cycle.py --account small --dry-run --force # full cycle, no orders placed
```
`--dry-run` shows exactly what it *would* buy/sell. Read `reports/<today>.md` afterwards.

### 4. VPS
Recommended: **Hetzner CX22** (~€4/mo) or **DigitalOcean $6 droplet**, Ubuntu 24.04. Then:
```bash
# from your PC, copy the folder (without .venv):
scp -r StockTrader root@YOUR_VPS_IP:/opt/stocktrader
ssh root@YOUR_VPS_IP
bash /opt/stocktrader/vps/setup.sh
nano /opt/stocktrader/.env                    # paste keys
cd /opt/stocktrader && .venv/bin/python test_connection.py
```
`setup.sh` installs the schedule in `vps/crontab.txt`: $1k account daily 12:45pm PT, $100k account 6:05am + 12:45pm PT, scoreboard 1:15pm PT.

## How a cycle works (v2)
1. **Screener** (free data, ~500 names): ranks by the rule in `config.yaml → screener.variant`. Default `pullback_in_uptrend`
   = last week's losers among top-half 12-2-month momentum names (see RESEARCH.md §2 for why). Names reporting earnings
   within 2 days or unaffordable for the account are dropped. Top 6 + anything held go to Claude.
2. **Research agents** (Haiku, ~$0.001/call): each candidate gets price stats + last 36h of **article bodies** with the
   company name and ticker **hidden** → `expected_5d_return_pct / confidence / bias / catalyst / risk`. Cached 3h.
3. **Portfolio decision** (Sonnet, one call): buys only if forecast ≥ `min_expected_return_pct` and confidence ≥ `min_confidence`.
4. **Executor** (plain Python, Claude can't override): stop = 2×ATR, target = 2× the stop distance, size so a stop-out
   costs `risk_per_trade_pct` of equity, total open risk ≤ `max_heat_pct`, PDT protection on the $1k account,
   max-hold-day exits, earnings blackout. Every buy is a **bracket order** (stop + target live at the broker).
5. **Scorecard**: every candidate, Claude's forecast, and what we did → `state/scorecard.csv`; `daily_summary.py`
   fills realized 1/5/10-day returns. `python daily_summary.py --scorecard` answers "does Claude beat the screener?"
6. **Report** appended to `reports/YYYY-MM-DD.md`; `state/costs.json` tracks Claude spend.

## Widening the universe (~500 stocks)
```bash
python build_universe.py     # ~3 min: pulls Alpaca's asset list, ranks by 30-day dollar volume, writes universe.yaml
```
`universe.yaml` is merged with `watchlist.yaml` automatically (your hand-picked names and `volatile` tags always win).
The screener ranks all ~500 daily; Claude still only researches the top 6, so cost is unchanged.
On the VPS this reruns every Sunday 6pm via cron.

## Backtesting the screener — fairly (v2)
```bash
python backtest.py --sweep              # first run downloads ~5 years for the 1,500-name pool (5-10 min, ~1GB RAM: PC only)
python backtest.py --sweep --offline    # re-run from the local Parquet lake in seconds
python backtest.py --variant reversal_5d
```
What makes it fair: the eligible universe is the top 500 by dollar volume **as of each date** (from `universe_wide.yaml`),
the last 12 months are a **hold-out** reported separately, 10 bps round-trip costs are charged, and both entry timings
(next open = 6:05am cycle, or close = a 12:45pm cycle) are shown. The bar for believing a rule: **train t ≥ 3 and a
positive test edge**. `random` shows what luck alone produces; on random data nothing passes.

## Dashboard (lightweight website)
```bash
python dashboard.py --open     # builds reports/dashboard.html from live Alpaca data and opens it
python dashboard.py --demo     # fake data, just to see the layout
```
One self-contained HTML file, no internet needed to view: both accounts' return-since-start curves on one
chart (hover for values), P&L tiles, open positions, recent fills, the watchlist with 30-day sparklines
and screener scores, Claude spend, and the latest cycle notes. Re-run it any time to refresh.

On the VPS the cron rebuilds it every 30 minutes and serves the `reports/` folder on port 8080, so you
can open `http://YOUR_VPS_IP:8080/dashboard.html` from your phone. (Open port 8080 in the VPS firewall,
or keep it closed and use `ssh -L 8080:localhost:8080 root@YOUR_IP` then browse localhost:8080.)

## Evaluating the agents (not the screener)
```bash
python eval_agents.py          # report; needs ~20+ filled scorecard rows (a week or two of cycles) to mean anything
python eval_agents.py --md     # also saves reports/eval_<date>.md   (cron runs this every Friday)
python eval_agents.py --demo   # synthetic data, to see the layout
```
Five sections: (1) structured-output reliability — parse-failure rate, cache hit rate, cost per call;
(2) Haiku forecast quality — correlation with realized 5-day returns, directional accuracy, error vs a naive 0% forecast,
and calibration by confidence bucket; (3) selection value — Claude-approved vs rejected vs the raw screener top-6, with a t-stat;
(4) the manager's buy/hold/sell calls graded on the next 5 days; (5) realized trades by exit reason.
`backtest.py` tests the rules; this tests the LLM layer on top of them.

## Decision journal (look back at any trade)
Every cycle writes one JSON per candidate to `state/journal/<date>/` with the screener features, the exact
anonymized text Claude read, its raw reply, the manager's action and reason, and the order if any.
`state/trades.json` tracks each trade from entry to exit (stop / target / time stop / sold) with realized P&L;
`daily_summary.py` closes them out after each session using the broker's fills.
```bash
python journal.py HOOD            # every time HOOD was researched, what Claude said, what we did, how it ended
python journal.py HOOD --full     # ...plus the full text it read and its raw reply
python journal.py --losers        # worst closed trades with their original reasoning
python journal.py --trades        # all trades
```
The dashboard has a "Trade history" section built from the same data.

## Knobs (config.yaml)
- `screener.variant` — the ranking rule; change only after a `--sweep` says so.
- `screener.candidates_per_cycle` — the main cost lever. 6 ≈ $0.01–0.02 per cycle.
- `research.min_expected_return_pct / min_confidence` — how picky the manager is.
- `accounts.*.risk_per_trade_pct / atr_stop_mult / reward_mult / max_heat_pct` — the risk layer.
- `research.daily_usd_cap` — hard stop on Claude spend per day (default $1).
- `watchlist.yaml` — add/remove names; mark `volatile: true` for smaller sizing.

## License
MIT — see `LICENSE`. Use it, learn from it, don't trust it with money you can't lose.

## Honest caveats
- Alpaca's free feed is IEX-only: prices can differ slightly from what you see on Robinhood/TradingView. Fine for paper.
- Paper fills are optimistic (no real slippage). Real results would be worse, especially for volatile names.
- One month is not enough data to know whether this makes money. It is enough to find bugs and bad habits.
- Cron doesn't know market holidays; the script checks Alpaca's clock and skips.
