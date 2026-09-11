# Stock Trader — an honest experiment in LLM-assisted systematic trading

A small, fully automated **paper-trading** system: a quantitative screener over ~500 liquid US stocks,
a multi-agent research layer (Claude Haiku agents for news, SEC filings and trade memory, run concurrently
by an orchestrator with typed state, timeouts and cost caps), a volatility-scaled risk layer that no model
can override, and — the part most "AI trading bot" repos skip — a fair backtester and an evaluation
harness with a per-agent **ablation** that measures whether each agent adds anything at all.

Two Alpaca paper accounts ($1k and $100k) run it live from a $4 VPS. **Nothing here touches real money.**

**What the evidence says so far** (see `RESEARCH.md`): on a point-in-time universe with costs and a
hold-out year, no price-based screening rule reaches statistical significance; the reversal family leans
the right way but weakly. The realistic ceiling for this class of system on large caps is roughly
break-even after costs with lower drawdown. The live experiment's real output is the scorecard/eval,
which answers "does an LLM reading news beat the screener alone?" — not P&L.

> This is a research project, not investment advice. Don't run it with real money.

## Architecture (v3)

```
universe.yaml (500 names, weekly, with industry tags)
   └─> screener.py        deterministic: pullback-in-uptrend / reversal / momentum, ATR, liquidity (free IEX data)
   └─> regime.py          deterministic: SPY vs 200-day + vol spike -> risk multiplier / buy freeze
         └─> orchestrator.py   asyncio; typed CycleState; per-agent timeout + retry; failures degrade, never crash
               ├─ news agent     (agents.py)   anonymized article bodies -> expected_5d_return, confidence, catalyst, risk
               ├─ filings agent  (filings.py)  SEC EDGAR XBRL, 8 quarters, anonymized -> next-earnings direction -> tilt
               ├─ memory agent   (memory.py)   the symbol's own past calls + outcomes + lessons -> thesis intact / broken
               └─ forecaster     deterministic: news + filings tilt + memory nudge; every component kept for the ablation
                     └─> decide.py        deterministic rule: screener picks, agents can only VETO; sell / rotate rules; no P&L shown to any model
                           └─> risk reviewer (orchestrator.py)  industry cap, correlation with the book
                                 └─> executor.py   ATR stops, risk-per-trade sizing, heat cap, stop cap, PDT guard, earnings windows,
                                                   bracket orders at the broker
                                       └─> journal.py + scorecard.py + state/cycles/   every input, output, order, outcome
post-mortem agent (memory.py)  after each closed trade -> lesson -> feeds the memory agent and the monthly report
ask.py           'ask the journal': plain-English questions answered from the system's own records only
propose.py       English idea -> strict spec (LLM) -> fair backtest (unchanged) -> rule-based verdict + multiple-testing count
backtest.py      point-in-time universe, hold-out year, costs, t>=3 bar, entry-timing test
eval_agents.py   forecast quality, calibration, selection lift, veto accuracy, per-agent ABLATION, fleet health
monthly_report.py  one page: accounts vs SPY, trades, eval, what changed
dashboard.py     self-contained HTML: equity vs SPY, positions with candles, trade history
```

Design rule: an LLM agent exists only where there is **text** to read. Everything numeric is a deterministic
node, so it can be unit-tested and the eval can say exactly which agent earns its cost. The critic / bull-bear
agent is a config flag that is **off**: the literature and our own eval say it adds cost, not accuracy.

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
python -m tests.test_offline && python -m tests.test_v3  # offline unit tests (no keys needed)
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
`setup.sh` installs the schedule in `vps/crontab.txt`: $1k account daily 12:45pm PT, $100k account 6:05am (manage-only) + 12:45pm PT,
scoreboard 1:15pm PT, eval Fridays, monthly report on the 1st. The filings agent needs EDGAR (`data.sec.gov`) reachable from the VPS.

## How a cycle works (v3)
1. **Screener** (free data, ~500 names): ranks by the rule in `config.yaml → screener.variant`. Default `pullback_in_uptrend`
   = last week's losers among top-half 12-2-month momentum names (see RESEARCH.md §2 for why). Names with earnings inside
   the hold period (or that reported yesterday), with a 2×ATR stop wider than 8%, or unaffordable for the account are dropped.
   Top 15 (large) / 6 (small) plus anything held become the candidates.
2. **Regime monitor**: SPY under its 200-day average halves risk per trade; a volatility spike on top freezes new buys.
3. **Agents, concurrently** (`orchestrator.py`, Haiku, ~$0.001–0.005/call): for each candidate the **news** agent reads
   36h of anonymized article bodies → numeric 5-day forecast; the **filings** agent reads 8 quarters of anonymized
   financial statements from EDGAR (cached 90 days) → next-earnings direction → a small tilt; the **memory** agent reads
   what we said about this name before and what happened → thesis intact / broken. A per-call timeout and one retry;
   a failed agent means a lower-confidence or missing forecast, never a crashed cycle.
4. **Decision rule** (`decide.py`, deterministic): buy screener picks in rank order unless the agents **veto**
   (forecast < −0.5%, bearish with confidence, or thesis broken); sell held names whose fresh forecast is clearly negative
   or whose thesis broke; when full, at most one rotation per cycle and only for a ≥1% forecast edge. No model sees P&L.
5. **Risk reviewer**: drops buys that would put a 4th name in one industry or that correlate > 0.75 with two names already held.
6. **Executor** (plain Python, no model can override): stop = 2×ATR, target = 2× the stop distance, size so a stop-out
   costs `risk_per_trade_pct` × regime multiplier of equity, total open risk ≤ `max_heat_pct`, PDT protection on the $1k
   account. Every buy is a **bracket order** (stop + target live at the broker). The 6:05am cycle runs with `--mode manage`:
   exits only, no entries (the backtest favoured close entries).
   **Exits, large account ("middle path", `accounts.large.hold`)**: no hard time stop. A **trailing stop** ratchets the
   broker's stop up to (high since entry − 1.5×ATR) once the trade is ahead by one stop distance, and never down; the
   **thesis check** (sell on a clearly negative fresh forecast) applies from day 5, a broken thesis sells at any age; from day
   10 a **flat** position (±1% of entry) is *stale* and hands its slot to any newcomer the agents pass, no rotation edge needed;
   day 30 is a hard ceiling. The small account keeps its hard 10-day stop (PDT + $1k leave no room for idling).
   Exit reasons in the ledger: `stop_loss`, `trail_stop`, `take_profit`, `sold: …` (thesis), `stale`, `hard_max`, `time_stop`.
   **Kill switch** (`killswitch.*`, deterministic): equity down 3% from the day's first cycle, or 8% from its peak, freezes new
   buys (exits and broker stops keep running). Daily trips clear tomorrow; a drawdown trip stays until
   `python killswitch.py --reset large` — after you've looked.
7. **Records**: `state/journal/` (everything each agent saw and said), `state/cycles/` (per-cycle agent stats),
   `state/scorecard.csv` (per-agent components + realized returns → the ablation), `state/trades.json`, `reports/YYYY-MM-DD.md`.
8. **After the close** (`daily_summary.py`): equity history, journal reconciled with broker fills, scorecard filled, and the
   **post-mortem** agent writes a two-line lesson for every trade that closed → `state/lessons.jsonl`.

Legacy v2 (Sonnet portfolio manager) is still available with `decision.mode: manager` in `config.yaml` for A/B runs.

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

## Propose a rule in English — and let the fair backtest judge it (v3)
```bash
python propose.py "buy last week's biggest losers that are still in a 12-month uptrend"
python propose.py --spec my_rule.json      # hand-written spec, no LLM
python propose.py --list                   # everything tried so far and how it did
```
Haiku only translates the idea into a strict spec (a weighted sum of z-scored features plus optional filters — it cannot
write code or touch the data). `backtest.py` then runs it unchanged: point-in-time universe, hold-out year, costs, random
baseline, both entry timings, next to the current variant. The verdict is a rule (train t ≥ 3 and positive hold-out edge),
every proposal is logged, and the tool prints the multiple-testing count with the best t you'd expect from luck alone,
because trying twenty ideas and keeping the winner is how backtests lie. PC only (needs the Parquet lake).

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
Eight sections: (1) structured-output reliability — parse-failure rate, cache hit rate, cost per call;
(2) forecast quality — correlation with realized 5-day returns, directional accuracy, error vs a naive 0% forecast,
and calibration by confidence bucket; (3) selection value — approved vs rejected vs the raw screener, with a t-stat;
(4) manager decisions (v2) graded on the next 5 days; (5) realized trades by exit reason; (6) **veto accuracy** — do the
names the agents vetoed do worse than the ones they passed, by veto reason; (7) **ablation** — forecast quality of
news-only vs news+filings vs news+memory vs combined, so each agent's marginal value is a number; (8) **fleet health** —
per-agent success rate, cache rate, cost and latency, plus the post-mortem lesson mix.
`backtest.py` tests the rules; this tests the LLM layer on top of them. `python monthly_report.py` rolls it all into one page.

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

## Claude Desktop over the journal (MCP, read-only)
```bash
pip install mcp
python mcp_journal.py            # stdio server; normally launched by Claude Desktop, see the file header for the config
```
Seven tools: `get_trade`, `list_trades`, `scorecard_summary`, `search_journal`, `cost_report`, `cycle_log`,
`killswitch_status`. Reads `state/` and `reports/` only; nothing can place, cancel or change anything. Run it on your PC
against a copy of the server's state (`scp -r root@VPS:/opt/stocktrader/state .`), not on the VPS.

## Ask the journal (v3)
```bash
python ask.py "why did we sell CIEN"
python ask.py "which vetoes were wrong this month"
python ask.py                                   # interactive
```
A Haiku agent that answers plain-English questions from the system's own records only (journal, trades, scorecard,
cycle summaries, lessons, equity history, latest eval) and says what's missing rather than guessing. Symbols in the
question pull in that name's full history including the exact text the news agent read. ~$0.002-0.01 per question,
cost-capped like every other agent; it cannot place orders or change anything.

## Knobs (config.yaml)
- `screener.variant` — the ranking rule; change only after a `--sweep` says so.
- `accounts.*.candidates_per_cycle` — the main cost lever. 15 names × 3 agents ≈ $0.03–0.05 per cycle (filings are cached).
- `decision.*` — veto thresholds, sell threshold, rotation edge. `decision.mode: manager` brings back the v2 Sonnet manager.
- `agents.*.enabled` — switch any agent off; the ablation in `eval_agents.py` tells you which ones to keep.
- `regime.*` — the market-state filter; `risk.*` — stop cap, industry cap, correlation cap; `killswitch.*` — loss freezes.
- `accounts.large.hold.*` — the middle-path exits (review day, stale day/band, trailing activation/distance, hard ceiling).
  Delete the block to go back to a hard `max_hold_days` time stop.
- `accounts.*.risk_per_trade_pct / atr_stop_mult / reward_mult / max_heat_pct` — the risk layer.
- `research.daily_usd_cap` — hard stop on Claude spend per day (default $1.50).
- `watchlist.yaml` — add/remove names; mark `volatile: true` for smaller sizing.

## License
MIT — see `LICENSE`. Use it, learn from it, don't trust it with money you can't lose.

## Honest caveats
- Alpaca's free feed is IEX-only: prices can differ slightly from what you see on Robinhood/TradingView. Fine for paper.
- Paper fills are optimistic (no real slippage). Real results would be worse, especially for volatile names.
- One month is not enough data to know whether this makes money. It is enough to find bugs and bad habits.
- Cron doesn't know market holidays; the script checks Alpaca's clock and skips.
