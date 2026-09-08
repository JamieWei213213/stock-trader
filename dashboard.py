"""Builds self-contained HTML dashboards: reports/dashboard.html (overview),
reports/dashboard_small.html ($1k) and reports/dashboard_large.html ($100k), linked by a nav bar.

Shows both accounts' equity curves (% return, one axis), P&L tiles, open positions,
recent fills, watchlist prices with 30-day sparklines, Claude spend, and the latest
cycle notes. No internet needed to view it — everything is inline.

Usage:
  python dashboard.py            # fetch live data from Alpaca, write reports/dashboard.html
  python dashboard.py --open     # ...and open it in your browser
  python dashboard.py --demo     # fake data, no API keys needed (for checking the layout)

On the VPS, cron regenerates it after each cycle; view it with:
  cd /opt/stocktrader && .venv/bin/python -m http.server 8080 --directory reports
then open http://YOUR_VPS_IP:8080/dashboard.html
"""
from __future__ import annotations

import argparse
import json
import re
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trader.settings import Settings

ACCOUNT_LABELS = {"small": "$1k account (daily)", "large": "$100k account (twice daily)"}


# ----------------------------------------------------------------------------- data
def collect_live(s: Settings) -> dict:
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
    from trader.broker import Broker
    from trader.screener import screen

    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "accounts": {}, "watchlist": []}
    brokers = {}
    for name in ("small", "large"):
        try:
            b = Broker(s.creds(name))
        except Exception as e:
            out["accounts"][name] = {"error": str(e)}
            continue
        brokers[name] = b
        snap = b.snapshot()
        start_cash = s.account_cfg(name)["starting_cash"]
        hist = b.trading.get_portfolio_history(
            GetPortfolioHistoryRequest(period="3M", timeframe="1D", extended_hours=False))
        pts = []
        for t, eq in zip(hist.timestamp, hist.equity):
            if eq:
                pts.append([datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d"), float(eq)])
        # ensure today's live equity is the last point
        today = datetime.now().strftime("%Y-%m-%d")
        if pts and pts[-1][0] == today:
            pts[-1][1] = snap.equity
        else:
            pts.append([today, snap.equity])
        fills = []
        orders = b.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=50))
        for o in orders:
            if o.filled_at and o.filled_avg_price:
                fills.append({
                    "time": o.filled_at.astimezone().strftime("%m-%d %H:%M"),
                    "symbol": o.symbol, "side": str(o.side).split(".")[-1].lower(),
                    "qty": float(o.filled_qty or 0), "price": float(o.filled_avg_price),
                    "type": str(o.type).split(".")[-1].lower(),
                })
        # stop/target levels from the executor's state file + 30-day candles for each open position
        pfile = s.state_dir / f"positions_{name}.json"
        levels = json.loads(pfile.read_text(encoding="utf-8")) if pfile.exists() else {}
        candles = {}
        if snap.positions:
            pb = b.daily_bars([p["symbol"] for p in snap.positions], 30)
            for p in snap.positions:
                try:
                    sub = pb.xs(p["symbol"], level="symbol").tail(30)
                    candles[p["symbol"]] = [[str(i.date()), round(float(r.open), 2), round(float(r.high), 2),
                                             round(float(r.low), 2), round(float(r.close), 2)] for i, r in sub.iterrows()]
                except KeyError:
                    pass
                lv = levels.get(p["symbol"], {})
                if isinstance(lv, dict):
                    p["stop"], p["target"], p["entry_date"] = lv.get("stop"), lv.get("target"), lv.get("date")
        out["accounts"][name] = {
            "label": ACCOUNT_LABELS[name], "start": start_cash, "equity": snap.equity, "cash": snap.cash,
            "daytrades": snap.daytrade_count, "positions": snap.positions, "history": pts, "fills": fills,
            "candles": candles,
        }

    # watchlist prices + sparklines (use whichever broker works)
    b = brokers.get("large") or brokers.get("small")
    if b:
        table = screen(s, b)
        held = {p["symbol"] for a in out["accounts"].values() for p in a.get("positions", [])}
        show = [sym for sym in table.index[:30]] + [h for h in held if h in table.index and h not in table.index[:30]]
        bars = b.daily_bars(show, 30)
        out["universe_size"] = len(table)
        for sym in show:
            r = table.loc[sym]
            try:
                closes = [round(float(x), 2) for x in bars.xs(sym, level="symbol")["close"].tail(30)]
            except KeyError:
                closes = []
            out["watchlist"].append({
                "symbol": sym, "price": round(float(r["price"]), 2), "ret_1d": float(r["ret_1d"]),
                "ret_5d": float(r["ret_5d"]), "ret_20d": float(r["ret_20d"]), "score": round(float(r["score"]), 2),
                "volatile": bool(r["volatile"]), "spark": closes,
            })
    out["costs"] = _costs(s)
    out["notes"] = _latest_notes(s)
    out["trades"] = _trades(s)
    out["benchmark"] = _benchmark(b, out["accounts"]) if b else None
    return out


def _benchmark(b, accounts: dict) -> dict | None:
    """SPY % return since the first day any account has history (the start of the experiment)."""
    firsts = [a["history"][0][0] for a in accounts.values() if a.get("history")]
    if not firsts:
        return None
    start = min(firsts)
    try:
        bars = b.daily_bars(["SPY"], 120).xs("SPY", level="symbol")["close"]
    except Exception:
        return None
    ser = bars[[str(i.date()) >= start for i in bars.index]]
    if ser.empty:
        return None
    base = float(ser.iloc[0])
    return {"symbol": "SPY", "pts": [[str(i.date()), round((float(v) / base - 1) * 100, 3)] for i, v in ser.items()]}


def _trades(s: Settings) -> list[dict]:
    from trader.journal import Journal
    tr = Journal(s.state_dir).trades
    return sorted(tr, key=lambda t: t["entry_time"], reverse=True)[:40]


def collect_demo(s: Settings) -> dict:
    import random
    random.seed(3)
    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M") + " (DEMO DATA)", "accounts": {}, "watchlist": []}
    for name, start in (("small", 1000), ("large", 100000)):
        eq, pts = start, []
        for i in range(22):
            d = (datetime.now() - timedelta(days=30 - i * 1.4)).strftime("%Y-%m-%d")
            eq *= 1 + random.gauss(0.001 if name == "small" else -0.0005, 0.012)
            pts.append([d, round(eq, 2)])
        out["accounts"][name] = {
            "label": ACCOUNT_LABELS[name], "start": start, "equity": pts[-1][1], "cash": pts[-1][1] * 0.4,
            "daytrades": 1 if name == "large" else 0,
            "positions": [{"symbol": "SOFI", "qty": 20, "avg_entry": 14.2, "current": 14.9, "market_value": 298,
                           "unrealized_pl": 14, "unrealized_plpc": 0.049, "stop": 13.4, "target": 15.8, "entry_date": "2026-09-01"},
                          {"symbol": "XOM", "qty": 2, "avg_entry": 118, "current": 116.5, "market_value": 233,
                           "unrealized_pl": -3, "unrealized_plpc": -0.013, "stop": 113.5, "target": 127.0, "entry_date": "2026-09-02"}],
            "history": pts,
            "candles": {sym: [[(datetime.now() - timedelta(days=30 - k)).strftime("%Y-%m-%d")] +
                              (lambda o: [o, round(o * (1 + abs(random.gauss(0, 0.012))), 2),
                                          round(o * (1 - abs(random.gauss(0, 0.012))), 2),
                                          round(o * (1 + random.gauss(0, 0.012)), 2)])(round(base * (1 + random.gauss(0, 0.02) * k / 10), 2))
                              for k in range(30)] for sym, base in (("SOFI", 14.0), ("XOM", 117.0))},
            "fills": [{"time": "09-01 06:31", "symbol": "SOFI", "side": "buy", "qty": 20, "price": 14.2, "type": "market"},
                      {"time": "09-02 06:31", "symbol": "XOM", "side": "buy", "qty": 2, "price": 118.0, "type": "market"}],
        }
    for st in s.watchlist:
        base = random.uniform(15, 600)
        spark = [round(base * (1 + random.gauss(0, 0.02) * k / 10), 2) for k in range(30)]
        out["watchlist"].append({"symbol": st["symbol"], "price": spark[-1], "ret_1d": random.gauss(0, 0.02),
                                 "ret_5d": random.gauss(0, 0.05), "ret_20d": random.gauss(0, 0.1),
                                 "score": round(random.gauss(0, 1), 2), "volatile": st["volatile"], "spark": spark})
    out["watchlist"].sort(key=lambda w: -w["score"])
    out["costs"] = {"today": 0.0165, "month": 0.41, "calls_month": 96}
    spy, pts = 0.0, []
    for i in range(22):
        d = (datetime.now() - timedelta(days=30 - i * 1.4)).strftime("%Y-%m-%d")
        spy += random.gauss(0.05, 0.6)
        pts.append([d, round(spy, 3)])
    out["benchmark"] = {"symbol": "SPY", "pts": pts}
    out["notes"] = "## 06:05 — small\nEquity $1,012.40 | cash $480.10 | positions ['SOFI','XOM']\n\n- **SOFI** +2.1% / conf 0.7 (bullish): ...\n"
    out["trades"] = [
        {"account": "large", "symbol": "PANW", "status": "closed", "entry_time": "2026-08-25 12:46", "qty": 12, "entry_price": 326.52,
         "stop": 286.25, "target": 407.06, "entry_reason": "Meets return and confidence thresholds; earnings beat", "catalyst": "Earnings beat on AI security tailwinds",
         "forecast_5d_pct": 2.8, "exit_time": "2026-09-01 06:06", "exit_price": 331.10, "exit_reason": "time_stop", "pnl": 54.96, "pnl_pct": 0.014},
        {"account": "large", "symbol": "CRDO", "status": "closed", "entry_time": "2026-08-27 12:46", "qty": 40, "entry_price": 95.10,
         "stop": 88.20, "target": 108.90, "entry_reason": "Post-earnings stabilization", "catalyst": "Analyst support despite margin concerns",
         "forecast_5d_pct": 2.5, "exit_time": "2026-08-29 09:31", "exit_price": 88.15, "exit_reason": "stop_loss", "pnl": -278.0, "pnl_pct": -0.073},
        {"account": "small", "symbol": "SOFI", "status": "open", "entry_time": "2026-09-01 12:46", "qty": 20, "entry_price": 14.2,
         "stop": 13.4, "target": 15.8, "entry_reason": "Bullish forecast above threshold", "catalyst": "Bank charter progress",
         "forecast_5d_pct": 2.2, "exit_time": None, "exit_price": None, "exit_reason": None, "pnl": None, "pnl_pct": None},
    ]
    return out


def _costs(s: Settings) -> dict:
    p = s.state_dir / "costs.json"
    if not p.exists():
        return {"today": 0, "month": 0, "calls_month": 0}
    data = json.loads(p.read_text())
    today = datetime.now().strftime("%Y-%m-%d")
    month = today[:7]
    return {"today": data.get(today, {}).get("usd", 0),
            "month": sum(v["usd"] for k, v in data.items() if k.startswith(month)),
            "calls_month": sum(v["calls"] for k, v in data.items() if k.startswith(month))}


def _latest_notes(s: Settings) -> str:
    reps = sorted(s.reports_dir.glob("20*.md"))
    if not reps:
        return "(no cycle reports yet)"
    text = reps[-1].read_text(encoding="utf-8", errors="replace")
    return text[-3500:]


# ----------------------------------------------------------------------------- html
TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stock Trader Dashboard</title>
<style>
:root{color-scheme:light dark;
 --bg:#f6f6f4;--surface:#fcfcfb;--line:#e4e3df;--text:#0b0b0b;--text2:#52514e;--muted:#8a8985;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0a7d33;--bad:#c62828;--grid:#ebebe8}
@media(prefers-color-scheme:dark){:root{--bg:#121211;--surface:#1a1a19;--line:#2c2c2a;--text:#fff;--text2:#c3c2b7;--muted:#8a8985;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4cbf73;--bad:#ef6b6b;--grid:#262624}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:28px 0 10px;color:var(--text2);font-weight:600;letter-spacing:.02em;text-transform:uppercase}
.sub{color:var(--muted);font-size:12px;margin-bottom:20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--text2)}.tile .v{font-size:24px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.tile .d{font-size:12px;color:var(--text2);margin-top:2px}
.up{color:var(--good)}.down{color:var(--bad)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;position:relative}
.legend{display:flex;gap:18px;font-size:12px;color:var(--text2);margin-bottom:8px}
svg text{fill:var(--text2);font-size:11px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:6px 9px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;white-space:nowrap}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font-size:11px;text-align:left;color:var(--muted);font-weight:600;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid var(--grid)}td.r,th.r{text-align:right}
.tag{font-size:10px;padding:1px 6px;border-radius:4px;border:1px solid var(--line);color:var(--text2)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:800px){.two{grid-template-columns:1fr}}
.scroll{overflow-x:auto}pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;color:var(--text2);margin:0}
.empty{color:var(--muted);font-style:italic;padding:8px}
.candles{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;margin-top:12px}.candles .card h3{margin:0 0 6px;font-size:13px}.candles .card .meta{font-size:11px;color:var(--text2);margin-bottom:4px}
.nav{display:flex;gap:6px;margin-bottom:14px}.nav a{padding:5px 12px;border:1px solid var(--line);border-radius:999px;color:var(--text2);text-decoration:none;font-size:12px;background:var(--surface)}.nav a.on{background:var(--text);color:var(--bg);border-color:var(--text)}
</style></head><body><div class="wrap">
<div class="nav" id="nav"></div>
<h1 id="title">Stock Trader — paper trading dashboard</h1>
<div class="sub">Generated __GENERATED__ · Claude spend today $__COST_TODAY__ · this month $__COST_MONTH__ (__CALLS__ calls)</div>
<div class="tiles" id="tiles"></div>
<h2 id="chartTitle">Return since start</h2>
<div class="card"><div class="legend" id="legend"></div><div id="chart"></div><div class="tip" id="tip"></div></div>
<h2>Open positions</h2><div class="two" id="positions"></div>
<div class="candles" id="candles"></div>
<h2>Recent fills</h2><div class="two" id="fills"></div>
<h2 id="watchTitle">Watchlist</h2><div class="card scroll" id="watch"></div>
<h2>Trade history</h2><div class="card scroll" id="trades"></div>
<h2>Latest cycle notes</h2><div class="card"><pre id="notes"></pre></div>
</div>
<script>
const D=__DATA__;
const fmt$=n=>(n<0?"-":"")+"$"+Math.abs(n).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const pct=n=>(n>=0?"+":"")+(n*100).toFixed(2)+"%";
const cls=n=>n>=0?"up":"down";
const S={small:"var(--s2)",large:"var(--s1)",spy:"var(--s3)"};
const PAGE=D.page||"all";const names=Object.keys(D.accounts).filter(k=>!D.accounts[k].error&&(PAGE==="all"||k===PAGE));
document.getElementById("nav").innerHTML=[["all","dashboard.html","Overview"],["small","dashboard_small.html","$1k account"],["large","dashboard_large.html","$100k account"]].map(([k,f,l])=>`<a href="${f}" class="${PAGE===k?'on':''}">${l}</a>`).join("");
if(PAGE!=="all"&&D.accounts[PAGE]){document.getElementById("title").textContent="Stock Trader — "+D.accounts[PAGE].label;}
const SINGLE=names.length===1;if(SINGLE)document.querySelectorAll(".two").forEach(e=>e.style.gridTemplateColumns="1fr");
// tiles
let t="";for(const k of names){const a=D.accounts[k],pl=a.equity-a.start;
t+=`<div class="tile"><div class="k"><span class="dot" style="background:${S[k]}"></span>${a.label}</div><div class="v">${fmt$(a.equity)}</div><div class="d ${cls(pl)}">${fmt$(pl)} (${pct(pl/a.start)}) · ${a.positions.length} positions · cash ${fmt$(a.cash)}</div></div>`;}
for(const k of Object.keys(D.accounts))if(D.accounts[k].error)t+=`<div class="tile"><div class="k">${k}</div><div class="d down">${D.accounts[k].error}</div></div>`;
document.getElementById("tiles").innerHTML=t;
// chart
(function(){const W=1100,H=300,L=52,R=16,T=14,B=28;const series=names.map(k=>({k,label:D.accounts[k].label,pts:D.accounts[k].history.map(p=>[p[0],SINGLE?p[1]:(p[1]/D.accounts[k].start-1)*100])}));
if(D.benchmark&&D.benchmark.pts.length>1){const base0=SINGLE?D.accounts[names[0]].start:null;series.push({k:"spy",label:"SPY (buy & hold)",bench:true,pts:D.benchmark.pts.map(p=>[p[0],SINGLE?base0*(1+p[1]/100):p[1]])});}
if(SINGLE)document.getElementById("chartTitle").textContent="Account equity";
const yl=v=>SINGLE?"$"+Math.round(v).toLocaleString():v.toFixed(1)+"%";
const dates=[...new Set(series.flatMap(s=>s.pts.map(p=>p[0])))].sort();if(!dates.length){document.getElementById("chart").innerHTML='<div class="empty">No history yet.</div>';return;}
const ys=series.flatMap(s=>s.pts.map(p=>p[1]));const base=SINGLE?D.accounts[names[0]].start:0;let lo=Math.min(base,...ys),hi=Math.max(base,...ys);if(hi-lo<(SINGLE?base*0.005:0.5)){const e=SINGLE?base*0.0025:0.25;hi+=e;lo-=e}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
const x=i=>L+(dates.length<2?0:i*(W-L-R)/(dates.length-1)),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
let g="";const ticks=5;for(let i=0;i<=ticks;i++){const v=lo+(hi-lo)*i/ticks,yy=y(v);g+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="var(--grid)"/><text x="${L-6}" y="${yy+4}" text-anchor="end">${yl(v)}</text>`;}
g+=`<line x1="${L}" x2="${W-R}" y1="${y(base)}" y2="${y(base)}" stroke="var(--muted)" stroke-dasharray="3 3"/>`;
const step=Math.max(1,Math.ceil(dates.length/8));dates.forEach((d,i)=>{if(i%step===0||i===dates.length-1)g+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle">${d.slice(5)}</text>`;});
let lines="";for(const s of series){const m=new Map(s.pts);const path=dates.map((d,i)=>m.has(d)?`${x(i)},${y(m.get(d))}`:null).filter(Boolean);lines+=`<polyline fill="none" stroke="${S[s.k]}" stroke-width="${s.bench?1.5:2}" ${s.bench?'stroke-dasharray="5 4"':''} stroke-linejoin="round" points="${path.join(" ")}"/>`;
const last=s.pts[s.pts.length-1];const li=dates.indexOf(last[0]);if(s.bench){const b0=SINGLE?D.accounts[names[0]].start:null;lines+=`<text x="${x(li)-8}" y="${y(last[1])+14}" text-anchor="end" style="fill:${S[s.k]}">SPY ${pct((SINGLE?last[1]/b0-1:last[1]/100))}</text>`;continue;}lines+=`<circle cx="${x(li)}" cy="${y(last[1])}" r="4" fill="${S[s.k]}" stroke="var(--surface)" stroke-width="2"/><text x="${x(li)-8}" y="${y(last[1])-9}" text-anchor="end" style="fill:var(--text);font-weight:600">${SINGLE?fmt$(last[1]):pct(last[1]/100)}</text>`;}
document.getElementById("chart").innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">${g}${lines}<line id="xh" x1="0" x2="0" y1="${T}" y2="${H-B}" stroke="var(--muted)" style="display:none"/><rect id="hit" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/></svg>`;
document.getElementById("legend").innerHTML=series.map(s=>`<span><span class="dot" style="background:${S[s.k]}"></span>${s.label}</span>`).join("");
const svg=document.querySelector("#chart svg"),tip=document.getElementById("tip"),xh=document.getElementById("xh");
svg.addEventListener("mousemove",e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)*W/r.width;let i=Math.round((px-L)/((W-L-R)/Math.max(1,dates.length-1)));i=Math.max(0,Math.min(dates.length-1,i));
xh.setAttribute("x1",x(i));xh.setAttribute("x2",x(i));xh.style.display="";let h=`<b>${dates[i]}</b>`;for(const s of series){const m=new Map(s.pts);if(!m.has(dates[i]))continue;if(s.bench){const v=m.get(dates[i]);const b0=SINGLE?D.accounts[names[0]].start:null;h+=`<br><span class="dot" style="background:${S[s.k]}"></span>SPY ${pct(SINGLE?v/b0-1:v/100)}`;continue;}const eq=D.accounts[s.k].history.find(p=>p[0]===dates[i])[1];h+=`<br><span class="dot" style="background:${S[s.k]}"></span>${fmt$(eq)} (${pct(eq/D.accounts[s.k].start-1)})`;}
tip.innerHTML=h;tip.style.display="block";const card=svg.parentElement.parentElement.getBoundingClientRect();let tx=e.clientX-card.left+14;if(tx+tip.offsetWidth>card.width-10)tx-=tip.offsetWidth+28;tip.style.left=tx+"px";tip.style.top=(e.clientY-card.top-10)+"px";});
svg.addEventListener("mouseleave",()=>{tip.style.display="none";xh.style.display="none";});})();
// positions & fills
function table(rows,cols){if(!rows.length)return'<div class="empty">none</div>';return`<table><tr>${cols.map(c=>`<th class="${c.r?'r':''}">${c.h}</th>`).join("")}</tr>${rows.map(r=>`<tr>${cols.map(c=>`<td class="${c.r?'r':''} ${c.cls?c.cls(r):''}">${c.f(r)}</td>`).join("")}</tr>`).join("")}</table>`;}
document.getElementById("positions").innerHTML=names.map(k=>`<div class="card"><div class="legend"><span><span class="dot" style="background:${S[k]}"></span>${D.accounts[k].label}</span></div>${table(D.accounts[k].positions,[
{h:"Symbol",f:r=>`<b>${r.symbol}</b>`},{h:"Qty",r:1,f:r=>r.qty},{h:"Entry",r:1,f:r=>r.avg_entry.toFixed(2)},{h:"Now",r:1,f:r=>r.current.toFixed(2)},{h:"Value",r:1,f:r=>fmt$(r.market_value)},{h:"P&L",r:1,cls:r=>cls(r.unrealized_pl),f:r=>`${fmt$(r.unrealized_pl)} (${pct(r.unrealized_plpc)})`}])}</div>`).join("");
document.getElementById("fills").innerHTML=names.map(k=>`<div class="card"><div class="legend"><span><span class="dot" style="background:${S[k]}"></span>${D.accounts[k].label}</span></div>${table(D.accounts[k].fills.slice(0,15),[
{h:"Time",f:r=>r.time},{h:"Side",f:r=>`<span class="tag">${r.side}</span>`},{h:"Symbol",f:r=>`<b>${r.symbol}</b>`},{h:"Qty",r:1,f:r=>r.qty},{h:"Price",r:1,f:r=>r.price.toFixed(2)},{h:"Type",f:r=>r.type}])}</div>`).join("");
// candlestick charts per open position, with entry / stop / target lines
(function(){const host=document.getElementById("candles");let html="";
for(const k of names){const A=D.accounts[k];for(const p of A.positions){const c=(A.candles||{})[p.symbol];if(!c||c.length<2)continue;
const W=340,H=170,L=44,R=8,T=10,B=22;const lvl=[["entry",p.avg_entry,"var(--text2)"],["stop",p.stop,"var(--bad)"],["target",p.target,"var(--good)"]].filter(x=>x[1]);
let lo=Math.min(...c.map(x=>x[3]),...lvl.map(x=>x[1])),hi=Math.max(...c.map(x=>x[2]),...lvl.map(x=>x[1]));const pad=(hi-lo)*.08||1;lo-=pad;hi+=pad;
const x=i=>L+(i+0.5)*(W-L-R)/c.length,y=v=>T+(hi-v)/(hi-lo)*(H-T-B),bw=Math.max(2,(W-L-R)/c.length*0.6);
let g="";for(let i=0;i<=3;i++){const v=lo+(hi-lo)*i/3;g+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L-4}" y="${y(v)+4}" text-anchor="end">${v.toFixed(v<20?2:1)}</text>`;}
c.forEach((d,i)=>{const [dt,o,h,l,cl]=d,up=cl>=o,col=up?"var(--good)":"var(--bad)";g+=`<line x1="${x(i)}" x2="${x(i)}" y1="${y(h)}" y2="${y(l)}" stroke="${col}" stroke-width="1"/><rect x="${x(i)-bw/2}" y="${y(Math.max(o,cl))}" width="${bw}" height="${Math.max(1,Math.abs(y(o)-y(cl)))}" fill="${up?'var(--surface)':col}" stroke="${col}" stroke-width="1"/>`;if(i%7===0)g+=`<text x="${x(i)}" y="${H-6}" text-anchor="middle">${dt.slice(5)}</text>`;});
for(const [name,v,col] of lvl){g+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${col}" stroke-dasharray="4 3"/><text x="${W-R-2}" y="${y(v)-3}" text-anchor="end" style="fill:${col}">${name} ${v.toFixed(2)}</text>`;}
const pl=p.unrealized_plpc;html+=`<div class="card"><h3><span class="dot" style="background:${S[k]}"></span>${p.symbol} <span class="${cls(pl)}">${pct(pl)}</span></h3><div class="meta">${p.qty} sh @ ${p.avg_entry.toFixed(2)} → ${p.current.toFixed(2)}${p.entry_date?' · since '+p.entry_date:''}</div><svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg></div>`;}}
host.innerHTML=html;})();
// watchlist
function spark(v){if(v.length<2)return"";const w=90,h=24,lo=Math.min(...v),hi=Math.max(...v);const p=v.map((y,i)=>`${(i/(v.length-1)*w).toFixed(1)},${(h-2-(hi===lo?h/2:(y-lo)/(hi-lo)*(h-4))).toFixed(1)}`).join(" ");return`<svg width="${w}" height="${h}"><polyline fill="none" stroke="${v[v.length-1]>=v[0]?'var(--good)':'var(--bad)'}" stroke-width="1.5" points="${p}"/></svg>`;}
if(D.universe_size)document.getElementById("watchTitle").textContent=`Top 30 of ${D.universe_size} screened (plus held)`;
document.getElementById("watch").innerHTML=table(D.watchlist,[{h:"Symbol",f:r=>`<b>${r.symbol}</b>${r.volatile?' <span class="tag">volatile</span>':''}`},{h:"Price",r:1,f:r=>r.price.toFixed(2)},
{h:"1d",r:1,cls:r=>cls(r.ret_1d),f:r=>pct(r.ret_1d)},{h:"5d",r:1,cls:r=>cls(r.ret_5d),f:r=>pct(r.ret_5d)},{h:"20d",r:1,cls:r=>cls(r.ret_20d),f:r=>pct(r.ret_20d)},{h:"Screen score",r:1,f:r=>r.score.toFixed(2)},{h:"30 days",f:r=>spark(r.spark)}]);
document.getElementById("notes").textContent=D.notes;
// trade history (from the decision journal)
(function(){const tr=(D.trades||[]).filter(t=>PAGE==="all"||t.account===PAGE);const el=document.getElementById("trades");
if(!tr.length){el.innerHTML='<div class="empty">no trades in the journal yet</div>';return;}
const closed=tr.filter(t=>t.status==="closed");const tot=closed.reduce((a,t)=>a+t.pnl,0);const wins=closed.filter(t=>t.pnl>0).length;
el.innerHTML=(closed.length?`<div class="meta" style="font-size:12px;color:var(--text2);margin-bottom:8px">${closed.length} closed · ${wins} winners (${Math.round(wins/closed.length*100)}%) · realized <span class="${cls(tot)}">${fmt$(tot)}</span></div>`:"")+table(tr,[
{h:"Entered",f:t=>t.entry_time},{h:"Acct",f:t=>`<span class="dot" style="background:${S[t.account]}"></span>${t.account}`},{h:"Symbol",f:t=>`<b>${t.symbol}</b>`},
{h:"Qty @ price",r:1,f:t=>`${t.qty} @ ${t.entry_price.toFixed(2)}`},{h:"Stop / target",r:1,f:t=>`${t.stop.toFixed(2)} / ${t.target.toFixed(2)}`},
{h:"Forecast",r:1,f:t=>t.forecast_5d_pct==null?"":pct(t.forecast_5d_pct/100)},{h:"Why in",f:t=>`<span style="font-size:12px">${t.entry_reason||""}</span>`},
{h:"Exit",f:t=>t.status==="closed"?`${t.exit_time}<br><span class="tag">${t.exit_reason}</span> @ ${t.exit_price.toFixed(2)}`:'<span class="tag">open</span>'},
{h:"P&L",r:1,cls:t=>t.pnl==null?"":cls(t.pnl),f:t=>t.pnl==null?"—":`${fmt$(t.pnl)} (${pct(t.pnl_pct)})`}]);})();
</script></body></html>"""


def _notes_for(notes: str, account: str) -> str:
    """Keep only this account's sections of the cycle report (sections start with '## HH:MM — name')."""
    parts = re.split(r"(?m)^(?=## )", notes)
    keep = [p for p in parts if re.match(rf"## \S+ [—-] {account}\b", p)]
    return "".join(keep) or "(no cycles for this account in the latest report)"


def render(data: dict, page: str = "all") -> str:
    data = dict(data, page=page)
    if page != "all":
        data["notes"] = _notes_for(data.get("notes", ""), page)
    c = data.get("costs", {})
    return (TEMPLATE.replace("__GENERATED__", data["generated"])
            .replace("__COST_TODAY__", f"{c.get('today', 0):.3f}")
            .replace("__COST_MONTH__", f"{c.get('month', 0):.2f}")
            .replace("__CALLS__", str(c.get("calls_month", 0)))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    s = Settings()
    data = collect_demo(s) if args.demo else collect_live(s)
    out = s.reports_dir / "dashboard.html"
    for page, fname in (("all", "dashboard.html"), ("small", "dashboard_small.html"), ("large", "dashboard_large.html")):
        (s.reports_dir / fname).write_text(render(data, page), encoding="utf-8")
        print(f"wrote {s.reports_dir / fname}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
