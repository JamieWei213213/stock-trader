"""Kill-switch control.   python killswitch.py            # status of both accounts
                          python killswitch.py --reset large   # clear a drawdown trip (after you've looked)
                          python killswitch.py --reset large --peak   # ...and restart the peak from current equity
"""
import argparse

from trader.killswitch import KillSwitch
from trader.settings import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", choices=["small", "large"])
    ap.add_argument("--peak", action="store_true", help="with --reset: also reset the equity peak to 0 (re-based next cycle)")
    args = ap.parse_args()
    s = Settings()
    if args.reset:
        k = KillSwitch(s.state_dir, args.reset, s.cfg.get("killswitch", {}))
        k.reset()
        if args.peak:
            k.st["peak"] = 0; k._save()
        print(f"{args.reset}: kill switch cleared" + (" and peak re-based" if args.peak else ""))
    for name in ("small", "large"):
        st = KillSwitch(s.state_dir, name, s.cfg.get("killswitch", {})).status()
        print(f"{name:5s} tripped={st.get('tripped')} peak=${float(st.get('peak', 0) or 0):,.0f} day_start=${float(st.get('day_start', 0) or 0):,.0f} {st.get('reason') or ''}")


if __name__ == "__main__":
    main()
