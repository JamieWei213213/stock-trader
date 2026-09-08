"""Password-protected static server for the reports/ folder (dashboard).

Reads DASH_USER / DASH_PASS from .env. Run via the systemd unit in vps/stocktrader-dash.service:
  systemctl status stocktrader-dash
"""
import base64
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
USER = os.getenv("DASH_USER", "")
PASS = os.getenv("DASH_PASS", "")
if not USER or not PASS:
    sys.exit("Set DASH_USER and DASH_PASS in .env before starting the dashboard server.")
TOKEN = "Basic " + base64.b64encode(f"{USER}:{PASS}".encode()).decode()


class AuthHandler(SimpleHTTPRequestHandler):
    def _authed(self) -> bool:
        return self.headers.get("Authorization") == TOKEN

    def _deny(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Stock Trader"')
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Login required")

    def do_GET(self):
        if not self._authed():
            return self._deny()
        if self.path in ("/", ""):
            self.path = "/dashboard.html"
        return super().do_GET()

    def do_HEAD(self):
        if not self._authed():
            return self._deny()
        return super().do_HEAD()

    def log_message(self, fmt, *args):  # keep the journal quiet
        pass


if __name__ == "__main__":
    port = int(os.getenv("DASH_PORT", "8080"))
    handler = partial(AuthHandler, directory=str(ROOT / "reports"))
    print(f"serving reports/ on port {port} (user {USER})")
    ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()
