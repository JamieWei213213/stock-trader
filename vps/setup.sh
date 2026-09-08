#!/usr/bin/env bash
# One-time VPS setup (Ubuntu 22.04/24.04). Run as root or with sudo:
#   bash vps/setup.sh
# Assumes the project folder was copied to /opt/stocktrader (see README).
set -euo pipefail
APP=/opt/stocktrader

apt-get update -y
apt-get install -y python3 python3-venv python3-pip git
timedatectl set-timezone America/Los_Angeles

cd "$APP"
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
mkdir -p state reports logs

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">>> Edit $APP/.env and add your API keys, then run: .venv/bin/python test_connection.py"
fi

# install cron schedule (see vps/crontab.txt)
crontab -l 2>/dev/null | grep -v stocktrader > /tmp/cron.keep || true
cat /tmp/cron.keep vps/crontab.txt | crontab -
echo ">>> cron installed:"; crontab -l

# password-protected dashboard server (needs DASH_USER / DASH_PASS in .env)
cp vps/stocktrader-dash.service /etc/systemd/system/stocktrader-dash.service
systemctl daemon-reload
systemctl enable --now stocktrader-dash
echo ">>> dashboard service:"; systemctl --no-pager status stocktrader-dash | head -5
