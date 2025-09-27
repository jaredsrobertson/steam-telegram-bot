#!/bin/bash
echo "Stopping existing bot instances..."
pkill -f steam_bot || true
sleep 20
echo "Starting bot..."
python steam_bot.py