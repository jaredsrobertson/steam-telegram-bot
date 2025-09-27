#!/bin/bash
echo "Stopping existing bot instances..."
pkill -f steam_bot
sleep 3
echo "Starting bot..."
python steam_bot.py