# Steam Intelligence Bot 🤖

A robust Telegram bot that enriches Steam store links with real-time analytics. It automatically detects Steam URLs in chat, fetches current pricing deals, uses GPT-4 to analyze multiplayer capabilities, and provides "gray market" price comparisons.

**Built for reliability and ease of use.**

## ✨ Features

* **Smart Detection:** Automatically recognizes Steam store links—no commands needed.
* **AI-Powered Analysis:** Uses OpenAI (GPT-4) to parse game descriptions and determine accurate player counts (e.g., "Up to 4 players" vs "Single-player").
* **Deal Hunting:** Cross-references prices with the **IsThereAnyDeal API** to find historical lows and current sales.
* **Visual Ratings:** Converts Steam review scores into instant visual indicators (🟢/🟡/🔴).
* **Gray Market Comparison:** Provides quick search links for G2A and Loaded for alternative pricing.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Core Lib:** `python-telegram-bot` (Async V20+)
* **Integrations:**
    * Steam Web API (Game Data)
    * OpenAI API (Player Count Analysis)
    * IsThereAnyDeal API (Price Comparison)
* **DevOps:** Environment variable configuration (`python-dotenv`) & structured logging.

## 🚀 Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/jaredsrobertson/steam-telegram-bot.git](https://github.com/jaredsrobertson/steam-telegram-bot.git)
    cd steam-telegram-bot
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**
    Create a `.env` file in the root directory. You will need API keys for the services used:
    ```ini
    TELEGRAM_TOKEN=your_telegram_bot_token
    STEAM_API_KEY=your_steam_web_api_key
    OPENAI_API_KEY=your_openai_key
    ITAD_API_KEY=your_isthereanydeal_key
    ```

4.  **Run the Bot**
    ```bash
    python steam_bot.py
    ```

## 🔍 How it Works

1.  **Event Loop:** The bot utilizes an asynchronous event loop to poll Telegram for updates without blocking.
2.  **Regex Matching:** Incoming messages are scanned for Steam App IDs.
3.  **Parallel Fetching:** Once a link is found, the bot queries Steam for metadata and IsThereAnyDeal for pricing.
4.  **LLM Processing:** The game description is sent to OpenAI with a strict system prompt to extract multiplayer details.
5.  **Response:** The data is aggregated into a clean HTML-formatted message and replied to the user.

## 🛡️ License

MIT License
