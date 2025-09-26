import os
import re
import json
import logging
import requests
import openai
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# --- Configuration ---
# Load API keys from the .env file
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
STEAM_API_KEY = os.getenv("STEAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ITAD_API_KEY = os.getenv("ITAD_API_KEY")

# Set up logging to see errors and bot status
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
openai.api_key = OPENAI_API_KEY

# --- Helper Functions ---

def get_steam_app_id(url: str) -> str | None:
    """Extracts the Steam App ID from a URL."""
    match = re.search(r'/app/(\d+)', url)
    return match.group(1) if match else None

def get_steam_game_details(app_id: str) -> dict | None:
    """Fetches game details from the official Steam API."""
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and data[app_id]["success"]:
            return data[app_id]["data"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from Steam API: {e}")
    return None

def get_steam_player_rating(app_id: str) -> dict | None:
    """Fetches the player rating summary from Steam's undocumented appreviews API."""
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and data.get("success") == 1:
            return data.get("query_summary")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Steam player rating: {e}")
    return None

def get_itad_deal(app_id: str) -> dict | None:
    """Fetches the best current deal from IsThereAnyDeal.com."""
    game_plain = None
    try:
        # First, get the ITAD 'plain' ID for the game
        plain_url = f"https://api.isthereanydeal.com/v02/game/plain/?key={ITAD_API_KEY}&shop=steam&game_id=app%2F{app_id}"
        response = requests.get(plain_url, timeout=10)
        response.raise_for_status()
        plain_data = response.json()
        if plain_data and plain_data.get(".meta") and plain_data[".meta"].get("active"):
             game_plain = plain_data["data"]["plain"]
    except requests.exceptions.RequestException as e:
        logger.error(f"ITAD Plain lookup failed: {e}")
        return None

    if game_plain:
        try:
            # Now, get the best price for that 'plain' ID
            prices_url = f"https://api.isthereanydeal.com/v01/game/prices/?key={ITAD_API_KEY}&plains={game_plain}&shops=steam,humble,gog,fanatical,greenmangaming,wingamestore"
            response = requests.get(prices_url, timeout=10)
            response.raise_for_status()
            price_data = response.json()["data"][game_plain]
            if price_data and price_data.get("list"):
                best_deal = price_data["list"][0]
                return {
                    "price": best_deal["price_new"],
                    "store": best_deal["shop"]["name"],
                    "cut": best_deal["price_cut"]
                }
        except requests.exceptions.RequestException as e:
            logger.error(f"ITAD Price lookup failed: {e}")
    return None

def analyze_game_with_llm(details: dict) -> dict | None:
    """Uses an LLM to summarize and reason about the specific player count."""
    game_name = details.get("name", "Unknown Game")
    description = details.get("detailed_description", "")
    categories = [cat['description'] for cat in details.get('categories', [])]

    prompt = f"""
    You are an expert game analyst. Analyze the provided game information for "{game_name}" and return a JSON object with two keys: "summary" and "players".

    Game Description: "{description}"
    API Categories: {categories}

    Instructions for your analysis:
    1.  For the "summary" key: Write a short, engaging summary of the game (1-2 sentences max).
    2.  For the "players" key:
        -   First, carefully read the game description to find the specific number of players.
        -   If a specific count is found, return that (e.g., "Up to 4 players", "8 vs 8 Multiplayer", "2-Player Co-op").
        -   If no specific number is mentioned, fall back to using the general API Categories provided (e.g., "Single-player, Online Multi-player").

    Your final output must be a valid JSON object.
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful game analyst that always responds in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=100
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}")
    return None

def format_price(details: dict) -> str:
    """Formats the price information from the Steam API."""
    if details.get("is_free", False):
        return "Free"
    if "price_overview" in details:
        return details["price_overview"]["final_formatted"]
    return "Price not available"

# --- Telegram Bot Handlers ---

async def handle_steam_link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The main function to handle messages containing Steam links."""
    message_text = update.message.text
    app_id = get_steam_app_id(message_text)
    if not app_id: return

    logger.info(f"Detected Steam link for App ID: {app_id}")
    
    # --- Gather all data in parallel ---
    game_details = get_steam_game_details(app_id)
    player_rating = get_steam_player_rating(app_id)
    itad_deal = get_itad_deal(app_id)

    if not game_details:
        await update.message.reply_text("Sorry, I couldn't fetch details for that game.", quote=True)
        return
        
    game_name = game_details.get("name", "Unknown Game")
    steam_price = format_price(game_details)

    analysis = analyze_game_with_llm(game_details)
    if not analysis:
        await update.message.reply_text("Sorry, the summary AI is having trouble right now.", quote=True)
        return

    # --- Build the final message ---
    rating_text = player_rating.get('review_score_desc', 'No rating found') if player_rating else 'No rating found'
    
    reply_parts = [
        f"**{game_name}**\n",
        f"{analysis['summary']}\n",
        f"**Players:** {analysis['players']}",
        f"**Steam Rating:** {rating_text}\n",
        f"**Price on Steam:** {steam_price}"
    ]

    if itad_deal:
        deal_text = f"**Best Deal:** **${itad_deal['price']:.2f}** (-{itad_deal['cut']}%) at {itad_deal['store']}"
        reply_parts.append(deal_text)
    
    await update.message.reply_text("\n".join(reply_parts), parse_mode='Markdown', quote=True)

# --- Main Bot Function ---

def main() -> None:
    """Starts the Telegram bot."""
    if not all([TELEGRAM_TOKEN, STEAM_API_KEY, OPENAI_API_KEY, ITAD_API_KEY]):
        logger.error("One or more API keys are missing! Check your .env file.")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_steam_link))

    logger.info("Bot is starting...")
    application.run_polling()


if __name__ == "__main__":
    main()