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
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
STEAM_API_KEY = os.getenv("STEAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ITAD_API_KEY = os.getenv("ITAD_API_KEY")

# --- Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
openai.api_key = OPENAI_API_KEY

# --- Helper Functions ---

def get_steam_app_id(url: str) -> str | None:
    match = re.search(r'/app/(\d+)', url)
    return match.group(1) if match else None

def get_steam_game_details(app_id: str) -> dict | None:
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

def get_itad_deal(app_id: str, game_name: str) -> dict | None:
    """
    Fetches the best deal using the ITAD API with correct GET method for prices.
    """
    game_slug = None

    # Step 1: Look up the game 'slug' using its Steam AppID
    try:
        lookup_url = f"https://api.isthereanydeal.com/games/lookup/v1?key={ITAD_API_KEY}&appid={app_id}"
        response = requests.get(lookup_url, timeout=10)
        response.raise_for_status()
        lookup_data = response.json()
        if lookup_data.get("found"):
            game_slug = lookup_data["game"]["slug"]
    except requests.exceptions.RequestException as e:
        logger.error(f"ITAD AppID lookup failed: {e}")

    # Step 2: Fallback to searching by title if AppID lookup fails
    if not game_slug:
        logger.info(f"ITAD AppID lookup failed. Trying fallback search with title: '{game_name}'")
        try:
            lookup_url = f"https://api.isthereanydeal.com/games/lookup/v1?key={ITAD_API_KEY}&title={requests.utils.quote(game_name)}"
            response = requests.get(lookup_url, timeout=10)
            response.raise_for_status()
            lookup_data = response.json()
            if lookup_data.get("found"):
                game_slug = lookup_data["game"]["slug"]
        except requests.exceptions.RequestException as e:
            logger.error(f"ITAD Title search fallback failed: {e}")
            return None

    # Step 3: If we have a slug, get the best prices using a GET request
    if game_slug:
        try:
            # Build the URL with query parameters for GET request
            shops = ["steam", "humble", "gog", "fanatical", "greenmangaming", "wingamestore"]
            shops_param = "&".join([f"shops[]={shop}" for shop in shops])
            prices_url = f"https://api.isthereanydeal.com/games/prices/v2?key={ITAD_API_KEY}&id={game_slug}&{shops_param}"
            
            # Use GET request instead of POST
            response = requests.get(prices_url, timeout=10)
            response.raise_for_status()
            
            price_data = response.json()
            # The response structure might be different for GET vs POST
            if price_data and isinstance(price_data, list) and len(price_data) > 0:
                game_data = price_data[0]
                if game_data.get("deals"):
                    best_deal = game_data["deals"][0]
                    return {
                        "price": best_deal["price"]["amount"],
                        "store": best_deal["shop"]["name"],
                        "cut": best_deal["cut"]
                    }
            elif price_data and price_data.get("deals"):
                # Alternative response structure
                best_deal = price_data["deals"][0]
                return {
                    "price": best_deal["price"]["amount"],
                    "store": best_deal["shop"]["name"],
                    "cut": best_deal["cut"]
                }
        except (requests.exceptions.RequestException, IndexError, KeyError) as e:
            logger.error(f"ITAD Price lookup failed for slug '{game_slug}': {e}")
    
    logger.warning(f"Could not find any ITAD data for '{game_name}'")
    return None

def analyze_game_with_llm(details: dict) -> dict | None:
    game_name = details.get("name", "Unknown Game")
    description = details.get("detailed_description", "")
    categories = [cat['description'] for cat in details.get('categories', [])]

    prompt = f"""
    You are an expert game analyst. Analyze the provided game information for "{game_name}" and return a JSON object with two keys: "summary" and "players".
    Instructions:
    1.  For "summary": Write a short, engaging summary (1-2 sentences max). Do not state the game title in the summary.
    2.  For "players": Find the specific number of players (e.g., "Up to 4 players"). If none is mentioned, use general categories (e.g., "Single-player, Online Co-op").
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
    if details.get("is_free", False):
        return "Free"
    if "price_overview" in details:
        return details["price_overview"]["final_formatted"]
    return "Price not available"

# --- Main Telegram Handler ---

async def handle_steam_link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_text = update.message.text
    app_id = get_steam_app_id(message_text)
    if not app_id: return

    logger.info(f"Detected Steam link for App ID: {app_id}")
    
    game_details = get_steam_game_details(app_id)
    if not game_details:
        await update.message.reply_text("Sorry, I couldn't fetch details for that game.", quote=True)
        return
        
    game_name = game_details.get("name", "Unknown Game")

    player_rating = get_steam_player_rating(app_id)
    itad_deal = get_itad_deal(app_id, game_name)
    analysis = analyze_game_with_llm(game_details)

    if not analysis:
        await update.message.reply_text("Sorry, the summary AI is having trouble right now.", quote=True)
        return

    steam_price = format_price(game_details)
    rating_text = player_rating.get('review_score_desc', 'No rating found') if player_rating else 'No rating found'
    
    reply_parts = [
        f"**{game_name}**\n",
        f"{rating_text}\n",
        f"{analysis['summary']}\n",
        f"**Players:** {analysis['players']}\n",
        f"**Price on Steam:** {steam_price}"
    ]

    if itad_deal:
        deal_text = f"**Best Deal:** **${itad_deal['price']:.2f}** (-{itad_deal['cut']}%) at {itad_deal['store']}"
        reply_parts.append(deal_text)
    
    await update.message.reply_text("\n".join(reply_parts), parse_mode='Markdown', quote=True)

# --- Main Bot Function ---

def main() -> None:
    if not all([TELEGRAM_TOKEN, STEAM_API_KEY, OPENAI_API_KEY, ITAD_API_KEY]):
        logger.error("One or more API keys are missing! Check your .env file.")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_steam_link))

    logger.info("Bot is starting...")
    application.run_polling()

if __name__ == "__main__":
    main()