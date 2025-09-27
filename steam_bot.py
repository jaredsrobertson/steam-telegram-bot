import os
import re
import json
import logging
import requests
import openai
import asyncio
import signal
import sys
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

# Global shutdown flag
shutdown_event = asyncio.Event()

# --- Helper Functions ---

def get_steam_app_id(url: str) -> str | None:
    match = re.search(r'/app/(\d+)', url)
    return match.group(1) if match else None

def get_steam_game_details(app_id: str) -> dict | None:
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=US"
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

def get_rating_emoji(rating_text: str) -> str:
    """Get colored emoji based on Steam rating"""
    rating_lower = rating_text.lower()
    
    # Positive ratings (green)
    if any(word in rating_lower for word in ['overwhelmingly positive', 'very positive', 'mostly positive', 'positive']):
        return "🟢"
    
    # Mixed ratings (yellow)
    elif any(word in rating_lower for word in ['mixed', 'mostly negative']):
        return "🟡"
    
    # Negative ratings (red)
    elif any(word in rating_lower for word in ['overwhelmingly negative', 'very negative', 'negative']):
        return "🔴"
    
    # Default for unknown ratings
    else:
        return "⚪"

def get_game_genres(details: dict) -> str:
    """Extract and format game genres from Steam API data"""
    genres = details.get('genres', [])
    if genres:
        # Get the first 3 genres to keep it concise
        genre_names = [genre['description'] for genre in genres[:3]]
        return ', '.join(genre_names)
    
    # Fallback to categories if no genres available
    categories = details.get('categories', [])
    if categories:
        # Filter for genre-like categories
        genre_categories = [cat['description'] for cat in categories 
                          if any(term in cat['description'].lower() 
                                for term in ['action', 'adventure', 'strategy', 'rpg', 'simulation', 'sports', 'racing'])]
        if genre_categories:
            return ', '.join(genre_categories[:2])
    
    return "Genre not available"

def analyze_players_with_llm(details: dict) -> str | None:
    """Use LLM to analyze maximum player count"""
    game_name = details.get("name", "Unknown Game")
    categories = [cat['description'] for cat in details.get('categories', [])]
    description = details.get("detailed_description", "")
    short_description = details.get("short_description", "")

    prompt = f"""
    Analyze this game data and find the MAXIMUM number of players who can play together simultaneously.

    Game: {game_name}
    Categories: {categories}
    Description: {description[:500]}
    Short Description: {short_description}

    Look for phrases like:
    - "up to X players"
    - "play with X friends" (add 1 for the host)
    - "X-player co-op"
    - "supports X players"
    - "multiplayer for X"

    Return ONLY ONE of these formats:
    - If you find a specific number: "Up to X players" (where X is the maximum)
    - If multiplayer but no specific number: "Multiplayer"
    - If only single-player: "Single-player"

    Examples:
    - "play with up to 3 friends" → "Up to 4 players"
    - "4-player co-op" → "Up to 4 players" 
    - "supports up to 16 players" → "Up to 16 players"
    - "online multiplayer" with no number → "Multiplayer"
    - no multiplayer mentioned → "Single-player"

    Return ONLY the result, no explanation.
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting maximum player counts from game descriptions. Be precise and follow the format exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=30
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}")
    return None

def format_price(details: dict) -> str:
    if details.get("is_free", False):
        return "Free"
    if "price_overview" in details:
        return details["price_overview"]["final_formatted"]
    return "Price not available"

def get_itad_deal(app_id: str, game_name: str) -> dict | None:
    """
    Fetches the best deal using the ITAD API with correct POST method for prices.
    """
    game_id = None

    # Step 1: Look up the game ID using its Steam AppID
    try:
        lookup_url = f"https://api.isthereanydeal.com/games/lookup/v1?key={ITAD_API_KEY}&appid={app_id}"
        response = requests.get(lookup_url, timeout=10)
        response.raise_for_status()
        lookup_data = response.json()
        if lookup_data.get("found"):
            game_id = lookup_data["game"]["id"]  # Use 'id' not 'slug'
    except requests.exceptions.RequestException as e:
        logger.error(f"ITAD AppID lookup failed: {e}")

    # Step 2: Fallback to searching by title if AppID lookup fails
    if not game_id:
        logger.info(f"ITAD AppID lookup failed. Trying fallback search with title: '{game_name}'")
        try:
            lookup_url = f"https://api.isthereanydeal.com/games/lookup/v1?key={ITAD_API_KEY}&title={requests.utils.quote(game_name)}"
            response = requests.get(lookup_url, timeout=10)
            response.raise_for_status()
            lookup_data = response.json()
            if lookup_data.get("found"):
                game_id = lookup_data["game"]["id"]  # Use 'id' not 'slug'
        except requests.exceptions.RequestException as e:
            logger.error(f"ITAD Title search fallback failed: {e}")
            return None

    # Step 3: If we have a game ID, get the best prices using correct POST request
    if game_id:
        try:
            # Force USD currency by adding country parameter
            prices_url = f"https://api.isthereanydeal.com/games/prices/v2?key={ITAD_API_KEY}&country=US"
            
            # POST request body should be a JSON array of game IDs
            payload = [game_id]
            
            response = requests.post(prices_url, json=payload, timeout=10)
            response.raise_for_status()
            
            price_data = response.json()
            if price_data and isinstance(price_data, list) and len(price_data) > 0:
                game_data = price_data[0]
                if game_data.get("deals") and len(game_data["deals"]) > 0:
                    best_deal = game_data["deals"][0]
                    return {
                        "price": best_deal["price"]["amount"],
                        "store": best_deal["shop"]["name"],
                        "cut": best_deal["cut"],
                        "url": best_deal.get("url", "")  # Add the purchase URL
                    }
        except (requests.exceptions.RequestException, IndexError, KeyError) as e:
            logger.error(f"ITAD Price lookup failed for game ID '{game_id}': {e}")
    
    logger.warning(f"Could not find any ITAD data for '{game_name}'")
    return None

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
    
    # Get genre from Steam data and player info from LLM
    game_genre = get_game_genres(game_details)
    player_analysis = analyze_players_with_llm(game_details)

    if not player_analysis:
        await update.message.reply_text("Sorry, the player analysis AI is having trouble right now.", quote=True)
        return

    steam_price = format_price(game_details)
    rating_text = player_rating.get('review_score_desc', 'No rating found') if player_rating else 'No rating found'
    rating_emoji = get_rating_emoji(rating_text)
    
    # Create Steam store URL
    steam_url = f"https://store.steampowered.com/app/{app_id}/"
    
    reply_parts = [
        f"\n",
        f"<b>{game_name}</b>",
        f"{rating_emoji} <i>{rating_text}</i>",
        f"🏷️ {game_genre}\n",
        f"👥 <b>{player_analysis}</b>\n",
        f"💰 <b>Steam:</b> <a href='{steam_url}'>{steam_price}🔗</a>"
    ]

    if itad_deal:
        deal_url = itad_deal.get('url', '')
        if deal_url:
            # Make the best deal price a clickable link
            deal_text = f"🔥 <b>Best Deal:</b> <a href='{deal_url}'><b>${itad_deal['price']:.2f}</b> (-{itad_deal['cut']}%) at {itad_deal['store']}🔗</a>"
        else:
            # Fallback if no URL is provided
            deal_text = f"🔥 <b>Best Deal:</b> <b>${itad_deal['price']:.2f}</b> (-{itad_deal['cut']}%) at {itad_deal['store']}"
        reply_parts.append(deal_text)
    
    await update.message.reply_text(
        "\n".join(reply_parts), 
        parse_mode='HTML', 
        quote=True, 
        disable_web_page_preview=True
    )

# --- Shutdown Handlers ---

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()

async def shutdown_application(application):
    """Gracefully shutdown the application"""
    logger.info("Shutting down application...")
    try:
        # Stop the updater first (this stops polling)
        if application.updater.running:
            await application.updater.stop()
            logger.info("Updater stopped")
        
        # Then stop the application
        await application.stop()
        logger.info("Application stopped")
        
        # Finally shutdown
        await application.shutdown()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        # Force shutdown if graceful shutdown fails
        try:
            await application.updater.stop()
            await application.stop()
            await application.shutdown()
        except:
            pass

# --- Main Bot Function ---

async def main():
    if not all([TELEGRAM_TOKEN, STEAM_API_KEY, OPENAI_API_KEY, ITAD_API_KEY]):
        logger.error("One or more API keys are missing! Check your .env file.")
        return

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Build application with proper timeout settings
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .get_updates_read_timeout(30)      # 30 second read timeout
        .get_updates_write_timeout(30)     # 30 second write timeout
        .get_updates_connect_timeout(30)   # 30 second connect timeout
        .read_timeout(30)                  # General read timeout
        .write_timeout(30)                 # General write timeout
        .build()
    )
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_steam_link))

    logger.info("Bot is starting...")
    
    try:
        # Initialize and start the bot
        await application.initialize()
        await application.start()
        
        # Clear any pending updates and start polling
        await application.updater.start_polling(
            drop_pending_updates=True,
            poll_interval=1.0,
            timeout=10
        )
        
        logger.info("Bot started successfully")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Bot encountered an error: {e}")
    finally:
        # Graceful shutdown
        await shutdown_application(application)

if __name__ == "__main__":
    # Clear any existing Telegram sessions first
    try:
        import requests
        # Delete webhook if set and clear pending updates
        webhook_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook"
        requests.get(webhook_url, timeout=5)
        
        updates_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?offset=-1&timeout=1"
        requests.get(updates_url, timeout=5)
        
        logger.info("Cleared Telegram session")
    except Exception as e:
        logger.warning(f"Could not clear Telegram session: {e}")
    
    # Run the bot
    asyncio.run(main())