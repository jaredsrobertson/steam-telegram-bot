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
STEAM_API_KEY = os.getenv("STEAM_API_KEY") # Good for price checks & structured data
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # The API returns a dict where the key is the app_id
        if data and data[app_id]["success"]:
            return data[app_id]["data"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from Steam API: {e}")
    return None

def analyze_game_with_llm(details: dict) -> dict | None:
    """Uses an LLM to summarize and reason about the specific player count."""
    game_name = details.get("name", "Unknown Game")
    # Use the more detailed description for better analysis
    description = details.get("detailed_description", "")
    categories = [cat['description'] for cat in details.get('categories', [])]

    # This prompt asks the AI to reason and return a structured JSON response
    prompt = f"""
    You are an expert game analyst. Analyze the provided game information for "{game_name}" and return a JSON object with two keys: "summary" and "players".

    Game Description: "{description}"
    API Categories: {categories}

    Instructions for your analysis:
    1.  **For the "summary" key:** Write a short, engaging summary of the game (2-3 sentences max).
    2.  **For the "players" key:**
        -   First, carefully read the game description to find the **specific number of players**. Look for phrases like "up to 4 players", "8-player co-op", "you and three friends", "1v1", "solo", etc.
        -   If the description says "you and up to three friends", you must reason that this means a total of **4 players**.
        -   If a specific count is found, return that (e.g., "Up to 4 players", "8 vs 8 Multiplayer", "2-Player Co-op").
        -   If **no specific number** is mentioned in the description, fall back to using the general API Categories provided (e.g., "Single-player, Online Multi-player").

    Your final output must be a valid JSON object.
    """

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            # This forces the model to return a JSON object, which is more reliable
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful game analyst that always responds in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Lower temperature for more factual responses
            max_tokens=200
        )
        
        # Parse the JSON string from the response
        content = response.choices[0].message.content
        result = json.loads(content)

        # Basic validation to ensure the keys we need are present
        if "summary" in result and "players" in result:
            return result
        else:
            logger.error("LLM response was missing required JSON keys.")
            return None

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
    
    if not app_id:
        return # Not a steam link, do nothing

    logger.info(f"Detected Steam link for App ID: {app_id}")
    
    # 1. Get structured data from Steam API
    game_details = get_steam_game_details(app_id)
    if not game_details:
        await update.message.reply_text("Sorry, I couldn't fetch details for that game.", quote=True)
        return
        
    game_name = game_details.get("name", "Unknown Game")
    price = format_price(game_details)

    # 2. Use LLM for summary and player mode reasoning
    analysis = analyze_game_with_llm(game_details)
    if not analysis:
        await update.message.reply_text("Sorry, the summary AI is having trouble right now.", quote=True)
        return

    # 3. Format and send the final message
    reply_text = (
        f"🎮 **{game_name}** 🎮\n\n"
        f"**Summary:** {analysis['summary']}\n\n"
        f"**Players:** {analysis['players']}\n"
        f"**Price:** {price}"
    )
    
    await update.message.reply_text(reply_text, parse_mode='Markdown', quote=True)


# --- Main Bot Function ---

def main() -> None:
    """Starts the Telegram bot."""
    if not all([TELEGRAM_TOKEN, STEAM_API_KEY, OPENAI_API_KEY]):
        logger.error("One or more API keys are missing! Check your .env file.")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add a handler that listens to all non-command text messages for Steam links
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_steam_link))

    logger.info("Bot is starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
