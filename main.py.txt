# main.py
"""
Chatterous - Discord bot (stable)

Includes:
- Discord bot with slash + prefix commands
- Optional Hugging Face integration (chat completions)
- Flask heartbeat bound to platform PORT env
- Global exception handling and graceful shutdown
- Trivia, RPS, Poll commands
- Auto-react and auto-reply (configurable via env)
- /help and !help commands

Environment variables (examples):
- DISCORD_BOT_TOKEN (required)
- HUGGINGFACE_API_KEY (optional)
- PORT (optional; used by hosting platforms)
- AUTO_REACT_CHANNELS (comma-separated IDs)
- AUTO_REACT_EMOJIS (comma-separated emojis)
- AUTO_REACT_KEYWORDS (comma-separated)
- AUTO_REACT_COOLDOWN
- AUTO_REPLY_CHANNELS, AUTO_REPLY_KEYWORDS, AUTO_REPLY_CHANCE, AUTO_REPLY_COOLDOWN

Install requirements: create a requirements.txt with at least:
- discord.py>=2.0.0
- Flask
- huggingface_hub (optional)

Copy this file into your project and set the env vars in your host.
"""

import os
import sys
import signal
import threading
import asyncio
import logging
import random
import time
from typing import Tuple, Optional

from flask import Flask, jsonify
import discord
from discord import app_commands
from discord.ext import commands

# Optional HF import
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# --------------------
# Configuration
# --------------------
MAX_RESPONSE_LENGTH = 1900
HF_TIMEOUT_SECONDS = 25
DEFAULT_FLASK_PORT = 5000

# --------------------
# Logging & globals
# --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatterous")

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
HF_KEY = os.environ.get("HUGGINGFACE_API_KEY")

if not DISCORD_TOKEN:
    logger.critical("DISCORD_BOT_TOKEN is missing in environment. Exiting.")
    sys.exit(1)

if not HF_KEY:
    logger.info("HUGGINGFACE_API_KEY not provided — HF features disabled.")

# Initialize HF client if available
hf_client = None
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
if HF_KEY and InferenceClient:
    try:
        hf_client = InferenceClient(token=HF_KEY)
        logger.info("Hugging Face client initialized.")
    except Exception as e:
        logger.exception("Failed to initialize Hugging Face client: %s", e)
        hf_client = None
else:
    if not InferenceClient:
        logger.warning("huggingface_hub not installed or failed to import. HF disabled.")

# --------------------
# Global exception & signal handling
# --------------------

def _handle_unhandled_exception(exc_type, exc, tb):
    logger.error("Uncaught exception", exc_info=(exc_type, exc, tb))

sys.excepthook = _handle_unhandled_exception


def _asyncio_exception_handler(loop, context):
    logger.error("Asyncio unhandled exception: %s", context)

try:
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_asyncio_exception_handler)
except RuntimeError:
    loop = None


def _graceful_shutdown(signum, frame):
    logger.info("Signal %s received, shutting down...", signum)
    try:
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)

# --------------------
# Discord bot setup
# --------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

@bot.event
async def on_ready():
    logger.info("Logged in as %s (id: %s)", bot.user, bot.user.id)
    try:
        synced = await bot.tree.sync()
        logger.info("Synced %d slash commands", len(synced))
    except Exception as e:
        logger.error("Failed to sync commands: %s", e)

    activity = discord.Game("/ask to chat with me!")
    try:
        await bot.change_presence(status=discord.Status.online, activity=activity)
    except Exception as e:
        logger.warning("Failed to set presence: %s", e)

# --------------------
# Hugging Face helpers
# --------------------

def query_huggingface_sync(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    if not hf_client:
        return None, "HF API key not configured or client unavailable"
    try:
        messages = [
            {"role": "system", "content": (
                "You are a friendly, helpful AI assistant. Keep your responses conversational, warm, "
                "and natural - like talking to a friend. Use emojis naturally. Be concise but personable."
            )},
            {"role": "user", "content": prompt}
        ]
        response = hf_client.chat_completion(
            messages=messages,
            model=HF_MODEL,
            max_tokens=400,
            temperature=0.8
        )
        text = None
        if hasattr(response, "choices") and response.choices:
            try:
                # handle possible shapes
                msg = response.choices[0].message
                if isinstance(msg, dict):
                    text = msg.get("content")
                else:
                    text = getattr(msg, "content", None)
            except Exception:
                text = None
        if not text:
            text = getattr(response, "generated_text", None) or str(response)
        return text, None
    except Exception as e:
        logger.exception("HF API call failed:")
        return None, f"HF error: {e}"


async def query_huggingface(prompt: str, timeout: int = HF_TIMEOUT_SECONDS) -> Tuple[Optional[str], Optional[str]]:
    loop = asyncio.get_event_loop()
    fut = loop.run_in_executor(None, lambda: query_huggingface_sync(prompt))
    try:
        result = await asyncio.wait_for(fut, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.error("Hugging Face call timed out after %s seconds", timeout)
        return None, "HF timeout"
    except Exception as e:
        logger.exception("Exception while calling HF in executor")
        return None, f"HF executor error: {e}"

# --------------------
# Commands: ask (HF), trivia, rps, poll
# --------------------
@bot.tree.command(name="ask", description="Chat with the AI assistant")
@app_commands.describe(question="What would you like to ask?")
async def ask_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    text, error = await query_huggingface(question)
    if text:
        out = text.strip()
        if len(out) > MAX_RESPONSE_LENGTH:
            out = out[:MAX_RESPONSE_LENGTH] + "..."
        await interaction.followup.send(f"✨ {out}")
    else:
        logger.info("AI failed (slash): %s", error)
        await interaction.followup.send("❌ Oops! I'm having trouble thinking right now. Try again in a moment!")

@bot.command(name="ask")
async def ask_command(ctx, *, question: str = ""):
    if not question:
        await ctx.send("💭 What's on your mind? Try `/ask your question here`")
        return
    thinking = await ctx.send("🤖 Thinking...")
    text, error = await query_huggingface(question)
    if text:
        out = text.strip()
        if len(out) > MAX_RESPONSE_LENGTH:
            out = out[:MAX_RESPONSE_LENGTH] + "..."
        await thinking.edit(content=f"✨ {out}")
    else:
        logger.info("AI failed (prefix): %s", error)
        await thinking.edit(content="❌ Oops! I'm having trouble thinking right now. Try again in a moment!")

# Trivia
TRIVIA_QUESTIONS = [
    {"q": "What is the capital of France?", "a": "paris"},
    {"q": "Which planet is known as the Red Planet?", "a": "mars"},
    {"q": "Who wrote 'Hamlet'?", "a": "william shakespeare"},
    {"q": "What is 9 * 9?", "a": "81"},
]

trivia_scores = {}
active_trivia = {}

@bot.tree.command(name="trivia", description="Start a trivia question")
async def trivia_slash(interaction: discord.Interaction):
    q = random.choice(TRIVIA_QUESTIONS)
    active_trivia[interaction.channel_id] = (q["a"].lower(), interaction.user.id)
    await interaction.response.send_message(f"🧠 **Trivia:** {q['q']} \nReply in chat with your answer!")

@bot.command(name="trivia")
async def trivia_cmd(ctx):
    q = random.choice(TRIVIA_QUESTIONS)
    active_trivia[ctx.channel.id] = (q["a"].lower(), ctx.author.id)
    await ctx.send(f"🧠 **Trivia:** {q['q']} \nReply in chat with your answer!")

# RPS
@bot.tree.command(name="rps", description="Play rock-paper-scissors")
@app_commands.describe(choice="Your choice: rock, paper, or scissors")
async def rps_slash(interaction: discord.Interaction, choice: str):
    choice = choice.lower()
    options = ["rock", "paper", "scissors"]
    if choice not in options:
        await interaction.response.send_message("Choose rock, paper, or scissors.")
        return
    bot_choice = random.choice(options)
    if choice == bot_choice:
        res = "It's a tie!"
    elif (choice == "rock" and bot_choice == "scissors") or \
         (choice == "paper" and bot_choice == "rock") or \
         (choice == "scissors" and bot_choice == "paper"):
        res = "You win! 🎉"
    else:
        res = "I win! 😈"
    await interaction.response.send_message(f"You chose **{choice}**. I chose **{bot_choice}**. {res}")

@bot.command(name="rps")
async def rps_cmd(ctx, choice: str = ""):
    if not choice:
        await ctx.send("Usage: `!rps <rock|paper|scissors>`")
        return
    choice = choice.lower()
    options = ["rock", "paper", "scissors"]
    if choice not in options:
        await ctx.send("Choose rock, paper, or scissors.")
        return
    bot_choice = random.choice(options)
    if choice == bot_choice:
        res = "It's a tie!"
    elif (choice == "rock" and bot_choice == "scissors") or \
         (choice == "paper" and bot_choice == "rock") or \
         (choice == "scissors" and bot_choice == "paper"):
        res = "You win! 🎉"
    else:
        res = "I win! 😈"
    await ctx.send(f"You chose **{choice}**. I chose **{bot_choice}**. {res}")

# Poll (slash only)
NUMBER_EMOJIS = [1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

@bot.tree.command(name="poll", description="Create a quick poll (up to 5 options)")
@app_commands.describe(question="Poll question", opts="Comma-separated options (max 5)", duration="Duration in seconds (default 30)")
async def poll_slash(interaction: discord.Interaction, question: str, opts: str, duration: int = 30):
    options = [o.strip() for o in opts.split(",") if o.strip()]
    if len(options) < 2 or len(options) > 5:
        await interaction.response.send_message("Provide between 2 and 5 comma-separated options.")
        return
    embed = discord.Embed(title="📊 " + question, description="\n".join(f"{NUMBER_EMOJIS[i]} {opt}" for i,opt in enumerate(options)))
    # send initial response and fetch the message object
    await interaction.response.send_message(embed=embed)
    try:
        sent_msg = await interaction.original_response()
    except Exception:
        # fallback: send as followup
        sent_msg = await interaction.followup.send(embed=embed)

    for i in range(len(options)):
        try:
            await sent_msg.add_reaction(NUMBER_EMOJIS[i])
            await asyncio.sleep(0.2)
        except Exception:
            logger.exception("Failed to add poll reaction")
    # wait for duration
    await asyncio.sleep(duration)
    try:
        sent_msg = await sent_msg.channel.fetch_message(sent_msg.id)
    except Exception:
        logger.exception("Failed to fetch poll message for tallying")
        return
    counts = []
    for i in range(len(options)):
        emoji = NUMBER_EMOJIS[i]
        react = discord.utils.get(sent_msg.reactions, emoji=emoji)
        counts.append((options[i], (react.count - 1) if react else 0))
    results = "\n".join(f"**{opt}** — {c} vote(s)" for opt,c in counts)
    await sent_msg.channel.send(f"🗳️ Poll finished! Results:\n{results}")

# --------------------
# Auto-react and Auto-reply configuration
# --------------------
AUTO_REACT_CHANNELS = os.environ.get("AUTO_REACT_CHANNELS", "")
AUTO_REACT_CHANNEL_IDS = [int(x) for x in AUTO_REACT_CHANNELS.split(",") if x.strip().isdigit()]
AUTO_REACT_EMOJIS = [e.strip() for e in os.environ.get("AUTO_REACT_EMOJIS", "👍,🤖,🔥").split(",") if e.strip()]
AUTO_REACT_KEYWORDS = [k.strip().lower() for k in os.environ.get("AUTO_REACT_KEYWORDS", "").split(",") if k.strip()]
AUTO_REACT_COOLDOWN = int(os.environ.get("AUTO_REACT_COOLDOWN", "10"))

AUTO_REPLY_CHANNELS = os.environ.get("AUTO_REPLY_CHANNELS", "")
AUTO_REPLY_CHANNEL_IDS = [int(x) for x in AUTO_REPLY_CHANNELS.split(",") if x.strip().isdigit()]
AUTO_REPLY_KEYWORDS = [k.strip().lower() for k in os.environ.get("AUTO_REPLY_KEYWORDS", "").split(",") if k.strip()]
AUTO_REPLY_CHANCE = int(os.environ.get("AUTO_REPLY_CHANCE", "20"))
AUTO_REPLY_COOLDOWN = int(os.environ.get("AUTO_REPLY_COOLDOWN", "30"))

FUN_REPLIES = [
    "Lol true! 😂",
    "That’s epic! 🔥",
    "I feel that. 🤝",
    "Wow, tell me more! 👀",
    "Haha, I can't stop laughing 🤣",
    "I'm just a bot, but that made my circuits happy. 🤖💖",
    "Emoji party! 🎉",
]

_last_react_time = {}
_last_reply_time = {}

async def try_add_reactions(message: discord.Message):
    for emoji in AUTO_REACT_EMOJIS:
        try:
            await message.add_reaction(emoji)
            await asyncio.sleep(0.25)
        except discord.Forbidden:
            logger.warning("Missing permission to add reactions in channel %s", message.channel.id)
            return
        except discord.HTTPException as e:
            logger.debug("Failed to add reaction %s: %s", emoji, e)
        except Exception:
            logger.exception("Unexpected error while reacting")

async def try_send_auto_reply(message: discord.Message):
    reply_text = random.choice(FUN_REPLIES)
    try:
        await message.channel.send(f"{message.author.mention} {reply_text}")
    except discord.Forbidden:
        logger.warning("Missing permission to send messages in channel %s", message.channel.id)
    except Exception:
        logger.exception("Failed to send auto-reply")

# --------------------
# on_message: integrates trivia check, auto-react, auto-reply, and command processing
# --------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Trivia answer handling
    try:
        data = active_trivia.get(message.channel.id)
        if data:
            answer, asked_by = data
            if message.content.strip().lower() == answer:
                uid = message.author.id
                trivia_scores[uid] = trivia_scores.get(uid, 0) + 1
                await message.channel.send(f"✅ {message.author.mention} — Correct! +1 point. Total: {trivia_scores[uid]}")
                del active_trivia[message.channel.id]
                await bot.process_commands(message)
                return
    except Exception:
        logger.exception("Error in trivia on_message handling")

    now = time.time()
    # Auto-react
    try:
        if AUTO_REACT_CHANNEL_IDS and message.channel.id in AUTO_REACT_CHANNEL_IDS:
            if AUTO_REACT_KEYWORDS:
                if not any(kw in message.content.lower() for kw in AUTO_REACT_KEYWORDS):
                    pass
                else:
                    key = (message.author.id, message.channel.id)
                    last = _last_react_time.get(key, 0)
                    if now - last >= AUTO_REACT_COOLDOWN:
                        _last_react_time[key] = now
                        await try_add_reactions(message)
            else:
                key = (message.author.id, message.channel.id)
                last = _last_react_time.get(key, 0)
                if now - last >= AUTO_REACT_COOLDOWN:
                    _last_react_time[key] = now
                    await try_add_reactions(message)
    except Exception:
        logger.exception("Auto-react failed")

    # Auto-reply
    try:
        if AUTO_REPLY_CHANNEL_IDS and message.channel.id in AUTO_REPLY_CHANNEL_IDS:
            if AUTO_REPLY_KEYWORDS and not any(kw in message.content.lower() for kw in AUTO_REPLY_KEYWORDS):
                pass
            else:
                key = (message.author.id, message.channel.id)
                last = _last_reply_time.get(key, 0)
                if now - last >= AUTO_REPLY_COOLDOWN:
                    roll = random.randint(1, 100)
                    if roll <= AUTO_REPLY_CHANCE:
                        _last_reply_time[key] = now
                        await try_send_auto_reply(message)
    except Exception:
        logger.exception("Auto-reply failed")

    await bot.process_commands(message)

# --------------------
# Help command (slash + prefix)
# --------------------
HELP_TEXT = (
    "**Chatterous Bot — Help**\n"
    "\nCommands:\n"
    "/ask or !ask <question> — Ask the AI (requires HUGGINGFACE_API_KEY).\n"
    "/trivia or !trivia — Start a trivia question.\n"
    "/rps or !rps <rock|paper|scissors> — Play rock-paper-scissors.\n"
    "/poll <question> <opts> — Create a poll (slash-only).\n"
    "/help or !help — Show this help message.\n"
    "\nAuto features:\n"
    "Auto-react and Auto-reply can be configured via env vars.\n"
    "Required environment variables: DISCORD_BOT_TOKEN. Optional: HUGGINGFACE_API_KEY.\n"
    "Set AUTO_REACT_CHANNELS (comma-separated channel IDs) to enable auto reactions.\n"
    "Set AUTO_REPLY_CHANNELS to enable fun auto-replies.\n"
    "\nPermissions your bot needs in server: Send Messages, Add Reactions, Read Messages/View Channel.\n"
    "\nDeployment notes to avoid 502/503:\n"
    "- Ensure your Flask app listens on the platform-provided PORT environment variable.\n"
    "- Provide the DISCORD_BOT_TOKEN env var (bot will exit if missing).\n"
    "- Use a stable host (Railway/Render/VPS) for best uptime — free hosts may sleep.\n"
)

@bot.tree.command(name="help", description="Show help and setup info")
async def help_slash(interaction: discord.Interaction):
    await interaction.response.send_message(HELP_TEXT)

@bot.command(name="help")
async def help_cmd(ctx):
    await ctx.send(HELP_TEXT)

# --------------------
# Flask heartbeat
# --------------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "Chatterous",
        "message": "Bot is running! 🤖✨"
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "uptime": "running"})


def run_flask():
    port = int(os.environ.get("PORT", DEFAULT_FLASK_PORT))
    logger.info("Starting Flask on 0.0.0.0:%s", port)
    app.run(host="0.0.0.0", port=port, threaded=True)

# --------------------
# Entrypoint
# --------------------
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask heartbeat server started (background thread)")

    try:
        logger.info("Starting Discord bot...")
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.exception("Bot crashed on run(): %s", e)
        sys.exit(1)