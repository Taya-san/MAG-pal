# config.py
# Loads all settings from the .env file and validates them.
# Every config value has a default so the bot won't crash
# if something is missing from .env — except the 3 required fields.

import os
from dotenv import load_dotenv

# Load variables from .env into os.environ
# This runs once when the module is imported
load_dotenv()


class Config:
    # Holds all bot configuration in one place
    # Access values like: config.DISCORD_TOKEN, config.PAL_NAME, etc.

    def __init__(self):
        # === REQUIRED: bot will crash at startup if these are missing ===

        # Discord bot token from https://discord.com/developers/applications
        self.DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")

        # OpenRouter API key from https://openrouter.ai/keys
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

        # Your Discord user ID (right-click your name → Copy ID)
        # Only this user's messages will be processed
        raw_id = os.getenv("OWNER_ID", "0")
        self.OWNER_ID = int(raw_id) if raw_id.strip() else 0

        # === PERSONALITY: controls how the AI talks ===

        # The bot's name — used in prompts and detection
        self.PAL_NAME = os.getenv("PAL_NAME", "MAG")

        # What the AI calls you in its system prompt
        self.USER_NAME = os.getenv("USER_NAME", "User")

        # Describe the bot's personality (e.g. "chill, sarcastic, supportive")
        self.PERSONALITY = os.getenv("PERSONALITY", "chill, curious, supportive")

        # Topics you're interested in (helps the AI know what you like)
        self.INTERESTS = os.getenv("INTERESTS", "")

        self.MAG_LANG = os.getenv("MAG_LANG", "")

        # How long responses should be: "short", "medium", "long", or "1-2 sentences"
        self.RESPONSE_LENGTH = os.getenv("RESPONSE_LENGTH", "short")

        # === AI MODEL: which LLM to use via OpenRouter ===

        # Model identifier on OpenRouter
        # "openrouter/free" auto-routes to best free model
        # Other examples: "deepseek/deepseek-r1:free", "meta-llama/llama-3.3-70b-instruct:free"
        self.MODEL = os.getenv("MODEL", "openrouter/free")

        # Max tokens the AI can generate per response
        # Higher = longer responses but costs more
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))

        # Creativity: 0.0 = very predictable, 1.0 = very random
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

        # === BEHAVIOR: how the bot acts ===

        # Enable verbose logging to palbot.log
        # Shows every prompt, AI response, latency, etc.
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # Summarize old conversations for long-term memory.
        # Every SUMMARY_INTERVAL messages, the AI generates a summary
        # of what was discussed and stores it in the session.
        # The summary is included in future prompts for context.
        self.SUMMARIZE = os.getenv("SUMMARIZE", "false").lower() == "true"
        self.SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "100"))

        # Where to store the SQLite database file
        self.DB_PATH = os.getenv("DB_PATH", "palbot.db")

        # How many minutes of silence before a continuation expires
        # If you talk again within this window, the bot treats it
        # as continuing the conversation
        self.HEURISTIC_CONTINUATION_MINUTES = int(
            os.getenv("HEURISTIC_CONTINUATION_MINUTES", "2")
        )

        raw_names = os.getenv("OWNER_NAMES", "")
        self.OWNER_NAMES = {}
        if raw_names:
            for pair in raw_names.split(","):
                pair = pair.strip()
                if ":" in pair:
                    uid, name = pair.split(":", 1)
                    self.OWNER_NAMES[int(uid.strip())] = name.strip()
        self.NAME_TO_OWNER = {v: k for k, v in self.OWNER_NAMES.items()}

    def validate(self):
        # Called at startup. If required config is missing,
        # raises ValueError with a clear message.
        # This fails FAST — better than mysterious runtime errors.

        missing = []
        if not self.DISCORD_TOKEN:
            missing.append("DISCORD_TOKEN")
        if not self.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if self.OWNER_ID == 0:
            missing.append("OWNER_ID")
        if missing:
            raise ValueError(
                f"Missing required config: {', '.join(missing)}. Check .env file."
            )
