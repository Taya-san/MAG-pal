# main.py
# ENTRY POINT — run this to start the bot.
#
# This file does three things:
#   1. Load and validate configuration from .env
#   2. Set up logging (console + file)
#   3. Create the bot and connect to Discord
#
# Usage: python main.py
#
# Make sure .env is configured first with:
#   DISCORD_TOKEN, OPENROUTER_API_KEY, OWNER_ID

import logging
import sys

from config import Config
from bot import PalBot


def setup_logging(debug: bool):
    # Configure logging to write to both:
    #   - palbot.log (file, for history/review)
    #   - console/stdout (to see in terminal)
    #
    # When debug=True, log everything (including full AI prompts).
    # When debug=False, only log info and above (startup, errors).

    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler("palbot.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    # 1. Load .env into os.environ
    from dotenv import load_dotenv  # Docs: https://github.com/theskumar/python-dotenv
    load_dotenv()

    # 2. Load and validate config
    config = Config.from_env()
    config.validate()

    # 2. Set up logging (after config so we know the debug setting)
    setup_logging(config.DEBUG)

    logger = logging.getLogger("palbot")
    logger.info("Starting MAG-pal...")

    # 3. Create the bot and connect to Discord
    # bot.run() is blocking — it starts the event loop and
    # doesn't return until the bot disconnects.
    bot = PalBot(config)
    bot.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()
