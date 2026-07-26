"""
Load and validate bot configuration from environment variables.

Every config value has a sensible default so the bot won't crash
if something is missing from .env — except the 3 required fields.
Access values like: config.DISCORD_TOKEN, config.PAL_NAME, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
# Docs: https://github.com/theskumar/python-dotenv


@dataclass(frozen=True)
class Config:
    """Immutable bot configuration loaded from environment.

    Use Config.from_env() to create an instance from a .env file.
    The load_dotenv() call must happen before this (in main.py).
    All defaults are set here; validate() checks required fields.
    """

    # === REQUIRED ===
    disc_token: str
    openrouter_key: str
    owner_id: int

    # === PERSONALITY ===
    pal_name: str
    user_name: str
    personality: str
    interests: str
    mag_lang: str
    response_length: str

    # === AI MODEL ===
    model: str
    max_tokens: int
    temperature: float

    # === BEHAVIOR ===
    debug: bool
    summarize: bool
    summary_interval: int
    db_path: str
    heuristic_continuation_minutes: int

    # === MULTI-USER (optional) ===
    owner_names: dict[int, str]
    name_to_owner: dict[str, int]

    # === LEGACY ALIASES (for backward compatibility) ===
    _LEGACY_ALIASES = {
        'DISCORD_TOKEN': 'disc_token', 'OPENROUTER_API_KEY': 'openrouter_key',
        'MODEL': 'model', 'MAX_TOKENS': 'max_tokens', 'TEMPERATURE': 'temperature',
        'OWNER_ID': 'owner_id', 'PAL_NAME': 'pal_name', 'USER_NAME': 'user_name',
        'PERSONALITY': 'personality', 'INTERESTS': 'interests', 'MAG_LANG': 'mag_lang',
        'RESPONSE_LENGTH': 'response_length', 'DEBUG': 'debug',
        'SUMMARIZE': 'summarize', 'SUMMARY_INTERVAL': 'summary_interval',
        'DB_PATH': 'db_path', 'HEURISTIC_CONTINUATION_MINUTES': 'heuristic_continuation_minutes',
        'OWNER_NAMES': 'owner_names', 'NAME_TO_OWNER': 'name_to_owner',
    }

    def __getattr__(self, name):
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'Config' object has no attribute '{name}'")


    @classmethod
    def from_env(cls) -> Config:
        """Build a frozen Config from environment variables.

        Call this AFTER load_dotenv() has been called (typically in main.py).
        """
        raw_id = os.getenv("OWNER_ID", "0")
        owner_id = int(raw_id) if raw_id.strip() else 0

        # Parse OWNER_NAMES: "12345:taya,67890:bob"
        raw_names = os.getenv("OWNER_NAMES", "")
        owner_names: dict[int, str] = {}
        if raw_names:
            for pair in raw_names.split(","):
                pair = pair.strip()
                if ":" in pair:
                    uid, name = pair.split(":", 1)
                    owner_names[int(uid.strip())] = name.strip()
        name_to_owner = {v: k for k, v in owner_names.items()}

        return cls(
            disc_token=os.getenv("DISCORD_TOKEN", ""),
            openrouter_key=os.getenv("OPENROUTER_API_KEY", ""),
            owner_id=owner_id,
            pal_name=os.getenv("PAL_NAME", "MAG"),
            user_name=os.getenv("USER_NAME", "User"),
            personality=os.getenv("PERSONALITY", "chill, curious, supportive"),
            interests=os.getenv("INTERESTS", ""),
            mag_lang=os.getenv("MAG_LANG", ""),
            response_length=os.getenv("RESPONSE_LENGTH", "short"),
            model=os.getenv("MODEL", "openrouter/free"),
            max_tokens=int(os.getenv("MAX_TOKENS", "500")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            summarize=os.getenv("SUMMARIZE", "false").lower() == "true",
            summary_interval=int(os.getenv("SUMMARY_INTERVAL", "100")),
            db_path=os.getenv("DB_PATH", "palbot.db"),
            heuristic_continuation_minutes=int(os.getenv("HEURISTIC_CONTINUATION_MINUTES", "2")),
            owner_names=owner_names,
            name_to_owner=name_to_owner,
        )

    def validate(self) -> None:
        """Raise ValueError if required fields are missing."""
        missing = []
        if not self.disc_token:
            missing.append("DISCORD_TOKEN")
        if not self.openrouter_key:
            missing.append("OPENROUTER_API_KEY")
        if self.owner_id == 0:
            missing.append("OWNER_ID")
        if missing:
            raise ValueError(
                f"Missing required config: {', '.join(missing)}. Check .env file."
            )
