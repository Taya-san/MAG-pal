"""
Load and validate bot configuration from environment variables.

Every config value has a sensible default so the bot won't crash
if something is missing from .env — except the 3 required fields.
Access values like: config.DISCORD_TOKEN, config.PAL_NAME, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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
    @property
    def DISCORD_TOKEN(self): return self.disc_token
    @property
    def OPENROUTER_API_KEY(self): return self.openrouter_key
    @property
    def OWNER_ID(self): return self.owner_id
    @property
    def PAL_NAME(self): return self.pal_name
    @property
    def USER_NAME(self): return self.user_name
    @property
    def PERSONALITY(self): return self.personality
    @property
    def INTERESTS(self): return self.interests
    @property
    def MAG_LANG(self): return self.mag_lang
    @property
    def RESPONSE_LENGTH(self): return self.response_length
    @property
    def MODEL(self): return self.model
    @property
    def MAX_TOKENS(self): return self.max_tokens
    @property
    def TEMPERATURE(self): return self.temperature
    @property
    def DEBUG(self): return self.debug
    @property
    def SUMMARIZE(self): return self.summarize
    @property
    def SUMMARY_INTERVAL(self): return self.summary_interval
    @property
    def DB_PATH(self): return self.db_path
    @property
    def HEURISTIC_CONTINUATION_MINUTES(self): return self.heuristic_continuation_minutes
    @property
    def OWNER_NAMES(self): return self.owner_names
    @property
    def NAME_TO_OWNER(self): return self.name_to_owner

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
