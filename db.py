from __future__ import annotations
# db.py
# SQLite database layer using aiosqlite (async version of sqlite3).
# Stores: your messages, extracted keywords, and per-channel session data.
#
# Why aiosqlite instead of regular sqlite3?
#   Regular sqlite3 BLOCKS the event loop when writing to disk.
#   aiosqlite uses await so the bot can keep processing Discord
#   messages while the database writes in the background.
#
# All SQL uses parameterized queries (? placeholders).
# This prevents SQL injection attacks.

from __future__ import annotations

import aiosqlite
from datetime import datetime, timedelta, timezone

__all__ = ["Database"]


class Database:
    # Wraps an aiosqlite connection with convenient methods.
    # Each method handles its own commit so callers don't need to.

    def __init__(self, db_path: str):
        # db_path: path to the .db file on disk
        # ":memory:" creates a temporary in-memory DB (useful for testing)
        self.db_path = db_path
        self.conn = None

    async def connect(self):
        # Opens the database connection and creates tables if needed.
        # Must be called before any other method.
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        # ^ This lets us access columns by name: row["content"] instead of row[0]
        await self._create_tables()
        await self._migrate()

    async def close(self):
        # Closes the database connection gracefully.
        # Called when the bot shuts down.
        if self.conn:
            await self.conn.close()

    async def _create_tables(self):
        # Creates the 3 tables and their indexes.
        # Using IF NOT EXISTS so this is safe to run every startup.
        await self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                message_id TEXT UNIQUE NOT NULL,
                channel_id TEXT NOT NULL,
                author TEXT NOT NULL DEFAULT 'user',
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            -- ^ Stores every message you send AND every bot reply.
            --   message_id is the Discord snowflake (unique per message).
            --   author is "user" (you) or "bot" (the bot's replies).
            --   We NEVER store messages from other users.

            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY,
                keyword TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 1,
                is_manual INTEGER DEFAULT 0,
                last_seen TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            -- ^ Tracks important words you use frequently.
            --   is_manual=1 means you added it with !remember.
            --   is_manual=0 means the bot auto-extracted it.
            --   Keywords are used as hints in the AI prompt.

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                channel_id TEXT UNIQUE NOT NULL,
                summary TEXT DEFAULT '',
                message_count INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            -- ^ Per-channel tracking.
            --   message_count helps decide when to summarize.
            --   summary stores a compressed version of old conversation
            --   (future feature — not actively used yet).

            CREATE INDEX IF NOT EXISTS idx_messages_channel
                ON messages(channel_id, created_at DESC);
            -- ^ Speeds up: "give me the last 20 messages from this channel"

            CREATE INDEX IF NOT EXISTS idx_keywords_freq
                ON keywords(frequency DESC);
            -- ^ Speeds up: "give me the top 10 most frequent keywords"

            CREATE TABLE IF NOT EXISTS user_aliases (
                id INTEGER PRIMARY KEY,
                owner_id INTEGER NOT NULL,
                alias TEXT NOT NULL,
                resolves_to INTEGER NOT NULL,
                UNIQUE(owner_id, alias)
            );
            -- ^ Per-user private nickname mappings.
            --   owner_id = who OWNS this alias entry
            --   alias = what they call this person ("bintang")
            --   resolves_to = target owner_id — whose data to pull
        """)
        await self.conn.commit()

    async def _migrate(self):
        # Handles schema changes for existing databases.
        # When you add a new column, add an ALTER TABLE here.
        # The try/except catches "column already exists" errors.
        try:
            await self.conn.execute(
                "ALTER TABLE messages ADD COLUMN author TEXT NOT NULL DEFAULT 'user'"
            )
            await self.commit()
        except aiosqlite.OperationalError:
            # Column already exists — that's fine
            pass

    async def store_message(
        self, message_id: str, channel_id: str, author: str, content: str
    ):
        # Saves a message to the database.
        # INSERT OR IGNORE means duplicates (same message_id) are silently skipped.
        # This prevents double-storage on reconnect.
        await self.conn.execute(
            "INSERT OR IGNORE INTO messages (message_id, channel_id, author, content) VALUES (?, ?, ?, ?)",
            (message_id, channel_id, author, content),
        )
        await self.commit()

    async def get_recent_messages(self, channel_id: str, limit: int = 20):
        # Returns the most recent N messages from a channel, oldest-first.
        # Used to build conversation context for the AI.
        # The DESC + reverse trick: SQL gets newest first (fast with index),
        # then we reverse to oldest-first (what the AI expects).
        cursor = await self.conn.execute(
            "SELECT author, content, created_at FROM messages WHERE channel_id = ? ORDER BY created_at DESC LIMIT ?",
            (channel_id, limit),
        )
        rows = await cursor.fetchall()
        return list(reversed(rows))

    async def upsert_keyword(self, keyword: str, manual: bool = False):
        # "Update or Insert" a keyword.
        # If the keyword already exists: increment frequency, update last_seen.
        # If it's new: insert with frequency=1.
        # manual=True means the user used !remember (never decays).
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            """
            INSERT INTO keywords (keyword, frequency, is_manual, last_seen)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                frequency = frequency + 1,
                is_manual = CASE WHEN ? THEN 1 ELSE is_manual END,
                last_seen = ?
        """,
            (keyword, 1 if manual else 0, now, manual, now),
        )
        await self.commit()

    async def get_top_keywords(self, limit: int = 20):
        # Returns the most frequent keywords.
        # Used in the AI prompt to remind it what you talk about.
        cursor = await self.conn.execute(
            "SELECT keyword, frequency, is_manual FROM keywords ORDER BY frequency DESC LIMIT ?",
            (limit,),
        )
        return await cursor.fetchall()

    async def get_all_keywords(self):
        # Returns every keyword ordered by frequency.
        # Used by the !kw command.
        cursor = await self.conn.execute(
            "SELECT keyword, frequency, is_manual, last_seen FROM keywords ORDER BY frequency DESC"
        )
        return await cursor.fetchall()

    async def remove_keyword(self, keyword: str):
        # Deletes a single keyword.
        # Used by !forget <word>.
        await self.conn.execute(
            "DELETE FROM keywords WHERE keyword = ?", (keyword,)
        )
        await self.commit()

    async def clear_keywords(self):
        # Deletes ALL auto-learned keywords but keeps manual ones.
        # Used by !forget (without arguments).
        await self.conn.execute("DELETE FROM keywords WHERE is_manual = 0")
        await self.commit()

    async def decay_keywords(self, days: int = 7):
        # Runs every 6 hours via the background task.
        # If a keyword hasn't been seen in 7+ days, its frequency drops by 1.
        # If frequency reaches 0, the keyword is deleted.
        # This prevents stale topics from hanging around forever.
        # Manual keywords (!remember) are exempt from decay.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        await self.conn.execute(
            "UPDATE keywords SET frequency = MAX(0, frequency - 1) WHERE last_seen < ? AND is_manual = 0",
            (cutoff,),
        )
        await self.conn.execute("DELETE FROM keywords WHERE frequency <= 0")
        await self.commit()

    async def get_or_create_session(self, channel_id: str):
        # Gets or creates a session row for a channel.
        # Uses INSERT OR IGNORE which is atomic (no race conditions).
        # Sessions track per-channel message counts and summaries.
        await self.conn.execute(
            "INSERT OR IGNORE INTO sessions (channel_id, summary, message_count) VALUES (?, '', 0)",
            (channel_id,),
        )
        await self.commit()
        cursor = await self.conn.execute(
            "SELECT * FROM sessions WHERE channel_id = ?", (channel_id,)
        )
        return await cursor.fetchone()

    async def increment_message_count(self, channel_id: str):
        # Adds 1 to the session's message_count for this channel.
        # Tracks how many messages you've sent in each channel.
        await self.conn.execute(
            """
            INSERT INTO sessions (channel_id, summary, message_count)
            VALUES (?, '', 1)
            ON CONFLICT(channel_id) DO UPDATE SET
                message_count = message_count + 1,
                last_updated = CURRENT_TIMESTAMP
        """,
            (channel_id,),
        )
        await self.commit()

    async def update_session_summary(self, channel_id: str, summary: str):
        # Saves an AI-generated summary of old conversation.
        # Resets message_count to 0 so the next cycle starts fresh
        # and only the latest SUMMARY_INTERVAL messages are fetched.
        await self.conn.execute(
            """
            INSERT INTO sessions (channel_id, summary, message_count)
            VALUES (?, ?, 0)
            ON CONFLICT(channel_id) DO UPDATE SET
                summary = ?,
                message_count = 0,
                last_updated = CURRENT_TIMESTAMP
        """,
            (channel_id, summary, summary),
        )
        await self.commit()

    async def get_message_count(self):
        cursor = await self.conn.execute("SELECT COUNT(*) FROM messages")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def add_alias(self, owner_id: int, alias: str, resolves_to: int):
        """Add a private nickname mapping. owner_id=who owns it, resolves_to=who it refers to."""
        await self.conn.execute(
            "INSERT OR REPLACE INTO user_aliases (owner_id, alias, resolves_to) VALUES (?, ?, ?)",
            (owner_id, alias.lower().strip(), resolves_to),
        )
        await self.commit()

    async def remove_alias(self, owner_id: int, alias: str):
        """Remove a nickname from the owner's private alias table."""
        await self.conn.execute(
            "DELETE FROM user_aliases WHERE owner_id = ? AND alias = ?",
            (owner_id, alias.lower().strip()),
        )
        await self.commit()

    async def resolve_alias(self, owner_id: int, alias: str) -> int | None:
        """Look up who a nickname resolves to. Returns target owner_id or None."""
        cursor = await self.conn.execute(
            "SELECT resolves_to FROM user_aliases WHERE owner_id = ? AND alias = ?",
            (owner_id, alias.lower().strip()),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_aliases(self, owner_id: int):
        """Return all alias mappings for an owner as list of {alias, resolves_to}."""
        cursor = await self.conn.execute(
            "SELECT alias, resolves_to FROM user_aliases WHERE owner_id = ?",
            (owner_id,),
        )
        return await cursor.fetchall()

    async def clear_messages(self):
        # Deletes ALL messages and resets all session data.
        # Destructive operation — requires !clear confirm.
        await self.conn.execute("DELETE FROM messages")
        await self.conn.execute(
            "UPDATE sessions SET message_count = 0, summary = ''"
        )
        await self.commit()

    async def commit(self):
        # Writes pending changes to disk.
        # If the connection was closed, this does nothing instead of crashing.
        if self.conn:
            await self.conn.commit()
