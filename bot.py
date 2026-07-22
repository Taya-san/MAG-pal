# bot.py
# THE HEART — Discord client, event handlers, commands, and background tasks.
#
# This is where the main loop lives: on_message receives Discord events,
# gates them (only you, no bots, no duplicates), stores data in the
# database, spawns AI processing in background tasks, and routes commands.
#
# Architecture:
#   on_message() → synchronous gates (microseconds) → command or spawn AI task
#   _process_ai_message() → heuristic → prompt → OpenRouter → response
#   _route_command() → dict dispatch to handler methods
#   keyword_decay_loop() → background task every 6 hours

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone

import discord
from discord.ext import tasks

from config import Config
from db import Database
from keywords import extract_keywords
from openrouter import OpenRouterClient
from responder import Responder, HeuristicResult
from memory_store import MemoryStore
from stream_intervention import StreamIntervention

logger = logging.getLogger("palbot")


class PalBot(discord.Client):
    # Main bot class. Extends discord.Client for the event loop.
    # Holds all state: database, AI client, stats, debug mode.

    def __init__(self, config: Config):
        # === DISCORD INTENTS ===
        # Intents control what events Discord sends us.
        # Intents.default() gives us: guilds, messages, reactions, etc.
        # message_content is PRIVILEGED — must be enabled in Discord
        # Developer Portal > Bot > Message Content Intent.
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config

        # === DEPENDENCIES ===
        self.db = Database(config.DB_PATH)
        self.openrouter = OpenRouterClient(config)
        self.memory_store = MemoryStore()
        self.responder = Responder(config, self.db, self.openrouter, self.memory_store)
        self.stream_intervention = StreamIntervention(self.memory_store)

        # === STATE ===

        # Debug mode toggle. Set from config but can be changed
        # at runtime with !debug command.
        self.debug_mode = config.DEBUG

        # Deduplication: keeps track of last 10,000 message IDs.
        # Prevents processing the same message twice (rare but possible
        # during Discord reconnect).
        self.seen_messages = deque(maxlen=10000)

        # Semaphore: limits concurrent AI calls to avoid rate limits.
        # Max 10 simultaneous API calls. Additional calls queue up.
        self._ai_semaphore = asyncio.Semaphore(10)

        # Per-channel tracking. Dicts keyed by channel ID (int).
        self.last_response_times = {}    # channel_id → datetime of last bot reply
        self.last_prompt = {}            # channel_id → last prompt sent to AI
        self.last_response_info = {}     # channel_id → last response data (for !bad)

        # Prevent unbounded memory growth in the dicts above.
        # If the bot is in many channels, old channel data gets evicted.
        self._MAX_CHANNEL_DATA = 200

        # Per-channel locks for summarization to prevent
        # two summarization tasks running for the same channel.
        self._summary_locks = {}

        # === STATISTICS ===
        # Tracks everything for !stats display and debugging.
        # start_time is used to calculate uptime.
        self.stats = {
            "messages_processed": 0,
            "ai_calls": 0,
            "heuristic_respond": 0,
            "heuristic_skip": 0,
            "heuristic_ask_ai": 0,
            "heuristic_silent": 0,
            "total_tokens": 0,
            "start_time": time.time(),
        }

    # ---- LIFECYCLE HOOKS ----

    async def setup_hook(self):
        # Called by discord.py after login but before on_ready.
        # This is where we set up things that need async.
        # Database connection and background tasks go here.
        await self.db.connect()
        self.keyword_decay_loop.start()

    async def close(self):
        # Called when the bot shuts down.
        # Order matters:
        # 1. Cancel the background task (so it doesn't fire during shutdown)
        # 2. Close the database connection
        # 3. Close the AI client (free HTTP connections)
        # 4. Call parent close (disconnects from Discord)
        self.keyword_decay_loop.cancel()
        await self.db.close()
        self.openrouter.close()
        await super().close()

    async def on_ready(self):
        # Fires once when the bot connects to Discord.
        # Logs connection info so you can verify it's working.
        logger.info(f"Bot online as {self.user} (ID: {self.user.id})")
        logger.info(f"Owner ID: {self.config.OWNER_ID}")
        if self.debug_mode:
            logger.info("DEBUG MODE ENABLED")

    # ---- MAIN MESSAGE HANDLER ----

    async def on_message(self, message):
        # THE ENTRY POINT — called once for every message the bot can see.
        # This runs in the Discord gateway event loop, so it must be FAST.
        # All heavy work (AI calls) is spawned as background tasks.

        # GATE 1: Ignore messages from bots (including itself)
        if message.author.bot:
            return

        # GATE 2: Ignore messages from anyone except the owner
        # This is the KEY privacy/security gate.
        # Other users' messages are NEVER stored, processed, or seen.
        if message.author.id != self.config.OWNER_ID:
            return

        # GATE 3: Ignore empty messages (images without text, etc.)
        if not message.content.strip():
            return

        # GATE 4: Ignore duplicates (already seen this message)
        # Prevents race conditions during Discord reconnect
        if message.id in self.seen_messages:
            return
        self.seen_messages.append(message.id)

        content = message.content.strip()

        # GATE 5: Is this a command? Route and return (no AI processing)
        if content.startswith("!"):
            await self._route_command(message)
            return

        # === IT'S A CHAT MESSAGE — PROCESS IT ===

        self.stats["messages_processed"] += 1

        # 1. Store the message in the database
        # author="user" means YOU sent this (not the bot)
        await self.db.store_message(
            str(message.id), str(message.channel.id), "user", content
        )

        # 2. Extract keywords and update database
        keywords = extract_keywords(content)
        for kw in keywords:
            await self.db.upsert_keyword(kw)

        # 4. Debug logging
        if self.debug_mode:
            logger.info(f"[DEBUG] Message: {content[:100]}")
            logger.info(f"[DEBUG] Keywords: {keywords}")

        # 5. Spawn AI processing as a BACKGROUND TASK
        # Using create_task so on_message returns immediately.
        # The AI call happens in the background without blocking
        # the Discord event loop. You can keep typing while
        # the bot is "thinking."
        asyncio.create_task(self._process_ai_message(message))

    # ---- COMMAND ROUTING ----

    async def _route_command(self, message):
        # Routes commands starting with "!" to handler methods.
        # Uses a dict lookup (O(1)) instead of if/elif chains.
        #
        # Format: !command [args]
        # Split on maxsplit=1 so only the first word is the command,
        # everything else is treated as arguments.

        # .strip() first to handle leading whitespace
        parts = message.content.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Map of command → handler method
        handlers = {
            "!debug": self._cmd_debug,
            "!kw": self._cmd_keywords,
            "!remember": self._cmd_remember,
            "!forget": self._cmd_forget,
            "!bad": self._cmd_bad,
            "!showprompt": self._cmd_showprompt,
            "!stats": self._cmd_stats,
            "!clear": self._cmd_clear,
            "!alias": self._cmd_alias,
            "!help": self._cmd_help,
        }

        handler = handlers.get(cmd)

        if handler:
            # Wrap every command handler in try/except so one bad command
            # can't crash the bot. Different exception types get
            # user-friendly error messages.
            try:
                await handler(message, args)
            except discord.Forbidden:
                await message.channel.send(
                    "I don't have permission to do that."
                )
            except discord.HTTPException as e:
                await message.channel.send(f"Discord error: {e}")
            except Exception as e:
                logger.error(
                    f"Command {cmd} failed: {e}", exc_info=True
                )
                await message.channel.send(f"Command failed: {e}")
        elif self.debug_mode:
            logger.info(f"[DEBUG] Unknown command: {cmd}")

    # ---- CHANNEL DATA CLEANUP ----

    def _cleanup_channel_data(self):
        # Prevents unbounded memory growth in per-channel dicts.
        # If the bot is in many channels across many servers,
        # these dicts could grow without limit. This removes
        # the oldest entries when we exceed _MAX_CHANNEL_DATA.
        #
        # Strategy: pop the first key (dicts preserve insertion
        # order in Python 3.7+) until we're under the limit.

        while len(self.last_response_times) > self._MAX_CHANNEL_DATA:
            self.last_response_times.pop(next(iter(self.last_response_times)))
        while len(self.last_prompt) > self._MAX_CHANNEL_DATA:
            self.last_prompt.pop(next(iter(self.last_prompt)))
        while len(self.last_response_info) > self._MAX_CHANNEL_DATA:
            self.last_response_info.pop(next(iter(self.last_response_info)))

    # ---- AI MESSAGE PROCESSING (BACKGROUND TASK) ----

    async def _process_ai_message(self, message):
        # Runs as a background task for every chat message.
        # Do NOT call this directly — use asyncio.create_task().
        #
        # Pipeline:
        #   1. Cleanup old channel data
        #   2. Wait for semaphore (rate limiting)
        #   3. Run heuristic evaluation (responder.evaluate)
        #   4. Build prompt (responder.build_prompt)
        #   5. Call OpenRouter (openrouter.call)
        #   6. Parse and handle response

        self._cleanup_channel_data()

        # Safety check: if bot isn't fully ready, skip
        if self.user is None:
            logger.warning("Bot not ready yet, skipping AI processing")
            return

        # Track if we should summarise: only mark after non-SKIP messages
        should_summarise = False

        # === AI PROCESSING (errors caught separately) ===
        try:
            # Acquire semaphore — limits concurrent AI calls to 10
            async with self._ai_semaphore:

                # === STEP 1: HEURISTIC EVALUATION ===
                result = await self.responder.evaluate(
                    message, self.user, self.last_response_times
                )

                # SKIP: noise/greeting — do nothing
                if result == HeuristicResult.SKIP:
                    self.stats["heuristic_skip"] += 1
                    if self.debug_mode:
                        logger.info("[DEBUG] Heuristic: SKIP")
                    return

                # Track message count for summarisation trigger.
                # Only non-SKIP messages count toward the summary interval.
                await self.db.increment_message_count(
                    str(message.channel.id)
                )
                should_summarise = True

                # Determine if <SILENT> instruction should be included
                should_include_silent = result == HeuristicResult.ASK_AI
                if result == HeuristicResult.RESPOND:
                    self.stats["heuristic_respond"] += 1
                else:
                    self.stats["heuristic_ask_ai"] += 1

                # === STEP 2: BUILD PROMPT ===
                prompt = await self.responder.build_prompt(
                    message, should_include_silent
                )
                self.last_prompt[message.channel.id] = prompt

                if self.debug_mode:
                    logger.info(f"[DEBUG] Prompt sent ({len(prompt)} messages)")
                    for msg in prompt:
                        logger.info(
                            f"[DEBUG]   [{msg['role']}]: {msg['content'][:200]}"
                        )

                # === STEP 3: CALL OPENROUTER ===
                use_streaming = bool(self.stream_intervention.keyword_index)

                if use_streaming:
                    response_text, usage = await self._stream_with_intervention(
                        message, prompt
                    )
                else:
                    async with message.channel.typing():
                        response_text, usage = await asyncio.to_thread(
                            self.openrouter.call, prompt
                        )

                # Update stats
                self.stats["ai_calls"] += 1
                if "prompt_tokens" in usage:
                    self.stats["total_tokens"] += (
                        usage["prompt_tokens"] + usage["completion_tokens"]
                    )

                if self.debug_mode:
                    logger.info(f"[DEBUG] Raw response: {response_text[:500]}")
                    logger.info(
                        "[DEBUG] Latency: {}s, Tokens: {}p + {}c".format(
                            usage.get("latency", "?"),
                            usage.get("prompt_tokens", "?"),
                            usage.get("completion_tokens", "?"),
                        )
                    )

                # === STEP 4: PARSE AND HANDLE RESPONSE ===
                should_respond, clean_text = self.responder.parse_response(
                    response_text
                )

                if self.debug_mode and not should_respond and response_text.strip():
                    logger.info(
                        f"[DEBUG] SILENT triggered. Full model output: {response_text!r}"
                    )

                if should_respond:
                    self.last_response_times[message.channel.id] = (
                        datetime.now(timezone.utc)
                    )

                    self.last_response_info[message.channel.id] = {
                        "user_message": message.content,
                        "bot_response": clean_text,
                        "prompt": prompt,
                        "timestamp": time.time(),
                    }

                    await message.channel.send(clean_text)

                    await self.db.store_message(
                        f"{message.id}-resp",
                        str(message.channel.id),
                        "bot",
                        clean_text,
                    )

                    if self.debug_mode:
                        logger.info(f"[DEBUG] Responded: {clean_text[:200]}")
                else:
                    self.stats["heuristic_silent"] += 1
                    if self.debug_mode:
                        logger.info("[DEBUG] Decision: <SILENT>")

        except Exception as e:
            logger.error(f"AI processing failed: {e}", exc_info=True)

        # === SUMMARISATION CHECK (runs even if AI failed) ===
        # The message count was already incremented above, so we
        # should still check even if the AI call errored out.
        if should_summarise and self.config.SUMMARIZE:
            await self._check_summarization(str(message.channel.id))

    async def _stream_with_intervention(self, message, prompt):
        """Stream with multi-intervention loop: scan reasoning tokens, inject memory on match.

        Creates a per-call StreamIntervention with shared keyword_index for isolation.
        Uses a while-True loop: on intervention, builds a continuation prompt and
        restarts the stream with enriched context. Loop continues until a stream
        completes without any intervention firing."""
        intervention = StreamIntervention(
            memory_store=self.memory_store,
            shared_index=self.stream_intervention.keyword_index,
            owner_id=self.config.OWNER_ID,
        )
        partial_reasoning = []
        output_tokens = []
        start = time.time()

        current_prompt = prompt
        try:
            async with message.channel.typing():
                while True:
                    interrupted = False
                    async for token, phase in self.openrouter.stream_with_reasoning(current_prompt):
                        if phase == 'reasoning':
                            partial_reasoning.append(token)
                            intervention.decrement_cooldowns(1)

                            for completed in intervention.buffer_token(token):
                                match = intervention.check_match(
                                    completed,
                                    partial_reasoning[-400:],
                                )
                                if match:
                                    current_prompt = self.responder.build_continuation_prompt(
                                        current_prompt,
                                        ''.join(partial_reasoning),
                                        match['context'],
                                    )
                                    intervention.reset_buffer()
                                    interrupted = True
                                    break
                            if interrupted:
                                break
                        elif phase == 'output':
                            output_tokens.append(token)
                    if not interrupted:
                        break

            for word in intervention.flush_buffer():
                match = intervention.check_match(word, partial_reasoning[-400:])
                if match:
                    current_prompt = self.responder.build_continuation_prompt(
                        current_prompt,
                        ''.join(partial_reasoning),
                        match['context'],
                    )
                    async for token, phase in self.openrouter.stream_with_reasoning(current_prompt):
                        if phase == 'reasoning':
                            partial_reasoning.append(token)
                        else:
                            output_tokens.append(token)

            latency = time.time() - start
            full_output = ''.join(output_tokens)

            intervention.apply_output_decay(len(full_output.split()))
            return full_output, {"latency": round(latency, 2)}
        finally:
            intervention.reset_buffer()

    async def _check_summarization(self, channel_id: str):
        # Checks if it's time to summarize this channel's conversation.
        # Triggered every SUMMARY_INTERVAL non-SKIP messages.
        # Uses a per-channel lock so only one summarization per channel
        # can run at a time.
        #
        # Fetches the last SUMMARY_INTERVAL messages, asks the AI to
        # summarize, and stores the result in sessions.summary.
        # Runs outside the semaphore so it doesn't block AI calls.
        try:
            # Get or create a per-channel lock
            if channel_id not in self._summary_locks:
                self._summary_locks[channel_id] = asyncio.Lock()

            # Lock ensures only one summarization task per channel
            async with self._summary_locks[channel_id]:
                session = await self.db.get_or_create_session(channel_id)
                count = session["message_count"]

                # Re-check inside the lock — another task might have
                # already summarized at this boundary.
                if count <= 0 or count % self.config.SUMMARY_INTERVAL != 0:
                    return

                logger.info(
                    f"Summarizing channel {channel_id} at {count} messages"
                )

                # Only fetch the messages since the last summary
                # (which is exactly SUMMARY_INTERVAL messages)
                messages = await self.db.get_recent_messages(
                    channel_id, self.config.SUMMARY_INTERVAL
                )

                if not messages:
                    return

                prompt = self.responder.build_summary_prompt(messages)

                summary_text, _ = await asyncio.to_thread(
                    self.openrouter.call, prompt
                )

                summary_text = summary_text.strip()

                # Guard against empty or non-substantive summaries
                if not summary_text or len(summary_text) < 10:
                    logger.warning(
                        f"Empty or too-short summary for {channel_id}, "
                        f"keeping previous"
                    )
                    return

                if self.debug_mode:
                    logger.info(
                        f"[DEBUG] Summary generated for {channel_id}: "
                        f"{summary_text[:200]}"
                    )

                await self.db.update_session_summary(
                    channel_id, summary_text
                )

                logger.info(f"Summary saved for channel {channel_id}")

        except Exception as e:
            logger.error(
                f"Summarization failed for {channel_id}: {e}",
                exc_info=True,
            )

    # ---- COMMAND HANDLERS ----

    async def _cmd_debug(self, message, args):
        # Toggle debug mode on/off at runtime.
        # When on, every message's AI prompt, latency, and
        # decision is logged to palbot.log.
        self.debug_mode = not self.debug_mode
        await message.add_reaction("\u2705")  # checkmark reaction
        await message.channel.send(
            f"Debug mode: {'ON' if self.debug_mode else 'OFF'}"
        )

    async def _cmd_keywords(self, message, args):
        # Lists all learned keywords with frequency and recency.
        # 📌 marks words you added with !remember (manual).
        keywords = await self.db.get_all_keywords()
        if not keywords:
            await message.channel.send("No keywords learned yet.")
            return
        lines = []
        for kw in keywords:
            tag = "\U0001f4cc" if kw["is_manual"] else "  "
            lines.append(
                f"{tag} {kw['keyword']} "
                f"(freq: {kw['frequency']}, "
                f"last: {kw['last_seen'][:10]})"
            )
        await message.channel.send(
            f"**Keywords ({len(lines)}):**\n" + "\n".join(lines[:25])
        )

    async def _cmd_remember(self, message, args):
        # Manually add a keyword (never decays, marked as manual).
        # Usage: !remember <word>
        if not args:
            await message.channel.send("Usage: !remember <word>")
            return
        word = args.strip().lower()
        await self.db.upsert_keyword(word, manual=True)
        await message.add_reaction("\u2705")

    async def _cmd_forget(self, message, args):
        # Remove keywords.
        # !forget <word> → remove specific keyword
        # !forget         → remove ALL auto-learned keywords
        if args:
            word = args.strip().lower()
            await self.db.remove_keyword(word)
            await message.channel.send(f"Forgot '{word}'")
        else:
            await self.db.clear_keywords()
            await message.channel.send(
                "Cleared all auto-learned keywords."
            )

    async def _cmd_bad(self, message, args):
        # Flag the last response as bad for review.
        # Logs the user message + bot response to palbot.log
        # so you can review and improve the system prompt.
        # Per-channel: each channel has its own "last response."
        info = self.last_response_info.get(message.channel.id)
        if not info:
            await message.channel.send(
                "No previous response to flag in this channel."
            )
            return
        logger.warning(
            "[BAD] User: %s | Bot: %s",
            info["user_message"],
            info["bot_response"],
        )
        await message.channel.send(
            "Flagged as bad response. Logged for review."
        )

    async def _cmd_showprompt(self, message, args):
        # Shows the exact prompt that was sent to the AI for the
        # last message in this channel. Useful for debugging:
        # - Is the context correct?
        # - Are the keywords what you expected?
        # - Is the system prompt right?
        #
        # Truncated to 1900 characters (Discord message limit).
        prompt = self.last_prompt.get(message.channel.id)
        if not prompt:
            await message.channel.send(
                "No prompt available for this channel."
            )
            return
        lines = []
        for i, msg in enumerate(prompt):
            role = msg["role"]
            content = msg["content"]
            lines.append(f"--- [{i}] {role} ---\n{content}\n")
        full = "\n".join(lines)
        if len(full) > 1900:
            full = full[:1900] + "\n...(truncated)"
        await message.channel.send(f"```\n{full}\n```")

    async def _cmd_stats(self, message, args):
        # Shows bot statistics: uptime, messages processed,
        # AI calls, heuristic breakdown, token usage.
        uptime = time.time() - self.stats["start_time"]
        msg_count = await self.db.get_message_count()
        text = (
            f"**Stats:**\n"
            f"Uptime: {uptime/3600:.1f}h\n"
            f"Messages processed: {self.stats['messages_processed']}\n"
            f"AI calls: {self.stats['ai_calls']}\n"
            f"  Heuristic respond: {self.stats['heuristic_respond']}\n"
            f"  Heuristic skip: {self.stats['heuristic_skip']}\n"
            f"  Heuristic ask AI: {self.stats['heuristic_ask_ai']}\n"
            f"  Heuristic silent: {self.stats['heuristic_silent']}\n"
            f"Total tokens: {self.stats['total_tokens']}\n"
            f"DB messages: {msg_count}"
        )
        await message.channel.send(text)

    async def _cmd_clear(self, message, args):
        if args.strip().lower() != "confirm":
            await message.channel.send(
                "\u26a0\ufe0f This will delete ALL message history "
                "and reset session data.\n"
                "Type `!clear confirm` to proceed."
            )
            return
        await self.db.clear_messages()
        await message.channel.send("Cleared all message history.")
        await message.add_reaction("\u2705")

    async def _cmd_alias(self, message, args):
        """Manage private nickname aliases: !alias add <nick> for <user>, list, remove."""
        parts = args.strip().split()
        if not parts:
            await message.channel.send(
                "**!alias usage:**\n"
                "`!alias add <nickname> for <username>` — nick someone\n"
                "`!alias list` — show your nicknames\n"
                "`!alias remove <nickname>` — remove a nickname"
            )
            return

        sub = parts[0].lower()
        owner = self.config.OWNER_ID

        if sub == "list":
            aliases = await self.db.get_aliases(owner)
            if not aliases:
                await message.channel.send("You have no aliases set.")
                return
            lines = []
            for row in aliases:
                res_name = self.config.OWNER_NAMES.get(row["resolves_to"], str(row["resolves_to"]))
                lines.append(f"  {row['alias']} -> {res_name}")
            await message.channel.send("**Your aliases:**\n" + "\n".join(lines))
            return

        if sub == "remove":
            if len(parts) < 2:
                await message.channel.send("Usage: `!alias remove <nickname>`")
                return
            await self.db.remove_alias(owner, parts[1])
            await message.channel.send(f"Removed alias '{parts[1].lower()}'")
            await message.add_reaction("\u2705")
            return

        if sub == "add":
            try:
                for_idx = parts.index("for")
                nickname = parts[1:for_idx]
                target_name = parts[for_idx + 1:]
            except (ValueError, IndexError):
                await message.channel.send(
                    "Usage: `!alias add <nickname> for <username>`\n"
                    "Example: `!alias add bintang for taya`"
                )
                return

            nickname = " ".join(nickname).lower().strip()
            target_name = " ".join(target_name).lower().strip()

            if not nickname or not target_name:
                await message.channel.send("Both nickname and target name are required.")
                return

            target_id = self.config.NAME_TO_OWNER.get(target_name)
            if not target_id:
                await message.channel.send(
                    f"Unknown user '{target_name}'. I don't know who that is.\n"
                    f"Known names: {', '.join(self.config.NAME_TO_OWNER.keys())}"
                )
                return

            await self.db.add_alias(owner, nickname, target_id)
            await message.channel.send(
                f"Got it! '{nickname}' now refers to {target_name}."
            )
            await message.add_reaction("\u2705")
            return

        await message.channel.send(f"Unknown subcommand '{sub}'. Try `!alias` for help.")

    async def resolve_mention(self, owner_id, text):
        """Check if any word in the message matches a known alias. Returns target owner_id or None."""
        words = set(text.lower().split())
        aliases = await self.db.get_aliases(owner_id)
        for row in aliases:
            if row["alias"] in words:
                return row["resolves_to"]
        return None

    async def _cmd_help(self, message, args):
        # Lists all available commands with brief descriptions.
        text = (
            "**Commands:**\n"
            "`!debug` \u2014 Toggle debug mode\n"
            "`!kw` \u2014 Show learned keywords\n"
            "`!remember <word>` \u2014 Manually add a keyword\n"
            "`!forget <word>` \u2014 Remove a keyword\n"
            "`!forget` \u2014 Clear all auto-keywords\n"
            "`!bad` \u2014 Flag last response as bad\n"
            "`!showprompt` \u2014 Show the last AI prompt\n"
            "`!stats` \u2014 Show bot statistics\n"
            "`!clear` \u2014 Clear message history\n"
            "`!help` \u2014 Show this message"
        )
        await message.channel.send(text)

    # ---- BACKGROUND TASKS ----

    @tasks.loop(hours=6)
    async def keyword_decay_loop(self):
        # Runs every 6 hours in the background.
        # Calls decay_keywords which reduces frequency of keywords
        # not seen in 7+ days. Old/unused topics fade away naturally.
        #
        # This runs as a discord.py task loop, which means it:
        # - Starts in setup_hook
        # - Waits for the bot to be ready (before_loop)
        # - Automatically stops if the bot disconnects
        # - Logs and continues on error
        try:
            logger.info("Running keyword decay...")
            await self.db.decay_keywords(days=7)
        except Exception as e:
            # Don't let a DB error permanently stop the loop
            # exc_info=True logs the full traceback
            logger.error(
                f"Keyword decay failed, will retry: {e}", exc_info=True
            )

    @keyword_decay_loop.before_loop
    async def before_decay(self):
        # This runs before the first iteration of the decay loop.
        # We wait until the bot is connected to Discord and the
        # database is ready before scheduling the first decay.
        await self.wait_until_ready()
