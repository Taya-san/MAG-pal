# bot.py
"""Discord client — message gating, AI processing, streaming intervention loop.

This is the main orchestrator. on_message() receives Discord events,
gates them (owner-only, no bots, no duplicates), stores data, and
spawns AI processing as background tasks. Commands are handled by
commands.py. The AI pipeline (heuristic -> prompt -> API -> response)
lives in _process_ai_message(). Multi-intervention streaming lives
in _stream_with_intervention().
"""

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
import commands

logger = logging.getLogger("palbot")


class PalBot(discord.Client):
    # Main bot class. Extends discord.Client for the event loop.
    # Holds all state: database, AI client, stats, debug mode.

    def __init__(self, config: Config):
        # === DISCORD INTENTS ===
        # Intents control what events Discord sends us.
        # Intents.default() gives us: guilds, messages, reactions, etc.
        # message_content is PRIVILEGED - must be enabled in Discord
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
        """Initialize async resources: connect DB, start background tasks."""
        # This is where we set up things that need async.
        # Database connection and background tasks go here.
        await self.db.connect()
        self.keyword_decay_loop.start()

    async def close(self):
        """Ordered shutdown: cancel tasks, close DB, close AI client, disconnect Discord."""
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
        """Log connection info when the bot connects to Discord."""
        # Logs connection info so you can verify it's working.
        logger.info(f"Bot online as {self.user} (ID: {self.user.id})")
        logger.info(f"Owner ID: {self.config.OWNER_ID}")
        if self.debug_mode:
            logger.info("DEBUG MODE ENABLED")

    # ---- MAIN MESSAGE HANDLER ----

    async def on_message(self, message):
        """Main message handler - gates, commands, or AI processing background task."""
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

        # === IT'S A CHAT MESSAGE - PROCESS IT ===

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
        """Dispatch !commands to handlers via commands.HANDLERS dict lookup."""
        parts = message.content.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = commands.HANDLERS.get(cmd)

        if handler:
            try:
                await handler(self, message, args)
            except discord.Forbidden:
                await message.channel.send("I don't have permission to do that.")
            except discord.HTTPException as e:
                await message.channel.send(f"Discord error: {e}")
            except Exception as e:
                logger.error(f"Command {cmd} failed: {e}", exc_info=True)
                await message.channel.send(f"Command failed: {e}")
        elif self.debug_mode:
            logger.info(f"[DEBUG] Unknown command: {cmd}")

    # ---- CHANNEL DATA CLEANUP ----

    def _cleanup_channel_data(self):
        """Evict oldest channel data when per-channel dicts exceed MAX_CHANNEL_DATA."""
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
        """AI processing pipeline: heuristic -> prompt -> API -> parse -> respond."""
        # Runs as a background task for every chat message.
        # Do NOT call this directly - use asyncio.create_task().
        #
        # Pipeline:
        #   1. Cleanup old channel data
        #   2. Wait for semaphore (rate limiting)
        #   3. Run heuristic evaluation (responder.evaluate)
        #   4. Build prompt (responder.build_prompt)
        #   5. Call OpenRouter (openrouter.call)
        #   6. Parse and handle response

        # Safety check: if bot isn't fully ready, skip
        if self.user is None:
            logger.warning("Bot not ready yet, skipping AI processing")
            return

        # Track if we should summarise: only mark after non-SKIP messages
        should_summarise = False

        # === AI PROCESSING (errors caught separately) ===
        try:
            # Acquire semaphore - limits concurrent AI calls to 10
            async with self._ai_semaphore:

                # === STEP 1: HEURISTIC EVALUATION ===
                result = await self.responder.evaluate(
                    message, self.user, self.last_response_times
                )

                # SKIP: noise/greeting - do nothing
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
                                match = await intervention.check_match(
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
                match = await intervention.check_match(word, partial_reasoning[-400:])
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
        """Trigger AI summarization when message count hits SUMMARY_INTERVAL."""
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

                # Re-check inside the lock - another task might have
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

    # ---- BACKGROUND TASKS ----

    @tasks.loop(hours=6)
    async def keyword_decay_loop(self):
        """Decrease frequency of stale keywords every 6 hours."""
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
        """Wait until ready before first keyword decay iteration."""
        # We wait until the bot is connected to Discord and the
        # database is ready before scheduling the first decay.
        await self.wait_until_ready()
