"""
Discord client — message gating, AI processing, streaming intervention loop.

This is the main orchestrator. on_message() receives Discord events,
gates them (owner-only, no bots, no duplicates), stores data in the
database, and spawns AI processing as background tasks.
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone

import discord
from discord.ext import tasks
# Docs: https://discordpy.readthedocs.io/en/stable/
#        https://docs.python.org/3/library/asyncio.html

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
    """
    Our bot class. Extends discord.Client, which is the main class from
    the discord.py library. discord.Client handles:
      - Connecting to Discord via WebSocket
      - Receiving events (messages, members joining, etc.)
      - Sending messages through the Discord API
    
    By overriding methods like on_message(), we hook into those events.
    The discord.py library calls on_message() automatically whenever
    someone sends a message in a channel the bot can see.
    """
    
    def __init__(self, config: Config):
        # === DISCORD INTENTS ===
        # Intents are Discord's permission system for bots. They control
        # what events your bot can RECEIVE from Discord's servers.
        #
        # Think of Intents like a checklist you send to Discord saying
        # "I want to receive these types of events." If you don't list
        # an intent, Discord simply won't send you those events.
        #
        # discord.Intents.default() returns a pre-set object with common
        # intents enabled: guilds, messages, reactions, voice, etc.
        # Three intents are NOT enabled by default:
        #   - message_content: PRIVILEGED — manually enabled below
        #   - members: required for member join/leave events
        #   - presences: required for online status events
        intents = discord.Intents.default()
        
        # intents.message_content = True enables the MESSAGE CONTENT intent.
        # This is a PRIVILEGED intent — you MUST explicitly enable it in
        # Discord's developer portal for your bot application.
        # Without it, on_message() will receive messages but message.content
        # will be empty (""). The library CAN see that a message was sent,
        # but Discord won't tell us what the message actually says.
        # This is a privacy measure by Discord.
        intents.message_content = True
        
        # super().__init__(intents=intents) calls discord.Client's constructor
        # and passes our intents config to it. discord.Client stores these
        # intents internally and sends them to Discord during the WebSocket
        # handshake, saying "I want to receive these event types."
        #
        # This is NOT overwriting — it's the FIRST setup of the parent class.
        # Without super().__init__(), PalBot would have no connection to Discord.
        super().__init__(intents=intents)
        
        # self.config stores the bot's configuration (loaded from .env).
        # config is a frozen dataclass (Config class from config.py), so
        # its values cannot be changed at runtime.
        self.config = config
        
        # === DEPENDENCIES ===
        # Database (db.py) - stores messages, keywords, sessions, aliases.
        # Uses aiosqlite (async SQLite) so it doesn't block the event loop.
        self.db = Database(config.DB_PATH)
        
        # OpenRouter client - handles API calls to LLMs.
        # Has two modes: sync call() for simple requests, async stream()
        # for streaming with reasoning detection.
        self.openrouter = OpenRouterClient(config)
        
        # MemoryStore (memory_store.py) - hierarchical memory storage.
        # Stores parsed AI responses as blocks with parent-child relations.
        # Has its own SQLite database (memory.db).
        self.memory_store = MemoryStore()
        
        # Responder (responder.py) - the decision engine.
        # Decides if we should respond, builds prompts, parses responses.
        self.responder = Responder(config, self.db, self.openrouter, self.memory_store)
        
        # StreamIntervention (stream_intervention.py) - the real-time
        # memory injection system. Scans AI reasoning tokens mid-generation
        # and injects relevant memory context.
        self.stream_intervention = StreamIntervention(self.memory_store)
        
        # === STATE ===
        # Debug mode toggle. When True, every AI prompt, response, and
        # decision is logged to palbot.log. Useful for troubleshooting.
        self.debug_mode = config.DEBUG
        
        # Deduplication deque (max 10,000 items). If Discord reconnects,
        # it might re-send the last message we already processed.
        # This deque prevents processing the same message twice.
        # deque is like a list but with a max length — old items
        # automatically fall off when new ones are added.
        self.seen_messages = deque(maxlen=10000)
        
        # Semaphore — limits concurrent AI API calls to 10.
        # When the 11th message comes in, it waits until one of the
        # previous 10 calls finishes. This prevents hitting OpenRouter's
        # rate limits and also limits memory/CPU usage from parallel
        # AI generation.
        self._ai_semaphore = asyncio.Semaphore(10)
        
        # Per-channel tracking dicts. These store state for each Discord
        # channel the bot interacts with. Each dict is keyed by channel ID.
        self.last_response_times = {}    # when did we last reply in this channel?
        self.last_prompt = {}            # what prompt did we last send?
        self.last_response_info = {}     # what was the last response? (for !bad)
        
        # Safety limit: if the bot ends up in thousands of channels,
        # these dicts would use too much memory. _cleanup_channel_data()
        # pops the oldest entries when we exceed this limit.
        self._MAX_CHANNEL_DATA = 200
        
        # Per-channel locks for summarization. When the AI generates a
        # summary of old conversation, we use a per-channel lock so that
        # two summarization tasks don't run simultaneously for the same
        # channel (wasteful). One lock per channel.
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

    # ---- DISCORD LIFECYCLE HOOKS ----
    # These are called automatically by discord.Client at specific points
    # in the bot's lifecycle. They're called "hooks" because we're
    # hooking into the library's internal event system.

    async def setup_hook(self):
        """
        Called by discord.py after login but before on_ready.
        
        This is where we set up async resources. The bot is connected to
        Discord at this point, but not ready to receive events yet.
        We use it to:
          1. Connect to the database (async SQLite connection)
          2. Start the keyword decay background task
        
        await self.db.connect() — we AWAIT this because connect() is an
        async function. In Python, await yields control back to the event
        loop while the database connection is being established, so the
        bot isn't blocked during this time.
        """
        await self.db.connect()
        self.keyword_decay_loop.start()

    async def close(self):
        """
        Ordered shutdown hook. Called when the bot disconnects.
        
        Order matters here:
          1. Cancel the background task first (so it doesn't fire during shutdown)
          2. Close the database connection
          3. Close the AI client (frees HTTP connections)
          4. Call parent close (disconnects from Discord WebSocket)
        
        If we closed the DB last, a background task might try to write
        to a closed connection and crash.
        """
        self.keyword_decay_loop.cancel()
        await self.db.close()
        self.openrouter.close()
        await super().close()

    async def on_ready(self):
        """
        Called once when the bot is fully connected and ready.
        
        Fires once when the bot connects to Discord after setup_hook.
        We log connection info so you can verify the bot is working
        when it starts up.
        
        self.user is the bot's own Discord user object. It contains
        the bot's username, ID, avatar, etc.
        """
        logger.info(f"Bot online as {self.user} (ID: {self.user.id})")
        logger.info(f"Owner ID: {self.config.OWNER_ID}")
        if self.debug_mode:
            logger.info("DEBUG MODE ENABLED")

    # ---- MAIN MESSAGE HANDLER ----
    # This is THE most important method in the entire bot.
    # It's called by discord.py for EVERY message the bot can see.

    async def on_message(self, message):
        """
        Main entry point — called by discord.py for every message.
        
        The 'message' parameter is a discord.Message object. It has
        attributes like:
          - message.author (discord.User who sent it)
          - message.content (the text of the message)
          - message.channel (where it was sent)
          - message.id (unique Discord snowflake ID)
          - message.reference (if it's a reply to another message)
        
        This must return quickly! It runs in the Discord event loop.
        All heavy work (AI calls) is spawned as BACKGROUND TASKS.
        
        We apply 5 gates before processing anything:
        """
        
        # GATE 1: Ignore messages from bots (including itself).
        # If we didn't check this, the bot would reply to its OWN messages,
        # creating an infinite loop. message.author.bot is True for any
        # bot account (including this bot), False for real users.
        if message.author.bot:
            return

        # GATE 2: Owner-only gate.
        # This is THE key privacy/security gate.
        # message.author.id is the Discord user ID of whoever sent the
        # message. We ONLY process messages from config.OWNER_ID.
        # Other users' messages are NEVER stored, processed, or seen.
        if message.author.id != self.config.OWNER_ID:
            return

        # GATE 3: Ignore empty messages.
        # This catches images without text, embeds, stickers, etc.
        # message.content.strip() removes leading/trailing whitespace,
        # then we check if anything remains. If the message had no text
        # content, it's not something we can process.
        if not message.content.strip():
            return

        # GATE 4: Deduplication check.
        # During Discord reconnects, the same message might be delivered
        # twice. message.id is a unique snowflake ID. We check if we've
        # already seen this ID in our seen_messages deque.
        # If we have, skip it. If not, add it to the deque.
        if message.id in self.seen_messages:
            return
        self.seen_messages.append(message.id)

        content = message.content.strip()

        # GATE 5: Check if this is a command (starts with "!").
        # If so, route it to the command handler and return immediately.
        # Commands don't need AI processing — they just do simple things
        # like toggling debug mode or showing stats.
        if content.startswith("!"):
            await self._route_command(message)
            return

        # === IT'S A CHAT MESSAGE — PROCESS IT ===
        # At this point, we know this is a real message from the owner.
        # Now we start the AI processing pipeline.
        
        self.stats["messages_processed"] += 1

        # Step 1: Store the message in the database.
        # We call db.store_message() with a unique ID, the channel ID,
        # author type ("user" meaning you), and the message content.
        # str(message.id) converts Discord's snowflake integer to a string.
        await self.db.store_message(
            str(message.id), str(message.channel.id), "user", content
        )

        # Step 2: Extract keywords from the message.
        # This uses regex to find important words (nouns, technical terms)
        # and stores them in the database with a frequency counter.
        keywords = extract_keywords(content)
        for kw in keywords:
            await self.db.upsert_keyword(kw)

        # Step 3 (Debug): Log what we got.
        if self.debug_mode:
            logger.info(f"[DEBUG] Message: {content[:100]}")
            logger.info(f"[DEBUG] Keywords: {keywords}")

        # Step 4: Spawn AI processing as a BACKGROUND TASK.
        # asyncio.create_task() schedules a coroutine to run "in the
        # background." on_message() returns immediately after this,
        # unblocking the Discord event loop. The AI processing happens
        # concurrently without blocking new messages from arriving.
        #
        # This is why Discord bots stay responsive even during long
        # AI generations — they never block on AI calls.
        asyncio.create_task(self._process_ai_message(message))

    # ---- COMMAND ROUTING ----
    # Dispatches !commands to handler functions in commands.py.
    # Uses a dict lookup (O(1)) instead of if/elif chains.

    async def _route_command(self, message):
        """
        Route !commands to handler functions in commands.py.
        
        message.content.strip().split(maxsplit=1) splits the message
        content into two parts at the first space:
          "!alias add bintang for taya" -> ["!alias", "add bintang for taya"]
        
        The first part (e.g. "!alias") is the command name.
        The second part (everything else) is the arguments.
        If there's no space at all, args = "".
        """
        parts = message.content.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Look up the command in commands.HANDLERS dict.
        # commands.HANDLERS is a dict like {"!debug": cmd_debug, ...}
        # If the command doesn't exist, handler = None.
        handler = commands.HANDLERS.get(cmd)

        if handler:
            # Every command handler is wrapped in try/except so a single
            # bad command can't crash the bot.
            try:
                # handler(self, message, args) calls the function from
                # commands.py with the bot instance, the message, and args.
                # The handler doesn't use 'self' — it receives the bot
                # as a parameter instead. This is called "dependency injection"
                # and makes the handlers testable without a Discord connection.
                await handler(self, message, args)
            except discord.Forbidden:
                # The bot tried to do something it doesn't have permission for
                await message.channel.send("I don't have permission to do that.")
            except discord.HTTPException as e:
                # Some Discord API error (timeout, bad request, etc.)
                await message.channel.send(f"Discord error: {e}")
            except Exception as e:
                # Catch-all for unexpected errors in command handlers
                logger.error(f"Command {cmd} failed: {e}", exc_info=True)
                await message.channel.send(f"Command failed: {e}")
        elif self.debug_mode:
            logger.info(f"[DEBUG] Unknown command: {cmd}")

    # ---- CHANNEL DATA CLEANUP ----
    # Prevents unbounded memory growth in per-channel dicts.

    def _cleanup_channel_data(self):
        """
        Remove oldest entries from per-channel dicts when they get too large.
        
        If the bot is in hundreds of channels across Discord servers,
        the per-channel dicts (last_response_times, last_prompt,
        last_response_info) would grow without limit.
        
        This method uses a simple strategy: while the dict has more
        entries than _MAX_CHANNEL_DATA, pop the OLDEST key.
        Python dicts preserve insertion order (since 3.7), so
        next(iter(dict)) gives us the first/oldest key.
        """
        while len(self.last_response_times) > self._MAX_CHANNEL_DATA:
            self.last_response_times.pop(next(iter(self.last_response_times)))
        while len(self.last_prompt) > self._MAX_CHANNEL_DATA:
            self.last_prompt.pop(next(iter(self.last_prompt)))
        while len(self.last_response_info) > self._MAX_CHANNEL_DATA:
            self.last_response_info.pop(next(iter(self.last_response_info)))

    # ---- AI MESSAGE PROCESSING (BACKGROUND TASK) ----
    # This is the main AI pipeline. It runs as a background task for
    # every chat message from the owner.
    #
    # Pipeline:
    #   1. Cleanup old channel data
    #   2. Wait for semaphore (rate limiting)
    #   3. Run heuristic evaluation (responder.evaluate)
    #   4. Build prompt (responder.build_prompt)
    #   5. Call OpenRouter (openrouter.call or _stream_with_intervention)
    #   6. Parse and handle response

    async def _process_ai_message(self, message):
        """
        The full AI pipeline: heuristic -> prompt -> API -> parse -> respond.
        
        This is spawned as a background task by asyncio.create_task().
        It runs CONCURRENTLY with other AI calls (up to 10 at a time
        due to the semaphore).
        """

        self._cleanup_channel_data()

        # Safety check: if the bot isn't fully connected yet, skip.
        # self.user is None before on_ready() fires.
        if self.user is None:
            logger.warning("Bot not ready yet, skipping AI processing")
            return

        # should_summarise tracks whether we should check if it's time
        # to summarize old conversations. Only non-SKIP messages count.
        should_summarise = False

        # === AI PROCESSING (errors caught separately) ===
        try:
            # async with self._ai_semaphore: — this is a Python context
            # manager for asyncio.Semaphore. It AWAITS until a slot is
            # available, then acquires one. When the block exits, it
            # automatically releases the slot.
            #
            # If 10 AI calls are already running, the 11th message will
            # WAIT here until one of the first 10 finishes.
            async with self._ai_semaphore:

                # === STEP 1: HEURISTIC EVALUATION ===
                # This is the decision engine. It checks the message
                # against fast rule-based heuristics and returns one of:
                #   SKIP: greeting/noise, don't respond
                #   RESPOND: clear request, definitely respond
                #   ASK_AI: ambiguous, let the AI decide
                #
                # The heuristics save ~80% of API costs because most
                # messages can be handled without calling the AI.
                result = await self.responder.evaluate(
                    message, self.user, self.last_response_times
                )

                # If the heuristic says SKIP, return immediately.
                # No AI call, no cost, no delay.
                if result == HeuristicResult.SKIP:
                    self.stats["heuristic_skip"] += 1
                    if self.debug_mode:
                        logger.info("[DEBUG] Heuristic: SKIP")
                    return

                # For non-SKIP messages, increment the session message count.
                # This is used to trigger periodic summarization.
                await self.db.increment_message_count(
                    str(message.channel.id)
                )
                should_summarise = True

                # Determine if we should include the <SILENT> instruction.
                # ASK_AI results include it (the AI can choose to stay silent).
                # RESPOND results don't (the heuristic already decided).
                should_include_silent = result == HeuristicResult.ASK_AI
                if result == HeuristicResult.RESPOND:
                    self.stats["heuristic_respond"] += 1
                else:
                    self.stats["heuristic_ask_ai"] += 1

                # === STEP 2: BUILD PROMPT ===
                # Assemble the full message array for the AI:
                #   - System prompt (personality, rules)
                #   - Recent conversation history (last 20 messages)
                #   - Keyword hints (what you've discussed)
                #   - Session summary (long-term memory)
                #   - Injection guard (security)
                #   - The user's current message
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
                # Two paths:
                #   Path A: use_streaming=True  -> streaming with intervention
                #   Path B: use_streaming=False -> simple non-streaming call
                #
                # We use streaming when the keyword index is non-empty
                # (there are flagged sentences to match against).
                # Otherwise, the simpler non-streaming path is used.
                use_streaming = bool(self.stream_intervention.keyword_index)

                if use_streaming:
                    response_text, usage = await self._stream_with_intervention(
                        message, prompt
                    )
                else:
                    # message.channel.typing() shows the "Bot is typing..."
                    # indicator in Discord. async with starts and stops it.
                    async with message.channel.typing():
                        # asyncio.to_thread() runs the synchronous openrouter.call()
                        # in a separate thread so it doesn't block the event loop.
                        response_text, usage = await asyncio.to_thread(
                            self.openrouter.call, prompt
                        )

                # Update statistics
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
                # parse_response() checks if the AI output includes
                # the <SILENT> token. If yes, the AI decided to stay
                # quiet. If no, it returned the actual response text.
                should_respond, clean_text = self.responder.parse_response(
                    response_text
                )

                if self.debug_mode and not should_respond and response_text.strip():
                    logger.info(
                        f"[DEBUG] SILENT triggered. Full model output: {response_text!r}"
                    )

                if should_respond:
                    # Update per-channel tracking
                    self.last_response_times[message.channel.id] = (
                        datetime.now(timezone.utc)
                    )
                    self.last_response_info[message.channel.id] = {
                        "user_message": message.content,
                        "bot_response": clean_text,
                        "prompt": prompt,
                        "timestamp": time.time(),
                    }

                    # Send the response to Discord
                    await message.channel.send(clean_text)

                    # Store the response in the database with a unique ID
                    # (message.id-resp to distinguish it from the user's message)
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
        # If we incremented the message count AND summarization is enabled
        # in config, check if it's time to generate a summary.
        if should_summarise and self.config.SUMMARIZE:
            await self._check_summarization(str(message.channel.id))

    async def _stream_with_intervention(self, message, prompt):
        """
        Stream from OpenRouter with real-time memory injection.
        
        Instead of waiting for the full response (like openrouter.call()),
        this method streams tokens as they arrive and scans them for
        keyword matches against stored memory.
        
        Key design: multi-intervention loop (while True).
        When a match fires, we:
          1. Save the reasoning so far
          2. Build a continuation prompt with the injected memory
          3. RESTART the stream with the new prompt
          4. The new stream is ALSO scanned for matches
          5. Loop continues until a stream completes naturally (no match)
        
        This is the core innovation of the bot.
        """
        
        # Create a per-call StreamIntervention instance. This gets its
        # own cooldowns and word buffer (isolated from other concurrent
        # calls), but shares the same keyword_index from the master
        # instance (self.stream_intervention.keyword_index).
        intervention = StreamIntervention(
            memory_store=self.memory_store,
            shared_index=self.stream_intervention.keyword_index,
            owner_id=self.config.OWNER_ID,
        )
        
        # partial_reasoning: accumulates ALL reasoning tokens from every
        # stream iteration. When we re-prompt, we send these back so the
        # AI knows what it was thinking before the interruption.
        partial_reasoning = []
        
        # output_tokens: the FINAL response text. Only tokens from the
        # 'output' phase are collected here. Reasoning tokens are never
        # shown to the user.
        output_tokens = []
        
        start = time.time()
        
        # current_prompt changes on each intervention (gets enriched
        # with injected memory context).
        current_prompt = prompt
        
        try:
            async with message.channel.typing():
                # The multi-intervention loop.
                # Each iteration starts a new stream. The loop continues
                # until a stream completes WITHOUT any intervention match.
                while True:
                    interrupted = False
                    
                    # Stream tokens from OpenRouter.
                    # stream_with_reasoning() is an ASYNC GENERATOR that
                    # yields (token_text, phase) tuples.
                    async for token, phase in self.openrouter.stream_with_reasoning(current_prompt):
                        
                        if phase == 'reasoning':
                            # Accumulate reasoning tokens
                            partial_reasoning.append(token)
                            
                            # Decrement all cooldowns by 1 per reasoning token
                            intervention.decrement_cooldowns(1)
                            
                            # Detect paragraph boundary: double newline or
                            # start of a structural block (equation, code).
                            para_boundary = (
                                '\n\n' in token
                                or '\n$$' in token
                                or '\n```' in token
                            )
                            
                            # Feed the token into the word buffer
                            # (needed to keep buffer in sync for future use).
                            intervention.buffer_token(token)
                            
                            if para_boundary:
                                # Extract completed paragraph from reasoning
                                # (everything up to the last boundary marker).
                                reasoning_text = ''.join(partial_reasoning)
                                parts = reasoning_text.split('\n\n')
                                # Use the most recently completed paragraph
                                paragraph = ''
                                for p in reversed(parts):
                                    p = p.strip()
                                    if len(p) > 20:  # meaningful paragraph
                                        paragraph = p
                                        break
                                
                                if paragraph:
                                    match = intervention.check_paragraph(paragraph)
                                    if match:
                                        current_prompt = self.responder.build_continuation_prompt(
                                            current_prompt,
                                            reasoning_text,
                                            match['context'],
                                        )
                                        intervention.reset_buffer()
                                        interrupted = True
                                        break
                                
                        elif phase == 'output':
                            output_tokens.append(token)
                    
                    if not interrupted:
                        break  # stream completed naturally — exit while True

            # After streaming, check the last paragraph for matches.
            reasoning_text = ''.join(partial_reasoning)
            last_paragraphs = [p.strip() for p in reasoning_text.split('\n\n') if len(p.strip()) > 20]
            if last_paragraphs:
                match = intervention.check_paragraph(last_paragraphs[-1])
                if match:
                    current_prompt = self.responder.build_continuation_prompt(
                        current_prompt,
                        reasoning_text,
                        match['context'],
                    )
                    # One more stream with the final enriched prompt
                    async for token, phase in self.openrouter.stream_with_reasoning(current_prompt):
                        if phase == 'reasoning':
                            partial_reasoning.append(token)
                        else:
                            output_tokens.append(token)

            latency = time.time() - start
            full_output = ''.join(output_tokens)

            # Apply output decay: cooldowns are reduced by 50% of output length
            intervention.apply_output_decay(len(full_output.split()))
            return full_output, {"latency": round(latency, 2)}
        finally:
            # ALWAYS reset the buffer, even on error
            intervention.reset_buffer()

    async def _check_summarization(self, channel_id: str):
        """
        Check if it's time to summarize this channel's conversation.
        
        Runs outside the AI semaphore so summarization doesn't
        block AI calls for new messages.
        
        Triggered every SUMMARY_INTERVAL non-SKIP messages.
        Uses per-channel locks to prevent duplicate summarization.
        """
        try:
            # Get or create a per-channel lock
            if channel_id not in self._summary_locks:
                self._summary_locks[channel_id] = asyncio.Lock()

            # Lock ensures only one summarization task per channel
            async with self._summary_locks[channel_id]:
                session = await self.db.get_or_create_session(channel_id)
                count = session["message_count"]

                # If the count hasn't hit the interval yet, skip.
                # Re-check inside the lock to prevent race conditions.
                if count <= 0 or count % self.config.SUMMARY_INTERVAL != 0:
                    return

                logger.info(f"Summarizing channel {channel_id} at {count} messages")

                # Fetch the last SUMMARY_INTERVAL messages
                messages = await self.db.get_recent_messages(
                    channel_id, self.config.SUMMARY_INTERVAL
                )
                if not messages:
                    return

                # Build a summary prompt and call the AI (non-streaming)
                prompt = self.responder.build_summary_prompt(messages)
                summary_text, _ = await asyncio.to_thread(
                    self.openrouter.call, prompt
                )
                summary_text = summary_text.strip()

                # Reject empty/non-substantive summaries
                if not summary_text or len(summary_text) < 10:
                    logger.warning(f"Empty summary for {channel_id}, keeping previous")
                    return

                if self.debug_mode:
                    logger.info(f"[DEBUG] Summary for {channel_id}: {summary_text[:200]}")

                await self.db.update_session_summary(channel_id, summary_text)
                logger.info(f"Summary saved for channel {channel_id}")

        except Exception as e:
            logger.error(f"Summarization failed for {channel_id}: {e}", exc_info=True)

    # ---- BACKGROUND TASKS ----
    # @tasks.loop(hours=6) is a discord.py decorator that makes the
    # following async function run every 6 hours indefinitely.
    # It automatically handles waiting, errors, and cleanup.
    
    @tasks.loop(hours=6)
    async def keyword_decay_loop(self):
        """
        Run every 6 hours: reduce frequency of stale keywords.
        
        Keywords not seen in 7+ days lose 1 frequency point.
        When frequency reaches 0, the keyword is deleted.
        Manual keywords (!remember) are exempt.
        """
        try:
            logger.info("Running keyword decay...")
            await self.db.decay_keywords(days=7)
        except Exception as e:
            logger.error(f"Keyword decay failed, will retry: {e}", exc_info=True)

    @keyword_decay_loop.before_loop
    async def before_decay(self):
        """
        Runs before the FIRST iteration of keyword_decay_loop.
        Waits until the bot is fully connected to Discord before
        scheduling the first decay.
        """
        await self.wait_until_ready()
