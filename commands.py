"""
Command handlers for PalBot — extracted from bot.py for clean separation.

Each !command maps to a handler function here. Called by bot._route_command()
in bot.py which dispatches using the HANDLERS dict.

Instead of being methods on the bot class (self), these are standalone
functions that receive (bot, message, args). The 'bot' parameter is
the PalBot instance, giving access to bot.db, bot.config, etc.

This separation means commands can be tested without a Discord connection:

    result = await cmd_stats(mock_bot, mock_message, "")
"""

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger("palbot")


async def cmd_debug(bot, message, args):
    """Toggle debug mode on/off at runtime."""
    bot.debug_mode = not bot.debug_mode
    await message.add_reaction("\u2705")  # checkmark emoji
    await message.channel.send(f"Debug mode: {'ON' if bot.debug_mode else 'OFF'}")


async def cmd_keywords(bot, message, args):
    """List all learned keywords with frequency and recency."""
    keywords = await bot.db.get_all_keywords()
    if not keywords:
        await message.channel.send("No keywords learned yet.")
        return
    lines = []
    for kw in keywords:
        # 📌 (pushpin emoji) marks manual keywords (from !remember) vs auto-learned
        tag = "\U0001f4cc" if kw["is_manual"] else "  "
        lines.append(f"{tag} {kw['keyword']} (freq: {kw['frequency']}, last: {kw['last_seen'][:10]})")
    await message.channel.send(f"**Keywords ({len(lines)}):**\n" + "\n".join(lines[:25]))


async def cmd_remember(bot, message, args):
    """Manually add a keyword that never decays."""
    if not args:
        await message.channel.send("Usage: !remember <word>")
        return
    word = args.strip().lower()
    # manual=True means this keyword is EXEMPT from decay
    await bot.db.upsert_keyword(word, manual=True)
    await message.add_reaction("\u2705")


async def cmd_forget(bot, message, args):
    """Remove keywords. !forget <word> or !forget to clear all auto."""
    if args:
        word = args.strip().lower()
        await bot.db.remove_keyword(word)
        await message.channel.send(f"Forgot '{word}'")
    else:
        await bot.db.clear_keywords()  # removes only auto-learned, keeps manual
        await message.channel.send("Cleared all auto-learned keywords.")


async def cmd_bad(bot, message, args):
    """Flag the last bot response as bad for review."""
    info = bot.last_response_info.get(message.channel.id)
    if not info:
        await message.channel.send("No previous response to flag in this channel.")
        return
    logger.warning("[BAD] User: %s | Bot: %s", info["user_message"], info["bot_response"])
    await message.channel.send("Flagged as bad response. Logged for review.")


async def cmd_showprompt(bot, message, args):
    """Show the exact prompt sent to the AI for the last message."""
    prompt = bot.last_prompt.get(message.channel.id)
    if not prompt:
        await message.channel.send("No prompt available for this channel.")
        return
    lines = []
    for i, msg in enumerate(prompt):
        lines.append(f"--- [{i}] {msg['role']} ---\n{msg['content']}\n")
    full = "\n".join(lines)
    # Discord has a 2000 character limit per message
    if len(full) > 1900:
        full = full[:1900] + "\n...(truncated)"
    await message.channel.send(f"```\n{full}\n```")


async def cmd_stats(bot, message, args):
    """Show bot statistics: uptime, messages, AI calls, heuristic breakdown."""
    uptime = time.time() - bot.stats["start_time"]
    msg_count = await bot.db.get_message_count()
    text = (
        f"**Stats:**\n"
        f"Uptime: {uptime/3600:.1f}h\n"
        f"Messages processed: {bot.stats['messages_processed']}\n"
        f"AI calls: {bot.stats['ai_calls']}\n"
        f"  Heuristic respond: {bot.stats['heuristic_respond']}\n"
        f"  Heuristic skip: {bot.stats['heuristic_skip']}\n"
        f"  Heuristic ask AI: {bot.stats['heuristic_ask_ai']}\n"
        f"  Heuristic silent: {bot.stats['heuristic_silent']}\n"
        f"Total tokens: {bot.stats['total_tokens']}\n"
        f"DB messages: {msg_count}"
    )
    await message.channel.send(text)


async def cmd_clear(bot, message, args):
    """Delete ALL message history. Requires !clear confirm (safety)."""
    if args.strip().lower() != "confirm":
        await message.channel.send(
            "\u26a0\ufe0f This will delete ALL message history and reset session data.\n"
            "Type `!clear confirm` to proceed."
        )
        return
    await bot.db.clear_messages()
    await message.channel.send("Cleared all message history.")
    await message.add_reaction("\u2705")


async def cmd_pin(bot, message, args):
    """!pin <sentence_id> — mark a sentence as permanent (never expires)."""
    if not args:
        await message.channel.send("Usage: `!pin <sentence_id>`. Get sentence IDs from the debug log.")
        return
    try:
        sid = int(args[0].strip())
    except ValueError:
        await message.channel.send("Sentence ID must be a number.")
        return
    cur = bot.memory_store.conn.cursor()
    cur.execute("SELECT text FROM sentences WHERE id = ? AND owner_id = ?",
               (sid, bot.config.OWNER_ID))
    row = cur.fetchone()
    if not row:
        await message.channel.send(f"Sentence {sid} not found or not yours.")
        return
    cur.execute("UPDATE sentences SET permanent = 1, expires_at = NULL WHERE id = ?", (sid,))
    bot.memory_store.conn.commit()
    await message.channel.send(f"Pinned sentence **{sid}**: _{row[0][:100]}..._\nThis memory will never expire.")

async def cmd_alias(bot, message, args):
    """
    Manage private nickname aliases.
    
    Syntax:
      !alias add <nickname> for <username>  — map nickname to a user
      !alias list                            — show your nicknames
      !alias remove <nickname>               — remove a nickname
    
    Aliases are PRIVATE per-user. User A can call user B "bintang"
    and user B has no idea. Each user manages their own namespace.
    """
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
    owner = bot.config.OWNER_ID

    if sub == "list":
        aliases = await bot.db.get_aliases(owner)
        if not aliases:
            await message.channel.send("You have no aliases set.")
            return
        lines = []
        for row in aliases:
            # Look up the display name for this owner_id
            res_name = bot.config.OWNER_NAMES.get(row["resolves_to"], str(row["resolves_to"]))
            lines.append(f"  {row['alias']} -> {res_name}")
        await message.channel.send("**Your aliases:**\n" + "\n".join(lines))
        return

    if sub == "remove":
        if len(parts) < 2:
            await message.channel.send("Usage: `!alias remove <nickname>`")
            return
        await bot.db.remove_alias(owner, parts[1])
        await message.channel.send(f"Removed alias '{parts[1].lower()}'")
        await message.add_reaction("\u2705")
        return

    if sub == "add":
        # Parse: !alias add <nickname> for <username>
        try:
            for_idx = parts.index("for")
            nickname_parts = parts[1:for_idx]
            target_parts = parts[for_idx + 1:]
        except (ValueError, IndexError):
            await message.channel.send("Usage: `!alias add <nickname> for <username>`")
            return

        nickname = " ".join(nickname_parts).lower().strip()
        target_name = " ".join(target_parts).lower().strip()

        if not nickname or not target_name:
            await message.channel.send("Both nickname and target name are required.")
            return

        # Resolve the target name to an owner_id using config's NAME_TO_OWNER
        target_id = bot.config.NAME_TO_OWNER.get(target_name)
        if not target_id:
            await message.channel.send(
                f"Unknown user '{target_name}'. Known: {', '.join(bot.config.NAME_TO_OWNER.keys())}"
            )
            return

        await bot.db.add_alias(owner, nickname, target_id)
        await message.channel.send(f"Got it! '{nickname}' now refers to {target_name}.")
        await message.add_reaction("\u2705")
        return

    await message.channel.send(f"Unknown subcommand '{sub}'. Try `!alias` for help.")


async def cmd_help(bot, message, args):
    """List all available commands."""
    text = (
        "**Commands:**\n"
        "`!debug` — Toggle debug mode\n"
        "`!kw` — Show learned keywords\n"
        "`!remember <word>` — Manually add a keyword\n"
        "`!forget <word>` — Remove a keyword\n"
        "`!forget` — Clear all auto-keywords\n"
        "`!bad` — Flag last response as bad\n"
        "`!showprompt` — Show the last AI prompt\n"
        "`!stats` — Show bot statistics\n"
        "`!clear` — Clear message history\n"
        "`!alias` — Manage nicknames\n"
        "`!help` — Show this message"
    )
    await message.channel.send(text)


async def resolve_mention(bot, owner_id: int, text: str) -> int | None:
    """
    Check if any word in the message matches a known alias.
    
    Returns:
        The target owner_id the alias resolves to, or None.
    """
    words = set(text.lower().split())
    aliases = await bot.db.get_aliases(owner_id)
    for row in aliases:
        if row["alias"] in words:
            return row["resolves_to"]
    return None


# Handler dispatch map.
# bot._route_command() does: handler = commands.HANDLERS.get(cmd)
# This is O(1) dict lookup instead of if/elif chains.
HANDLERS = {
    "!debug": cmd_debug,
    "!kw": cmd_keywords,
    "!remember": cmd_remember,
    "!forget": cmd_forget,
    "!bad": cmd_bad,
    "!showprompt": cmd_showprompt,
    "!stats": cmd_stats,
    "!clear": cmd_clear,
    "!alias": cmd_alias,
    "!help": cmd_help,
}
