# MAG-pal

Personal Discord AI pal bot with adaptive response system, hierarchical long-term memory, and real-time streaming intervention.

## How It Works

```
User message → Discord
     ↓
bot.py: on_message() — gates (owner only), stores in DB, spawns AI task
     ↓
responder.py: evaluate() — fast heuristic pre-filter
     ├── SKIP (80%): greetings, noise → no API call
     ├── RESPOND: @mention, question, continuation → call AI
     └── ASK_AI: ambiguous → let AI decide (<SILENT>)
     ↓
responder.py: build_prompt() — assemble system + history + keywords + session summary
     ↓
openrouter.py: call() or stream_with_reasoning()
     ↓
memory_store.py: parse_and_store() — break AI response into hierarchical blocks
     ↓
stream_intervention.py: scan reasoning tokens mid-stream → inject memory → re-prompt
     ↓
Discord: send response (or stay silent if <SILENT>)
```

## Quick Start

```bash
cp .env.example .env
# Edit .env with your DISCORD_TOKEN, OPENROUTER_API_KEY, OWNER_ID
pip install -r requirements.txt
python main.py
```

## Architecture

| File | Role |
|------|------|
| `main.py` | Entry point — creates bot, validates config, starts event loop |
| `bot.py` | Discord client — `on_message()` gating, command routing, `_process_ai_message()` pipeline, `_stream_with_intervention()` multi-injection loop |
| `responder.py` | Decision engine — `evaluate()` (SKIP/RESPOND/ASK_AI heuristics), `build_prompt()` (conversation assembly), `build_continuation_prompt()` (re-prompt after injection) |
| `openrouter.py` | OpenRouter API client — sync `call()` for summarization, async `stream_with_reasoning()` with two-phase reasoning detection |
| `memory_store.py` | Hierarchical memory — `MemoryStore` (SQLite CRUD for blocks/sentences/relations), `BlockParser` (markdown line-by-line structural detection), LDA classifier |
| `stream_intervention.py` | Streaming intervention engine — token buffering, keyword index matching, three-tier context retrieval, cooldown management |
| `db.py` | Bot database — messages, keywords, sessions, `user_aliases` |
| `config.py` | `.env` loader with validation |
| `keywords.py` | Regex-based keyword extraction from messages |
| `stopwords.py` | 1232 English + Indonesian stopwords |
| `text_utils.py` | Shared text utilities — `extract_top_words()`, `filter_injected_keywords()`, `merge_top_words()`, table keyword extraction |
| `tree_parser.py` | Standalone parse tree (Node + TreeParser) for response structure visualization |

## Data Flow: Memory System

### Storage (when the AI responds)

```
AI response text
     ↓
BlockParser.parse_and_store(text, owner_id)
     ↓
Detects: code blocks (```), equations ($$), tables (|...|), lists (1. / -), paragraphs
     ↓
Creates: blocks (paragraph/code/table/list/equation) + sentences + block_relations
     ↓
_propagate_top_words(): parent keywords → children, bare reference injection, type seeding
     ↓
_classify_sentences(): LDA scores → label='flagged' or 'unlabeled'
```

### Retrieval (during generation)

```
AI reasoning stream token
     ↓
StreamIntervention.buffer_token() — assemble sub-words into complete words
     ↓
StreamIntervention.check_match(word, reasoning_tokens)
     ↓
Keyword index lookup → get sentence's block top_words
     ↓
filter_injected_keywords() → clean content words only
     ↓
Compute ratio = matched / total content keywords
     ↓
Tier decision:
  ├── ratio ≥ 70% → deep: return full block tree
  ├── is structural word → specific: return matching child block
  └── default → surface: return parent paragraph
     ↓
Set cooldown (150 tokens per sentence)
     ↓
Build continuation prompt → restart stream with injected context
```

## Tier System

| Tier | Trigger | Returns | Example |
|------|---------|---------|---------|
| **Deep** | ≥70% of block's content keywords match reasoning | Full block tree (parent + all children) | AI deeply discussing the topic |
| **Specific** | Reasoning mentions structural type word ("code", "table") | Exact child block of that type | AI says "the code looks like..." |
| **Surface** | ≥1 keyword matches | Parent paragraph (intro text) | AI briefly mentions concept |

## Cooldown System

- Each injected sentence gets a **150-token cooldown** (counted per reasoning token)
- Same sentence cannot re-trigger within a single response
- **Per-call isolation**: each `_stream_with_intervention()` creates a fresh intervention instance — multiple concurrent messages don't interfere
- **Output decay**: after response, cooldowns are reduced by 50% of output length (proxy for "user reading time")
- **Multi-intervention**: the `while True` loop restarts the stream on each match; stops only when a stream completes without triggering

## Privacy: owner_id

| Level | How |
|-------|-----|
| **Storage** | Every block and sentence stores `owner_id` (Discord user ID, NULL = public) |
| **Indexing** | `build_match_index(owner_id=123)` filters `WHERE owner_id IS NULL OR owner_id = 123` |
| **Streaming** | Per-call `StreamIntervention(owner_id=...)` uses the filtered index |

## User Aliases

Private per-user nickname system (`!alias`):

```
!alias add bintang for taya   → maps "bintang" → taya's owner_id
!alias list                     → shows your nicknames
!alias remove bintang           → removes
```

Aliases are per-user and invisible to others. Each person builds their own nickname dictionary. Configure known users in `.env`:

```env
OWNER_NAMES=12345:taya,67890:bob
```

## Configuration (.env)

```env
# Required
DISCORD_TOKEN=your_token
OPENROUTER_API_KEY=sk-or-...
OWNER_ID=your_discord_id

# Personality
PAL_NAME=MAG
USER_NAME=User
PERSONALITY=chill, curious, supportive
INTERESTS=programming, math
RESPONSE_LENGTH=short

# AI Model
MODEL=openrouter/free
MAX_TOKENS=500
TEMPERATURE=0.7

# Behavior
DEBUG=false
SUMMARIZE=false
SUMMARY_INTERVAL=100
HEURISTIC_CONTINUATION_MINUTES=2

# Multi-user (optional)
OWNER_NAMES=12345:taya,67890:bob
```

## Commands

| Command | Description |
|---------|-------------|
| `!debug` | Toggle debug logging |
| `!kw` | Show learned keywords |
| `!remember <word>` | Manually add keyword |
| `!forget <word>` | Remove keyword |
| `!bad` | Flag last response as bad |
| `!showprompt` | Show the last AI prompt |
| `!stats` | Show bot statistics |
| `!clear confirm` | Clear message history |
| `!alias add <nick> for <user>` | Add nickname mapping |
| `!alias list` | Show your nicknames |
| `!alias remove <nick>` | Remove nickname |

## Database Schema

### Bot DB (`db.py` — palbot.db)

```sql
messages(message_id, channel_id, author, content, created_at)
keywords(keyword, frequency, is_manual, last_seen)
sessions(channel_id, summary, message_count, last_updated)
user_aliases(owner_id, alias, resolves_to) UNIQUE(owner_id, alias)
```

### Memory Store (`memory_store.py` — memory.db)

```sql
blocks(id, text, type, parent_id, top_words, sentiment, owner_id, created_at)
sentences(id, text, block_id, line_number, type, embedding, score, label, owner_id, created_at)
block_relations(id, parent_id, child_id, relation_type)
```

## Testing

```bash
# Core memory + propagation
python memory_store.py

# Streaming intervention
python -c "from stream_intervention import *; ..."

# Full stress test (requires venv)
source ai_stuff/rag_env/bin/activate
python tests/test_synthetic.py
```
