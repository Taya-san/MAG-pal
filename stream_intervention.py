from __future__ import annotations
"""
Stream Intervention — real-time memory injection during AI reasoning.

During streaming generation, the bot scans the model's chain-of-thought
reasoning tokens for keywords that match previously-flagged sentences
from past conversations stored in MemoryStore.

When a match fires, the bot interrupts the stream, re-prompts the AI with
the recalled memory as context, and resumes generation — all before the
user sees any output. This is designed to boost small models (8B-35B) by
giving them access to hierarchical long-term memory during reasoning.

Four phases:
1. BUILD: build_match_index() creates an inverted keyword index from
   all flagged sentences in the MemoryStore.
2. SCAN: buffer_token() assembles streaming sub-word tokens into complete
   words using regex word-boundary detection.
3. MATCH: check_match() compares completed words against the keyword index
   and decides which tier of context to return (surface/deep/specific).
4. COOLDOWN: each injected sentence gets a 150-token cooldown, preventing
   re-injection within the same response.
"""

import json
import re
import logging
from text_utils import filter_injected_keywords, extract_top_words

logger = logging.getLogger("palbot")

# === CONSTANTS ===
# These control the behavior of the intervention system.
# They're module-level so tests can import and check them.

COOLDOWN_LIMIT = 150      # How many reasoning tokens before same sentence can trigger again
OUTPUT_DECAY_RATIO = 0.5  # After response, reduce cooldowns by 50% of output length
DEEP_THRESHOLD = 0.7      # Minimum ratio of matched keywords for "deep" tier (>=70%)
WORD_MIN_LEN = 3          # Ignore words shorter than 3 characters (a, an, to, etc.)

# Structural type names that trigger the "specific" tier.
# When the AI mentions one of these during reasoning, and the memory
# store has a block of that type, the specific child block is injected.
STRUCTURAL_NAMES = {'table', 'tables', 'code', 'list', 'lists', 'equation', 'equations'}

# Maps plural/singular structural words to canonical block types.
# Used by _get_child_block() to find the right child block.
TYPE_MAP = {
    'table': 'table', 'tables': 'table',
    'code': 'code',
    'list': 'list', 'lists': 'list',
    'equation': 'equation', 'equations': 'equation',
}


class StreamIntervention:
    """
    Per-call intervention engine that scans streaming reasoning tokens.
    
    Each AI call gets its OWN instance of StreamIntervention (created
    in bot.py:_stream_with_intervention). This gives each concurrent
    call its own:
      - cooldowns dict (whose memory was injected, when)
      - word_buffer (sub-word tokens being assembled)
    
    BUT all instances share the same keyword_index (a dict passed by
    reference). This means the memory contents are shared across calls
    while the cooldowns and buffers are isolated.
    
    Args:
        memory_store: MemoryStore instance for DB queries. Not needed
                     if shared_index is provided (per-call instance).
        shared_index: Pre-built keyword_index dict. When provided, the
                     instance skips build_index() and uses this directly.
        owner_id: Discord user ID. Filters memory to only this user's
                 flagged sentences and public (NULL) ones.
    """
    
    def __init__(self, memory_store=None, shared_index=None, owner_id=None):
        self.store = memory_store
        self.owner_id = owner_id
        
        # If a shared_index is provided, use it directly. This avoids
        # rebuilding the index for every per-call instance.
        if shared_index is not None:
            self.keyword_index = shared_index
        else:
            self.keyword_index = {}
            
        # Per-call mutable state — each concurrent call gets fresh copies
        self.cooldowns = {}      # {sentence_id: remaining_tokens}
        self.word_buffer = ""    # accumulates sub-word characters into complete words
        
        # Only build index if we have a store AND no shared index
        if shared_index is None and memory_store is not None:
            self.build_index()

    # ===== INDEX BUILDING =====
    # The keyword index is an INVERTED INDEX: {lowercase_word: [sentence_info]}.
    # For every flagged sentence, every keyword in its block's top_words gets
    # an entry pointing back to that sentence. So "matrix" might point to
    # 5 different sentences from 3 different blocks.

    def build_index(self):
        """
        Build the keyword index from MemoryStore.
        
        Calls MemoryStore.build_match_index(self.owner_id) which runs:
          SELECT sentences with label='flagged'
          JOIN their blocks for top_words
          Filter WHERE (owner_id IS NULL OR owner_id = ?)
          Return {keyword: [{sentence_id, text, block_id, block_type}]}
        
        The index is stored in self.keyword_index and is shared across
        per-call instances via reference.
        """
        if not self.store:
            self.keyword_index = {}
            return
        self.keyword_index = self.store.build_match_index(self.owner_id)
        logger.info(f"Built intervention index: {len(self.keyword_index)} keywords")

    def rebuild_index(self):
        """
        Rebuild the index IN-PLACE so shared references stay valid.
        
        If we did self.keyword_index = new_dict, per-call instances that
        hold a reference to OLD dict would never see the update.
        Instead, we .clear() and .update() the existing dict object.
        """
        if not self.store:
            return
        new_index = self.store.build_match_index(self.owner_id)
        self.keyword_index.clear()
        self.keyword_index.update(new_index)
        logger.info(f"Rebuilt intervention index: {len(self.keyword_index)} keywords")

    # ===== WORD BUFFERING =====
    # Streaming APIs send SUB-WORD tokens (like "sym", "metric").
    # These are NOT full words. We need to:
    #   1. Accumulate tokens in a buffer
    #   2. Detect when a complete word has been received
    #   3. Extract the complete word
    #   4. Check it against the keyword index
    #
    # The regex pattern that does this:
    #   (?:^|\W)([a-zA-Z]{3,})\W
    #
    # Breaking this down:
    #   (?:^|\W)  = start of string OR any non-word character (space, punctuation)
    #   ([a-zA-Z]{3,}) = 3+ alphabetic characters (the word to capture)
    #   \W        = a non-word character AFTER the word (confirms it's complete)
    #
    # So "symmetric " matches because:
    #   (?:^|\W) matches the space before "symmetric"
    #   ([a-zA-Z]{3,}) captures "symmetric"
    #   \W matches the space after
    #
    # But "symmetri" (no trailing char) doesn't match — the word isn't complete.

    def buffer_token(self, token: str) -> list[str]:
        """
        Feed a streaming sub-word token and get back completed words.
        
        Imagine the API sends these tokens in sequence:
          " pro", "perty", " of", " sym", "metric", " matric", "es."
        
        buffer_token processes each one:
          1. Appends " pro" to buffer: buffer = " pro"
             No trailing \W — nothing emitted.
          2. Appends "perty": buffer = " property"
             Still no trailing \W — nothing emitted.
          3. Appends " of": buffer = " property of"
             Regex finds " property " → emits "property". Buffer = " of"
          4. Appends " sym": buffer = " of sym"
             No complete word — nothing emitted.
          5. Appends "metric": buffer = " of symmetric"
             No trailing \W — nothing emitted.
          6. Appends " matric": buffer = " of symmetric matric"
             Regex finds " symmetric " → emits "symmetric". Buffer = "matric"
          7. Appends "es.": buffer = "matrices."
             Regex finds "matrices." → emits "matrices". Buffer = "."
        
        Returns:
            List of lowercase completed words (may be empty).
        """
        self.word_buffer += token
        emitted = []
        while True:
            # re.search finds the FIRST match anywhere in the buffer
            m = re.search(r'(?:^|\W)([a-zA-Z]{3,})\W', self.word_buffer)
            if not m:
                break
            # m.group(1) is the captured word (letters only, no surrounding chars)
            word = m.group(1).lower()
            # Remove the matched portion from the buffer
            self.word_buffer = self.word_buffer[m.end():]
            emitted.append(word)
        return emitted

    @staticmethod
    def _buf_extract(buffer: str) -> tuple[list[str], str]:
        """
        Static version of buffer_token. Same logic, but operates on
        a passed-in string instead of self.word_buffer.
        
        Returns:
            (extracted_words_list, remaining_buffer_string)
        """
        emitted = []
        while True:
            m = re.search(r'(?:^|\W)([a-zA-Z]{3,})\W', buffer)
            if not m:
                break
            word = m.group(1).lower()
            buffer = buffer[m.end():]
            emitted.append(word)
        return emitted, buffer

    def flush_buffer(self) -> list[str]:
        """
        Extract the last word from the buffer at stream end.
        
        Unlike buffer_token(), this uses a DIFFERENT regex that
        doesn't require a trailing boundary character:
          (?:^|\W)?([a-zA-Z]{3,})$
        
        The $ anchors to the END of the string. This extracts
        anything left in the buffer when the stream ends.
        
        Returns:
            List with one trailing word, or empty list if buffer is clean.
        """
        m = re.match(r'(?:^|\W)?([a-zA-Z]{3,})$', self.word_buffer)
        if not m:
            return []
        self.word_buffer = ""
        return [m.group(1).lower()]

    def reset_buffer(self):
        """Clear the word buffer. Called in finally block for cleanup."""
        self.word_buffer = ""

    # ===== MATCHING =====
    # This is the core algorithm. For a completed word:
    #   1. Look up the word in the keyword index
    #   2. For each matching sentence (skip if on cooldown):
    #      a. Get the sentence's block-level top_words from DB
    #      b. Strip generic injected keywords (for ratio purity)
    #      c. Count how many of these appear in recent reasoning
    #      d. Compute ratio = matched / total_clean_keywords
    #   3. Decide tier based on ratio and word type
    #   4. Pick the highest-ratio match across all candidates
    #   5. Set cooldown on the winning sentence

    async def check_match(self, word: str, reasoning_tokens: list[str]) -> dict | None:
        """
        Check if a completed word triggers a memory intervention.
        
        NOTE: marked async for future async DB compatibility.
        Currently all internal calls are sync (fast SQLite reads <1ms).
        
        Args:
            word: The completed lowercase word to check.
            reasoning_tokens: List of recent raw reasoning tokens
                             (for cross-checking other keywords).
        
        Returns:
            Dict with tier, sentence_id, context, matched_word, ratio
            or None if no match found or all candidates on cooldown.
        """
        # Skip short words and empty indexes
        if len(word) < WORD_MIN_LEN or not self.keyword_index:
            return None
            
        # O(1) dict lookup for the word
        matches = self.keyword_index.get(word)
        if not matches:
            return None

        candidates = []
        for item in matches:
            sid = item['sentence_id']
            
            # Cooldown check: skip if this sentence was recently injected
            if self.cooldowns.get(sid, 0) > 0:
                continue

            # Fetch the block's aggregated keywords from DB.
            # This is a fast in-memory SQLite read (<1ms).
            top_words_list = self._get_sentence_top_words(sid)
            if not top_words_list:
                continue

            # Strip generic injected keywords for the RATIO calculation.
            # We ONLY count content words (eigenvalues, matrix, gradient),
            # NOT structural descriptors (function, example, property).
            # This prevents generic words from inflating the match ratio.
            clean_top = filter_injected_keywords(top_words_list)
            if not clean_top:
                continue

            # Count how many of the block's content keywords appear
            # in the recent reasoning tokens.
            matched_words = {word}  # start with the triggering word
            for raw in reasoning_tokens:
                # Strip non-letter chars from the raw token
                wc = re.sub(r'[^a-z]', '', raw.lower())
                if wc in clean_top:
                    matched_words.add(wc)

            # ratio = matched / total_content_keywords
            ratio = len(matched_words) / len(clean_top)
            
            # Check if the trigger word is a structural type name
            is_structural = word.lower() in STRUCTURAL_NAMES

            context = None
            tier = None

            # === TIER 1: DEEP (>=70% overlap) ===
            # The AI reasoning strongly overlaps with this memory.
            # Return the FULL BLOCK TREE (parent text + all children).
            if ratio >= DEEP_THRESHOLD:
                tree = self.store.get_block_tree(item['block_id']) if self.store else None
                context = self._format_tree(tree) if tree else item['text']
                tier = 'deep'

            # === TIER 2: SPECIFIC (structural word) ===
            # The AI mentioned "code" or "table". Return just that
            # specific child block.
            elif is_structural:
                context = self._get_child_block(item['block_id'], word) if self.store else item['text']
                tier = 'specific' if context else None

            # === TIER 3: SURFACE (everything else) ===
            # The AI mentioned a concept in passing. Return the
            # parent paragraph that introduced this topic.
            else:
                parent = self._get_parent_text(item['block_id']) if self.store else item['text']
                context = parent or item['text']
                tier = 'surface'

            if context and tier:
                candidates.append((ratio, tier, sid, context, word))

        if not candidates:
            return None

        # Sort candidates by ratio (highest first)
        candidates.sort(key=lambda x: -x[0])
        best_ratio, best_tier, best_sid, best_context, best_word = candidates[0]
        
        # Set cooldown: this sentence can't trigger again for COOLDOWN_LIMIT tokens
        self.cooldowns[best_sid] = COOLDOWN_LIMIT
        logger.info(f"Intervention: {best_tier} match on '{best_word}' ({best_ratio:.0%} overlap)")
        
        return {
            'tier': best_tier,
            'sentence_id': best_sid,
            'context': best_context,
            'matched_word': best_word,
            'ratio': best_ratio,
        }

    # ===== COOLDOWN MANAGEMENT =====
    # Cooldowns prevent the same sentence from being injected repeatedly.
    # Each sentence gets a counter. Every reasoning token decrements all
    # counters. When a counter hits 0, the sentence can trigger again.

    def decrement_cooldowns(self, n: int = 1):
        """
        Reduce ALL active cooldowns by n (default 1).
        
        Called once per reasoning token. When a cooldown reaches 0,
        the sentence becomes eligible for re-injection.
        
        Args:
            n: Number to subtract from each active cooldown.
        """
        expired = []
        for sid, remaining in self.cooldowns.items():
            new = remaining - n
            if new <= 0:
                expired.append(sid)
            else:
                self.cooldowns[sid] = new
        for sid in expired:
            del self.cooldowns[sid]

    def apply_output_decay(self, output_length: int):
        """
        Accelerate cooldown expiration based on output length.
        
        After the AI's response is generated, cooldowns are reduced by
        50% of the output word count. This is a heuristic:
        "the more words the AI produced after an injection, the further
        it's moved past that topic, so cooldown should expire faster."
        
        Args:
            output_length: Number of words in the final response.
        """
        decay = int(output_length * OUTPUT_DECAY_RATIO)
        expired = []
        for sid, remaining in self.cooldowns.items():
            new = remaining - decay
            if new <= 0:
                expired.append(sid)
            else:
                self.cooldowns[sid] = new
        for sid in expired:
            del self.cooldowns[sid]

    # ===== PARAGRAPH SCORING =====
    # Score completed paragraphs against the keyword index.
    #   - ratio <= 0.5: surface tier (parent paragraph only)

    def _get_block_top_words(self, block_id: int) -> list[str]:
        if not self.store:
            return []
        cur = self.store.conn.cursor()
        cur.execute("SELECT top_words FROM blocks WHERE id = ?", (block_id,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else []

    def score_paragraph(self, paragraph_text: str) -> list[dict]:
        para_words = set(extract_top_words(paragraph_text))
        if not para_words:
            return []

        block_data = {}
        for w in para_words:
            if w not in self.keyword_index:
                continue
            for item in self.keyword_index[w]:
                bid = item['block_id']
                if bid not in block_data:
                    block_data[bid] = {'matched': set(), 'sids': set()}
                block_data[bid]['matched'].add(w)
                block_data[bid]['sids'].add(item['sentence_id'])

        results = []
        for bid, info in block_data.items():
            top_words = self._get_block_top_words(bid)
            if not top_words:
                continue
            clean = filter_injected_keywords(top_words)
            if not clean:
                continue

            matched_count = sum(1 for kw in clean if kw in info['matched'])
            ratio = matched_count / len(clean) if clean else 0

            all_cool = all(self.cooldowns.get(sid, 0) > 0 for sid in info['sids'])
            if all_cool:
                continue

            has_structural = bool(info['matched'] & STRUCTURAL_NAMES)

            results.append({
                'block_id': bid,
                'ratio': ratio,
                'matched_words': info['matched'],
                'has_structural': has_structural,
                'sentence_ids': info['sids'],
            })

        results.sort(key=lambda x: -x['ratio'])
        return results

    def check_paragraph(self, paragraph_text: str) -> dict | None:
        scores = self.score_paragraph(paragraph_text)
        if not scores:
            return None

        best = scores[0]
        # Prioritization: require margin of at least 1 matched keyword
        # over the runner-up to avoid ties between different topics.
        if len(scores) >= 2:
            top_count = len(best['matched_words'])
            second_count = len(scores[1]['matched_words'])
            if top_count - second_count < 1:
                logger.info(f"Paragraph intervention: tie avoided "
                           f"(top={top_count}, second={second_count})")
                return None

        bid = best['block_id']
        ratio = best['ratio']
        has_structural = best['has_structural']
        best_sids = best['sentence_ids']
        best_matched = best['matched_words']

        context = None
        tier = None

        if ratio > 0.5:
            tree = self.store.get_block_tree(bid) if self.store else None
            context = self._format_tree(tree) if tree else None
            tier = 'deep'
        else:
            parent = self._get_parent_text(bid) if self.store else None
            if parent:
                context = parent
                tier = 'surface'
            elif best_sids:
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM sentences WHERE id = ?",
                           (list(best_sids)[0],))
                row = cur.fetchone()
                if row:
                    context = row[0]
                    tier = 'surface'

        if has_structural and context:
            struct_word = next((w for w in best_matched if w in STRUCTURAL_NAMES), None)
            if struct_word:
                child = self._get_child_block(bid, struct_word) if self.store else None
                if child and tier in ('surface', None):
                    context = context + '\n\n' + child
                    tier = 'specific'

        for sid in best_sids:
            self.cooldowns[sid] = COOLDOWN_LIMIT
        
        # Extend TTL for matched sentences (memory reinforcement)
        if self.store and best_sids:
            self.store._extend_sentence_ttl(best_sids)

        if not context or not tier:
            return None

        logger.info(f"Paragraph intervention: {tier} match (ratio={ratio:.0%}, block={bid})")
        return {
            'tier': tier,
            'context': context,
            'block_id': bid,
            'ratio': ratio,
        }

    # ===== DB HELPERS =====
    # These methods query the MemoryStore database to fetch context
    # for the different tiers. They're called synchronously because
    # SQLite in-memory queries complete in <1ms.

    def _get_sentence_top_words(self, sentence_id: int) -> list[str]:
        """
        Fetch the block-level top_words for a sentence.
        
        top_words are stored on the BLOCK (not the sentence) because
        they represent the aggregated keywords of the entire block
        (code block, table, paragraph) that this sentence belongs to.
        
        Args:
            sentence_id: The sentence's database ID.
            
        Returns:
            List of keyword strings, or empty list if not found.
        """
        if not self.store:
            return []
        cur = self.store.conn.cursor()
        cur.execute("""
            SELECT b.top_words FROM sentences s
            JOIN blocks b ON s.block_id = b.id
            WHERE s.id = ?
        """, (sentence_id,))
        row = cur.fetchone()
        if not row:
            return []
        return json.loads(row[0])

    def _get_parent_text(self, block_id: int) -> str | None:
        """
        Get the parent block's text for a child block.
        
        For surface-tier matches, we want the PARAGRAPH that introduced
        this code/table/list/equation — the parent's text provides
        the context that was lost when we jumped into inline content.
        
        The SQL joins block_relations (parent-child links) with blocks
        to get the parent's text for this child.
        
        Args:
            block_id: The child block's ID.
            
        Returns:
            Parent text string, or None if no parent exists.
        """
        if not self.store:
            return None
        cur = self.store.conn.cursor()
        cur.execute("""
            SELECT b.text FROM block_relations r
            JOIN blocks b ON b.id = r.parent_id
            WHERE r.child_id = ? AND r.relation_type = 'child_of'
        """, (block_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def _get_child_block(self, block_id: int, structural_word: str) -> str | None:
        """
        Get the text of a specific child block by type.
        
        For specific-tier matches, the AI mentioned a structural type
        name ("code", "table"). We return that specific child block's
        text. If the block itself is the target type, return its own
        text (for orphan blocks).
        
        Args:
            block_id: Block ID to search under.
            structural_word: "code", "table", "list", or "equation".
            
        Returns:
            The child block's text, or None if not found.
        """
        if not self.store:
            return None
        target_type = TYPE_MAP.get(structural_word.lower())
        if not target_type:
            return None
        block = self.store.get_block(block_id)
        if not block:
            return None
        children = self.store.get_block_children(block_id) if block['type'] != target_type else []
        matching = [c for c in children if c['type'] == target_type]
        if matching:
            return matching[0]['text']
        if block['type'] == target_type:
            return block['text']
        return None

    def _format_tree(self, node: dict) -> str:
        """
        Format a block tree as a string for deep-tier injection.
        
        Deep tier returns the full hierarchy: parent text + all children.
        Code blocks get pipe-prefixed (|), other content gets dash (-).
        Each child is truncated to 200 characters for prompt economy.
        
        Args:
            node: Dict with 'text' and 'children' keys (from get_block_tree).
            
        Returns:
            Formatted multi-line string.
        """
        lines = [node['text']]
        for c in node.get('children', []):
            ctype = c.get('type', 'paragraph')
            # Use different prefixes for different content types
            prefix = '  | ' if ctype == 'code' else '  - '
            lines.append(f"{prefix}{c['text'][:200]}")
        return '\n'.join(lines)
