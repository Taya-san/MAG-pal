"""
Stream Intervention — real-time memory injection during AI reasoning.

During streaming generation, the bot scans the model's chain-of-thought
reasoning tokens for keywords that match previously-flagged "interesting"
sentences from past conversations stored in the MemoryStore.

When a match fires, the bot interrupts the stream, re-prompts the AI with
the recalled memory as context, and resumes generation — all before the user
sees any output. This is designed to boost small models (8B-35B) by giving
them access to hierarchical long-term memory during reasoning.

How it works:

1. BUILD PHASE (once):
   build_index() queries MemoryStore for all flagged sentences,
   builds an inverted index: {lowercase_word: [sentence_id, ...]}.
   Structural type words ("code", "table", etc.) are indexed only
   when the block or its children actually match that type.

2. SCAN PHASE (per reasoning token):
   buffer_token() accumulates sub-word streaming tokens into
   complete words using a regex boundary detector. Completed
   words are emitted and checked against the keyword index.

3. MATCH PHASE (per completed word):
   check_match() looks up the word in the keyword index, fetches
   the sentence's block-level top_words, filters out generic
   injected keywords, computes a match ratio, and decides the tier:
     - deep (≥70%): returns full block tree
     - specific (structural word): returns matching child block
     - surface (≥1 word): returns parent paragraph

4. COOLDOWN PHASE:
   Each injected sentence gets a 150-token cooldown, preventing
   re-injection within the same response. Cooldowns decrement
   per reasoning token and decay by 50% of output length.
"""

import json
import re
import logging
from text_utils import filter_injected_keywords

logger = logging.getLogger("palbot")

# === CONSTANTS ===

COOLDOWN_LIMIT = 150           # reasoning tokens before a sentence can trigger again
OUTPUT_DECAY_RATIO = 0.5       # fraction of output length applied as extra cooldown decay
DEEP_THRESHOLD = 0.7           # fraction of top_words that must match for deep tier
WORD_MIN_LEN = 3               # minimum characters for a word to be considered

# Structural type names that trigger the "specific" tier.
# When the AI mentions one of these words in reasoning, and a matching
# block exists, the specific child block is injected directly.
STRUCTURAL_NAMES = {'table', 'tables', 'code', 'list', 'lists', 'equation', 'equations'}

# Maps plural/alias structural words to canonical block types.
# Used by _get_child_block() to find the right child block.
TYPE_MAP = {
    'table': 'table', 'tables': 'table',
    'code': 'code',
    'list': 'list', 'lists': 'list',
    'equation': 'equation', 'equations': 'equation',
}


class StreamIntervention:
    """
    Per-call intervention engine for scanning streaming reasoning tokens.
    
    Instances are created fresh for each _stream_with_intervention() call
    to provide per-call isolation of cooldowns and word buffer. The
    keyword_index is shared via reference from a master instance so
    all calls share the same matchable keyword set.
    
    Args:
        memory_store: MemoryStore instance for DB queries (optional if shared_index given).
        shared_index: Pre-built keyword_index dict to share across calls.
        owner_id: Discord user ID for filtering flagged sentences by owner.
    """
    
    def __init__(self, memory_store=None, shared_index=None, owner_id=None):
        self.store = memory_store
        self.owner_id = owner_id
        
        # If shared_index is provided, use it directly (per-call instance
        # shares the master's index). Otherwise, build from store.
        if shared_index is not None:
            self.keyword_index = shared_index
        else:
            self.keyword_index = {}
            
        # Per-call mutable state — each concurrent stream gets its own
        self.cooldowns = {}      # {sentence_id: remaining_tokens}
        self.word_buffer = ""    # accumulates sub-word tokens into complete words
        
        # Only build index if we have a store and no shared index
        if shared_index is None and memory_store is not None:
            self.build_index()

    # ===== INDEX BUILDING =====

    def build_index(self):
        """
        Build the inverted keyword index from MemoryStore.
        
        Calls build_match_index() which queries all flagged sentences,
        filters by owner_id (if set), and returns a dict of
        {lowercase_word: [{sentence_id, text, block_id, block_type}]}.
        
        The index maps every content keyword from every flagged block
        to every sentence in that block, so a single keyword can point
        to multiple sentences from the same block.
        """
        if not self.store:
            self.keyword_index = {}
            return
        self.keyword_index = self.store.build_match_index(self.owner_id)
        logger.info(f"Built intervention index: {len(self.keyword_index)} keywords")

    def rebuild_index(self):
        """
        Rebuild the index in-place so shared references stay valid.
        
        Instead of assigning a new dict (which would break shared
        references in per-call instances), we clear and update
        the existing dict object in-place.
        """
        if not self.store:
            return
        new_index = self.store.build_match_index(self.owner_id)
        self.keyword_index.clear()
        self.keyword_index.update(new_index)
        logger.info(f"Rebuilt intervention index: {len(self.keyword_index)} keywords")

    # ===== WORD BUFFERING =====

    def buffer_token(self, token):
        """
        Accumulate a streaming sub-word token and emit completed words.
        
        Streaming APIs send sub-word tokens like "s", "ym", "metric".
        This method appends each token to an internal buffer and uses
        a regex to detect word boundaries (non-word characters like
        spaces, punctuation, newlines).
        
        The regex pattern: (?:^|\\W)([a-zA-Z]{3,})\\W
        - (?:^|\\W): start of buffer or a non-word char before the word
        - ([a-zA-Z]{3,}): the word (minimum 3 letters)
        - \\W: a non-word char after the word (space, punctuation)
        
        A word is only emitted when followed by a boundary character —
        this prevents premature extraction of partial words.
        
        Returns:
            List of completed words (may be empty if no boundary found).
        """
        self.word_buffer += token
        emitted = []
        while True:
            m = re.search(r'(?:^|\W)([a-zA-Z]{3,})\W', self.word_buffer)
            if not m:
                break
            word = m.group(1).lower()
            self.word_buffer = self.word_buffer[m.end():]
            emitted.append(word)
        return emitted

    @staticmethod
    def _buf_extract(buffer):
        """
        Static utility: extract completed words from a buffer string.
        
        Same regex logic as buffer_token() but operates on a passed-in
        string instead of the instance buffer. Returns (words, remaining).
        Used for testing and debugging.
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

    def flush_buffer(self):
        """
        Extract any trailing word left in the buffer at stream end.
        
        Unlike buffer_token(), this uses a different regex anchored
        to the end of the buffer ($). It extracts a word of 3+ letters
        that has no trailing boundary character (stream ended mid-token).
        
        Returns:
            List containing the single trailing word, or empty list.
        """
        m = re.match(r'(?:^|\W)?([a-zA-Z]{3,})$', self.word_buffer)
        if not m:
            return []
        self.word_buffer = ""
        return [m.group(1).lower()]

    def reset_buffer(self):
        """Clear the word buffer — called in finally block for cleanup."""
        self.word_buffer = ""

    # ===== MATCHING =====

    def check_match(self, word, reasoning_tokens):
        """
        Check if a completed word triggers a memory intervention.
        
        Algorithm:
        1. Look up the word in the keyword_index dict.
        2. For each matching sentence (skipping those on cooldown):
           a. Fetch the sentence's block-level top_words from the DB.
           b. Filter out generic injected keywords (function, example, etc.)
              via filter_injected_keywords() to get clean content words.
           c. Scan the last N reasoning tokens for matches against clean_top.
           d. Compute ratio = matched_words / len(clean_top).
        3. Decide the tier:
           - deep: ratio >= DEEP_THRESHOLD (0.7)
           - specific: word is a STRUCTURAL_NAME and child block exists
           - surface: default, returns parent paragraph text
        4. Sort candidates by ratio descending, pick the best one.
        5. Set cooldown on the winning sentence_id.
        
        Args:
            word: Lowercase completed word to check.
            reasoning_tokens: List of recent raw reasoning tokens for
                             cross-checking other words in the same block.
        
        Returns:
            dict with tier, sentence_id, context, matched_word, ratio
            or None if no match found.
        """
        if len(word) < WORD_MIN_LEN or not self.keyword_index:
            return None
        matches = self.keyword_index.get(word)
        if not matches:
            return None

        candidates = []
        for item in matches:
            sid = item['sentence_id']
            # Skip sentences on cooldown (recently injected)
            if self.cooldowns.get(sid, 0) > 0:
                continue

            # Fetch the block's aggregated keywords from DB
            top_words_list = self._get_sentence_top_words(sid)
            if not top_words_list:
                continue

            # Strip generic type keywords for ratio calculation.
            # We only count content words (eigenvalues, matrix, gradient),
            # not structural descriptors (function, example, property).
            clean_top = filter_injected_keywords(top_words_list)
            if not clean_top:
                continue

            # Count how many of the block's content keywords appear
            # in the recent reasoning tokens.
            matched_words = {word}
            for raw in reasoning_tokens:
                wc = re.sub(r'[^a-z]', '', raw.lower())
                if wc in clean_top:
                    matched_words.add(wc)

            ratio = len(matched_words) / len(clean_top)
            is_structural = word.lower() in STRUCTURAL_NAMES

            context = None
            tier = None

            # --- Tier 1: Deep (≥70% overlap) — return full block tree ---
            if ratio >= DEEP_THRESHOLD:
                tree = self.store.get_block_tree(item['block_id']) if self.store else None
                context = self._format_tree(tree) if tree else item['text']
                tier = 'deep'

            # --- Tier 2: Specific (structural word) — return matching child ---
            elif is_structural:
                context = self._get_child_block(item['block_id'], word) if self.store else item['text']
                tier = 'specific' if context else None

            # --- Tier 3: Surface (≥1 word) — return parent paragraph ---
            else:
                parent = self._get_parent_text(item['block_id']) if self.store else item['text']
                context = parent or item['text']
                tier = 'surface'

            if context and tier:
                candidates.append((ratio, tier, sid, context, word))

        if not candidates:
            return None

        # Pick the highest-ratio match (best candidate)
        candidates.sort(key=lambda x: -x[0])
        best_ratio, best_tier, best_sid, best_context, best_word = candidates[0]
        
        # Lock this sentence for COOLDOWN_LIMIT reasoning tokens
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

    def decrement_cooldowns(self, n=1):
        """
        Reduce all active cooldowns by n tokens.
        
        Called once per reasoning token. When a cooldown hits 0 or below,
        the sentence becomes eligible for re-injection.
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

    def apply_output_decay(self, output_length):
        """
        Accelerate cooldown expiration after stream completion.
        
        The output word count serves as a proxy for "how far past
        the injected memory are we now?" — reducing cooldowns by
        50% of the output length means previously-injected sentences
        become available sooner for the next message.
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

    # ===== DB HELPERS =====

    def _get_sentence_top_words(self, sentence_id):
        """
        Fetch the block-level top_words for a sentence.
        
        The top_words are stored on the block, not the sentence —
        they represent the aggregated keywords of the entire block
        (code block, table, paragraph) that this sentence belongs to.
        
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

    def _get_parent_text(self, block_id):
        """
        Get the parent block's text for a child block.
        
        For surface-tier matches, we return the paragraph that
        introduced this code/table/list/equation — the parent's
        text that provides context.
        
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

    def _get_child_block(self, block_id, structural_word):
        """
        Get the text of a specific child block by type.
        
        For specific-tier matches, the AI mentioned a structural
        type name ("table", "code"). We return that specific child
        block's text. If the block itself is the target type, we
        return its own text.
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

    def _format_tree(self, node):
        """
        Format a block tree as text for deep-tier injection.
        
        Deep tier returns the full hierarchy: parent text + all children.
        Code blocks get pipe-prefixed lines (|), other content gets
        bullet-prefixed lines (-). Truncated to 200 chars per child.
        """
        lines = [node['text']]
        for c in node.get('children', []):
            ctype = c.get('type', 'paragraph')
            prefix = '  | ' if ctype == 'code' else '  - '
            lines.append(f"{prefix}{c['text'][:200]}")
        return '\n'.join(lines)
