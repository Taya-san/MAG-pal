from __future__ import annotations
"""
Hierarchical memory store — SQLite-backed block/sentence storage.

This module provides two classes:
  - MemoryStore: SQLite CRUD for blocks, sentences, and block_relations.
    Blocks are hierarchical (parent paragraph owns child code/table/list).
    Keywords (top_words) are stored per-block for retrieval.

  - BlockParser: parses markdown text into the above structure.
    Detects code blocks (```), equations ($$), tables (|...|),
    list items (1. / -), and paragraphs.

Used by StreamIntervention for mid-reasoning memory injection.
The LDA classifier (loaded via pickle) flags "definitional" sentences.
"""

import json
import pickle
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
# Docs: https://numpy.org/doc/
# Docs: https://github.com/cjhutto/vaderSentiment
#        https://docs.python.org/3/library/sqlite3.html
#        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

from text_utils import (
    extract_top_words,
    merge_top_words,
    detect_bare_reference,
    inject_type_keywords,
    filter_injected_keywords,
)
from stopwords import STOPWORDS

__all__ = ["MemoryStore", "BlockParser"]

# Split text into sentences at punctuation + capital letter boundaries
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(\[])')


class MemoryStore:
    """
    SQLite-backed memory with blocks, sentences, and hierarchical relations.
    
    Memory is structured as a TREE:
      B1: "Key properties of symmetric matrices:"  (paragraph)
        └── B2: "1. A = A^T"                        (list item, as sentence)
        └── B3: The actual code block               (code, as child block)
    
    Each block has:
      - text: The raw content
      - type: paragraph/code/table/list/equation
      - top_words: JSON list of extracted keywords
      - owner_id: Discord user ID (NULL = public)
    
    Sentences are individual lines/statements within a block.
    block_relations link parent blocks to their children.
    """

    def __init__(self, db_path=None, lda_path=None):
        """Connect to SQLite, create tables, optionally load LDA model."""
        self.db_path = db_path or str(Path(__file__).parent / "memory.db")
        # sqlite3.connect() opens a connection to a SQLite database file.
        # If the file doesn't exist, SQLite creates it.
        # sqlite3.Row enables column access by name: row['text'] instead of row[0]
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.lda = None
        if lda_path:
            self.load_lda(lda_path)
        # _create_tables runs CREATE TABLE IF NOT EXISTS for all tables
        self._create_tables()

    def _create_tables(self):
        """
        Create the database schema if it doesn't exist.
        
        Three tables:
          blocks: each markdown block (paragraph, code, table, list, equation)
          sentences: individual sentences/lines within blocks
          block_relations: parent-child links between blocks (for tree structure)
        
        executescript() runs multiple SQL statements separated by semicolons.
        CREATE TABLE IF NOT EXISTS is safe to call every time — it won't
        overwrite existing tables.
        """
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'paragraph',
                parent_id INTEGER REFERENCES blocks(id),
                top_words TEXT DEFAULT '[]',
                owner_id INTEGER,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                block_id INTEGER REFERENCES blocks(id),
                line_number INTEGER DEFAULT 0,
                type TEXT NOT NULL DEFAULT 'sentence',
                embedding BLOB,
                score REAL DEFAULT 0.0,
                label TEXT DEFAULT 'unlabeled',
                owner_id INTEGER,
                created_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                expires_at REAL,
                permanent INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS block_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER REFERENCES blocks(id),
                child_id INTEGER REFERENCES blocks(id),
                relation_type TEXT DEFAULT 'child_of'
            );

            CREATE INDEX IF NOT EXISTS idx_sentences_block ON sentences(block_id);
            CREATE INDEX IF NOT EXISTS idx_blocks_parent ON blocks(parent_id);
            CREATE INDEX IF NOT EXISTS idx_relations_parent ON block_relations(parent_id);
            CREATE INDEX IF NOT EXISTS idx_relations_child ON block_relations(child_id);
        """)
        self.conn.commit()
        # _migrate() handles adding columns to existing databases
        self._migrate()

    def _migrate(self):
        """
        Add columns to existing databases for schema upgrades.
        
        When we add a new column, existing database files
        don't have it. ALTER TABLE adds the new column. If it already
        exists, sqlite3 raises OperationalError which we catch silently.
        """
        for table in ('blocks', 'sentences'):
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN owner_id INTEGER")
                self.conn.commit()
            except sqlite3.OperationalError:
                pass
        
        # Memory decay columns
        for col in ('access_count INTEGER DEFAULT 0', 'expires_at REAL', 'permanent INTEGER DEFAULT 0'):
            try:
                self.conn.execute(f"ALTER TABLE sentences ADD COLUMN {col}")
                self.conn.commit()
            except sqlite3.OperationalError:
                pass

    # ===== INSERT METHODS =====
    # These create new blocks, sentences, and relations in the database.

    def _block_richness(self, text, block_type):
        """Estimate how information-rich a block's text is."""
        if block_type == 'code':
            return len(text.split('\n'))
        return len(text.split())

    # ===== MEMORY DECAY =====
    # Sentences expire over time. TTL is based on:
    #   - Content length (longer = more important)
    #   - Keyword uniqueness at block level (rarer keywords = more important)
    #   - Access frequency (more matched = lives longer)
    # Expired sentences are deleted with their entire block subtree.

    BASE_TTL = 7 * 24 * 3600  # 7 days default

    def _compute_keyword_uniqueness(self, block_id, owner_id):
        """Score how unique this block's keywords are (0.0-1.0)."""
        cur = self.conn.cursor()
        cur.execute("SELECT top_words FROM blocks WHERE id = ?", (block_id,))
        row = cur.fetchone()
        if not row:
            return 0.5
        top_words = json.loads(row[0])
        if not top_words:
            return 0.5

        scores = []
        for word in top_words:
            cur.execute(
                "SELECT COUNT(*) FROM blocks "
                "WHERE (owner_id IS NULL OR owner_id = ?) AND top_words LIKE ?",
                (owner_id, f'%"{word}"%')
            )
            count = cur.fetchone()[0]
            scores.append(1.0 / max(count, 1))

        return sum(scores) / len(scores)

    def _compute_initial_ttl(self, text, block_id, owner_id):
        """Compute initial TTL for a new sentence."""
        word_bonus = min(len(text.split()) * 3600, 7 * 24 * 3600)
        uniqueness = self._compute_keyword_uniqueness(block_id, owner_id)
        uniqueness_bonus = uniqueness * 14 * 24 * 3600
        return self.BASE_TTL + word_bonus + uniqueness_bonus

    def _extend_sentence_ttl(self, sentence_ids):
        """Extend TTL for accessed sentences. More accesses = longer extension."""
        if not sentence_ids:
            return
        cur = self.conn.cursor()
        now = time.time()
        for sid in sentence_ids:
            cur.execute("SELECT access_count, expires_at FROM sentences WHERE id = ?", (sid,))
            row = cur.fetchone()
            if not row:
                continue
            ac = row[0] + 1
            old_expires = row[1]
            if old_expires is None:
                continue  # permanent
            extension = 86400 * min(ac, 30)
            new_expires = max(old_expires, now) + extension
            cur.execute(
                "UPDATE sentences SET access_count = ?, expires_at = ? WHERE id = ?",
                (ac, new_expires, sid)
            )
        self.conn.commit()

    def apply_memory_decay(self):
        """Delete expired sentences + their entire block subtrees."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, block_id FROM sentences WHERE permanent = 0 AND expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),)
        )
        expired = cur.fetchall()
        if not expired:
            return 0, 0

        expired_ids = [r[0] for r in expired]
        affected_blocks = set(r[1] for r in expired)

        cur.execute(
            f"DELETE FROM sentences WHERE id IN ({','.join('?' * len(expired_ids))})",
            expired_ids
        )
        deleted = cur.rowcount

        for bid in affected_blocks:
            cur.execute(
                "SELECT COUNT(*) FROM sentences WHERE block_id = ?",
                (bid,)
            )
            if cur.fetchone()[0] == 0:
                self._delete_block_tree(bid)

        self.conn.commit()
        return deleted

    def _delete_block_tree(self, block_id):
        """Delete a block and all its child blocks recursively."""
        cur = self.conn.cursor()
        cur.execute("SELECT child_id FROM block_relations WHERE parent_id = ?", (block_id,))
        for row in cur.fetchall():
            self._delete_block_tree(row[0])
        cur.execute("DELETE FROM block_relations WHERE parent_id = ? OR child_id = ?",
                   (block_id, block_id))
        cur.execute("DELETE FROM sentences WHERE block_id = ?", (block_id,))
        cur.execute("DELETE FROM blocks WHERE id = ?", (block_id,))

    def find_similar_block(self, text, block_type, owner_id, threshold=0.5):
        """
        Check if a similar block already exists, comparing by keyword overlap.
        
        Uses Jaccard similarity on top_words (ignoring structural type names).
        Returns (existing_id, richness_score) if a match is found, else None.
        
        This is a simple dedup — refine later with embeddings or LLM-based
        comparison for better semantic matching.
        """
        new_words = set(
            w for w in extract_top_words(text, content_type=block_type)
            if w not in ('code', 'equation', 'table', 'list')
        )
        if not new_words:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, text FROM blocks "
                "WHERE type = ? AND text = ? AND (owner_id IS NULL OR owner_id = ?)",
                (block_type, text.strip(), owner_id,)
            )
            row = cur.fetchone()
            if row is not None:
                return (row[0], self._block_richness(row[1], block_type))
            return None

        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, text, top_words FROM blocks "
            "WHERE type = ? AND (owner_id IS NULL OR owner_id = ?) "
            "ORDER BY id DESC",
            (block_type, owner_id,)
        )

        best = None
        best_score = threshold

        for row in cur.fetchall():
            try:
                existing_words = set(json.loads(row[2]))
            except Exception:
                continue
            existing_words -= {'code', 'equation', 'table', 'list'}
            if not existing_words:
                continue

            intersection = new_words & existing_words
            score = len(intersection) / min(len(new_words), len(existing_words)) if min(new_words, existing_words) else 0

            if score >= best_score:
                best_score = score
                best = (row[0], self._block_richness(row[1], block_type))

        return best

    def _handle_structural_dedup(self, existing_id, text, top_words,
                                     new_richness, existing_richness):
        """Replace a code/eq/table block if the new text is richer."""
        if new_richness <= existing_richness:
            return existing_id, False
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE blocks SET text = ?, top_words = ?, created_at = ? WHERE id = ?",
            (text.strip(), json.dumps(top_words), time.time(), existing_id)
        )
        self.conn.commit()
        return existing_id, False

    def _handle_paragraph_dedup(self, existing_id, text, top_words):
        """Merge new unique sentences into a paragraph/list block."""
        cur = self.conn.cursor()
        cur.execute("SELECT text, top_words FROM blocks WHERE id = ?", (existing_id,))
        old_row = cur.fetchone()
        if old_row is None:
            return existing_id, False

        old_text, old_top_words_json = old_row
        old_top_words = json.loads(old_top_words_json) if old_top_words_json else []

        old_sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(old_text) if len(s.strip()) > 3]
        new_sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if len(s.strip()) > 3]

        def _content_words(sent):
            return set(
                w for w in re.sub(r'[^\w\s]', ' ', sent.lower()).split()
                if w not in STOPWORDS and len(w) > 2
            )

        added = []
        for new_sent in new_sents:
            new_content = _content_words(new_sent)
            if not new_content:
                added.append(new_sent)
                continue

            all_old = set()
            for old_sent in old_sents:
                all_old |= _content_words(old_sent)

            if new_content - all_old:
                added.append(new_sent)

        if not added:
            return existing_id, False

        merged_text = old_text + ' ' + ' '.join(added)
        merged_top_words = list(dict.fromkeys(old_top_words + top_words))[:12]
        cur.execute(
            "UPDATE blocks SET text = ?, top_words = ?, created_at = ? WHERE id = ?",
            (merged_text.strip(), json.dumps(merged_top_words), time.time(), existing_id)
        )
        self.conn.commit()
        return existing_id, False

    def add_block(self, text, block_type='paragraph', parent_id=None, owner_id=None):
        """
        Insert a new block into the database, with dedup.
        
        Checks for similar existing blocks by keyword overlap. If found:
          - code/eq/table: replace if richer, otherwise skip.
          - paragraph/list: merge new unique sentences into the existing block.
        If no similar block exists, inserts a new one.
        
        Returns:
            (block_id, is_new) tuple.
        """
        top_words = extract_top_words(text, content_type=block_type)
        structural_types = {'code', 'table', 'list', 'equation'}
        if block_type in structural_types and block_type not in top_words:
            top_words.append(block_type)

        existing = self.find_similar_block(text, block_type, owner_id)
        if existing is not None:
            existing_id, existing_richness = existing
            new_richness = self._block_richness(text, block_type)

            if block_type in ('code', 'equation', 'table'):
                return self._handle_structural_dedup(
                    existing_id, text, top_words, new_richness, existing_richness
                )
            return self._handle_paragraph_dedup(existing_id, text, top_words)

        top_words = json.dumps(top_words)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO blocks (text, type, parent_id, top_words, owner_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (text.strip(), block_type, parent_id, top_words, owner_id, time.time())
        )
        self.conn.commit()
        return cur.lastrowid, True

    def add_sentence(self, text, block_id, line_number=0, sent_type='sentence',
                     embedding=None, score=0.0, label='unlabeled', strip=True,
                     owner_id=None):
        """
        Insert a sentence linked to a parent block.
        
        Each block contains multiple sentences. A code block might have
        several code_line sentences. A table has table_row sentences.
        A paragraph has sentence-type sentences.
        
        Args:
            text: Sentence text.
            block_id: FK to the parent block.
            line_number: Position within the block (for ordering).
            sent_type: 'sentence', 'code_line', 'table_row', 'list_item', 'equation'.
            embedding: Optional numpy float32 vector (for LDA). Stored as BLOB.
            score: LDA classifier score (positive = definitional/flagged).
            label: 'flagged' or 'unlabeled' (set by LDA classifier).
            strip: Whether to strip whitespace (False for code lines, which
                   preserve indentation).
            owner_id: Discord user ID for privacy filtering.
        """
        text_content = text.strip() if strip else text
        # Convert numpy array to binary bytes for SQLite BLOB storage
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        cur = self.conn.cursor()
        # Compute initial TTL for memory decay
        ttl = self._compute_initial_ttl(text_content, block_id, owner_id)
        expires_at = time.time() + ttl
        cur.execute(
            "INSERT INTO sentences (text, block_id, line_number, type, embedding, score, label, owner_id, created_at, access_count, expires_at, permanent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0)",
            (text_content, block_id, line_number, sent_type, emb_blob, float(score), label, owner_id, time.time(), expires_at)
        )
        self.conn.commit()
        return cur.lastrowid

    def add_relation(self, parent_id, child_id, relation_type='child_of'):
        """
        Link a child block to its parent block.
        
        This creates a TREE structure. For example:
          Parent (paragraph): "Key properties:"
          Child (table): | Property | Value |
        
        block_relations connects these. When querying, we can get
        "give me all children of block X" or "who is block Y's parent?"
        """
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO block_relations (parent_id, child_id, relation_type) VALUES (?, ?, ?)",
            (parent_id, child_id, relation_type)
        )
        self.conn.commit()
        return cur.lastrowid

    # ===== QUERY METHODS =====
    # These fetch data from the database for retrieval and intervention.

    def get_block(self, block_id):
        """Fetch a single block by its ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
        return cur.fetchone()

    def get_block_children(self, block_id):
        """Fetch all child blocks for a parent block, ordered by ID."""
        # JOIN block_relations to find children, then get the blocks
        cur = self.conn.cursor()
        cur.execute("""
            SELECT b.* FROM blocks b
            JOIN block_relations r ON b.id = r.child_id
            WHERE r.parent_id = ? AND r.relation_type = 'child_of'
            ORDER BY b.id
        """, (block_id,))
        return cur.fetchall()

    def get_sentences_by_block(self, block_id):
        """Fetch all sentences in a block, ordered by line_number."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM sentences WHERE block_id = ? ORDER BY line_number",
            (block_id,)
        )
        return cur.fetchall()

    def get_tree(self, block_id=None):
        """
        Fetch root blocks (or a specific block) with their children.
        
        This returns the hierarchical tree structure. If block_id is None,
        returns all root blocks (parent_id IS NULL). Otherwise returns
        the specified block. Each node has a 'children' key with sub-blocks.
        
        Returns:
            List of dicts with block data + 'children' list.
        """
        cur = self.conn.cursor()
        if block_id is None:
            cur.execute("SELECT * FROM blocks WHERE parent_id IS NULL ORDER BY id")
        else:
            cur.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
        roots = cur.fetchall()

        result = []
        for root in roots:
            node = dict(root)
            node['children'] = self.get_block_children(root['id'])
            result.append(node)
        return result

    def load_lda(self, lda_path):
        """Load a pre-trained LDA classifier model from a pickle file."""
        with open(lda_path, 'rb') as f:
            self.lda = pickle.load(f)

    def count_stats(self):
        """Return dict with counts of sentences, blocks, flagged, and relations."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sentences")
        s = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM blocks")
        b = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sentences WHERE label != 'unlabeled'")
        fl = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM block_relations")
        r = cur.fetchone()[0]
        return {'sentences': s, 'blocks': b, 'flagged': fl, 'relations': r}

    def close(self):
        """Close the database connection. Must be called on shutdown."""
        self.conn.close()

    # ===== INTERVENTION SYSTEM METHODS =====
    # These are used by StreamIntervention for keyword indexing and
    # context retrieval.

    # Keywords that identify structural element types.
    # Used by build_match_index() to decide whether a keyword like "code"
    # refers to an actual structural element or just a passing mention.
    # Only indexed when the block (or its children) match the type.
    STRUCTURAL_NAMES = {'table', 'tables', 'code', 'list', 'lists', 'equation', 'equations'}

    def _block_matches_structural(self, word, block_id):
        """
        Check if a structural keyword matches this block or its children.
        
        A word like "code" should only be indexed if the block itself is
        a code block OR any of its children are code blocks. This prevents
        false matches where "code" was injected into a paragraph that
        doesn't actually have code-related content.
        
        Example:
          Block: "gradient descent:" (paragraph type)
          Child: ```python code block``` (code type)
          -> _block_matches_structural("code", block_id) returns True
             because the paragraph has a code child.
        """
        wl = word.lower()
        block = self.get_block(block_id)
        if not block:
            return False
        if self._type_matches_structural(wl, block['type']):
            return True
        for child in self.get_block_children(block_id):
            if self._type_matches_structural(wl, child['type']):
                return True
        return False

    def _type_matches_structural(self, wl, block_type):
        """Match a lowercase structural word to a canonical block type."""
        return (
            (wl in ('table', 'tables') and block_type == 'table') or
            (wl == 'code' and block_type == 'code') or
            (wl in ('list', 'lists') and block_type == 'list') or
            (wl in ('equation', 'equations') and block_type == 'equation')
        )

    def build_match_index(self, owner_id=None):
        """
        Build the inverted keyword index for the intervention system.
        
        This is the KEY method for the streaming intervention system.
        It queries ALL flagged sentences, joins with their blocks to
        get top_words, and builds:
          {lowercase_keyword: [{sentence_id, text, block_id, block_type}]}
        
        Filtering:
          - owner_id: Filters to (owner_id IS NULL) OR (owner_id = ?).
            NULL = public/shared, ? = the current user.
          - Structural words: pass through _block_matches_structural().
            Only indexed if the block or its children match that type.
          - Generic injected keywords: stripped via filter_injected_keywords()
            to prevent "function", "example", "row" from being matchable.
        
        Returns:
            dict: {word: [{sentence_id, text, block_id, block_type}]}
        
        Note: apply_memory_decay() runs first to remove expired entries.
        """
        self.apply_memory_decay()
        cur = self.conn.cursor()
        if owner_id is not None:
            cur.execute("""
                SELECT s.id, s.text, s.block_id, b.top_words, b.type
                FROM sentences s
                JOIN blocks b ON s.block_id = b.id
                WHERE s.label = 'flagged'
                  AND (b.owner_id IS NULL OR b.owner_id = ?)
            """, (owner_id,))
        else:
            cur.execute("""
                SELECT s.id, s.text, s.block_id, b.top_words, b.type
                FROM sentences s
                JOIN blocks b ON s.block_id = b.id
                WHERE s.label = 'flagged'
            """)
        index = {}
        for row in cur.fetchall():
            top_words = json.loads(row['top_words'])
            block_type = row['type']
            block_id = row['block_id']

            clean_words = filter_injected_keywords(top_words)
            for w in top_words:
                wl = w.lower()
                is_structural = wl in self.STRUCTURAL_NAMES
                if is_structural:
                    # Only index if block or its children match the type
                    if not self._block_matches_structural(wl, block_id):
                        continue
                elif wl not in clean_words:
                    # Skip generic injected keywords
                    continue

                if wl not in index:
                    index[wl] = []
                index[wl].append({
                    'sentence_id': row['id'],
                    'text': row['text'],
                    'block_id': block_id,
                    'block_type': block_type,
                })
        return index

    def get_block_tree(self, block_id):
        """
        Return a block and its children (full tree) for deep-tier injection.
        
        Used by StreamIntervention when ratio >= DEEP_THRESHOLD (0.7).
        Returns both the block's own text and all its child blocks,
        so the AI gets the full context of the memory.
        """
        block = self.get_block(block_id)
        if not block:
            return None
        node = dict(block)
        node['children'] = [dict(c) for c in self.get_block_children(block_id)]
        return node

# =============================================================================
# BlockParser — parses markdown text into MemoryStore blocks
# =============================================================================
# This is a single-pass state machine that walks through text line by line
# and detects structural boundaries:
#
# 1. Empty lines: Reset block for structural types, keep for paragraphs
# 2. Code blocks: Detect ``` fences, create code block + code_line sentences
# 3. Equations: Detect $$ delimiters, create equation block + equation sentences
# 4. Lists: Detect "1. " or "- " or "* " lines, create list block + list_item sentences
# 5. Tables: Detect |...| lines, create table block + table_row sentences
# 6. Paragraphs: Everything else — split into sentences, merge consecutive paragraphs

from block_parser import BlockParser
