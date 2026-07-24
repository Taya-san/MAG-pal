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
                created_at REAL NOT NULL
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
        
        When we add a new column (like owner_id), existing database files
        don't have it. ALTER TABLE adds the new column. If it already
        exists, sqlite3 raises OperationalError which we catch silently.
        """
        for table in ('blocks', 'sentences'):
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN owner_id INTEGER")
                self.conn.commit()
            except sqlite3.OperationalError:
                # Column already exists - that's fine
                pass

    # ===== INSERT METHODS =====
    # These create new blocks, sentences, and relations in the database.

    def add_block(self, text, block_type='paragraph', parent_id=None, owner_id=None):
        """
        Insert a new block into the database.
        
        A block represents a structural unit: a paragraph, a code block,
        a table, a list, or an equation. Each block gets:
          - auto-generated ID (sqlite AUTOINCREMENT)
          - top_words extracted from its text
          - sentiment scored by VADER
          - owner_id for privacy filtering
        
        Structural Type Seeding:
          For code/table/list/equation blocks, the type name is appended
          to top_words. This is necessary because a code block's text
          (like "x = 1\n") doesn't contain the word "code" — but we want
          "code" to be a searchable keyword so the intervention system can
          match when the AI says "the code looks like...".
        
        Args:
            text: Block content (markdown text).
            block_type: 'paragraph', 'code', 'table', 'list', 'equation'.
            parent_id: FK to parent block (None = root block).
            owner_id: Discord user ID (None = public, visible to all).
        
        Returns:
            Auto-increment ID of the new block.
        """
        top_words = extract_top_words(text, content_type=block_type)
        
        # Seed structural type name for retrieval
        structural_types = {'code', 'table', 'list', 'equation'}
        if block_type in structural_types and block_type not in top_words:
            top_words.append(block_type)
            top_words = top_words[:12]  # cap at 12 keywords
        top_words = json.dumps(top_words)
        
        
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO blocks (text, type, parent_id, top_words, owner_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (text.strip(), block_type, parent_id, top_words, owner_id, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

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
        cur.execute(
            "INSERT INTO sentences (text, block_id, line_number, type, embedding, score, label, owner_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (text_content, block_id, line_number, sent_type, emb_blob, float(score), label, owner_id, time.time())
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
        """
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

class BlockParser:
    """
    Line-by-line markdown parser that stores hierarchical blocks into MemoryStore.
    
    Parse order (first match wins):
      1. Code blocks (lines starting with ```)
      2. Equations (lines starting with $$)
      3. List items (numbered/bulleted lines)
      4. Tables (lines starting and ending with |)
      5. Paragraphs (everything else)
    
    After parsing, _propagate_top_words() runs to:
      - Merge parent keywords into child blocks
      - Detect "bare" parents and inject structural type keywords
      - Seed type names on code/table/list/equation child blocks
    
    If embed_fn and LDA model are available, _classify_sentences() then
    scores each sentence and labels it 'flagged' or 'unlabeled'.
    """
    
    def __init__(self, memory_store, embed_fn=None):
        self.store = memory_store
        self.embed = embed_fn

    def parse_and_store(self, text, owner_id=None):
        """
        Parse markdown text into hierarchical blocks and store in DB.
        
        Args:
            text: Raw markdown text (typically an AI response).
            owner_id: Discord user ID for privacy filtering.
        
        Returns:
            (list_of_block_ids, list_of_sentence_ids)
        """
        all_block_ids = []
        all_sentence_ids = []
        lines = text.split('\n')
        i = 0
        current_block_id = None   # active block being built
        last_block_type = None     # type of the last closed block
        
        while i < len(lines):
            stripped = lines[i].strip()

            # ---- Empty lines ----
            if not stripped:
                # Empty lines reset block scope for structural types
                if last_block_type in ('table', 'code', 'list', 'equation'):
                    current_block_id = None
                # For paragraphs: keep current_block_id so consecutive
                # paragraphs merge into one block
                i += 1
                continue

            # ---- Code block detection ----
            if stripped.startswith('```'):
                code_lines = [lines[i]]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])  # include closing ```
                    i += 1

                code_text = '\n'.join(code_lines)
                block_id = self.store.add_block(code_text, 'code', current_block_id, owner_id)
                all_block_ids.append(block_id)

                for ln, code_line in enumerate(code_lines):
                    sid = self.store.add_sentence(
                        code_line, block_id, ln, 'code_line',
                        score=0.0, label='unlabeled', strip=False, owner_id=owner_id
                    )
                    all_sentence_ids.append(sid)

                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                last_block_type = 'code'
                continue

            # ---- Equation block detection ----
            if stripped.startswith('$$'):
                eq_lines = [lines[i]]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('$$'):
                    eq_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    eq_lines.append(lines[i])
                    i += 1

                eq_text = '\n'.join(eq_lines)
                block_id = self.store.add_block(eq_text, 'equation', current_block_id, owner_id)
                all_block_ids.append(block_id)

                for ln, eql in enumerate(eq_lines):
                    sid = self.store.add_sentence(
                        eql, block_id, ln, 'equation',
                        score=0.0, label='unlabeled', owner_id=owner_id
                    )
                    all_sentence_ids.append(sid)

                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                last_block_type = 'equation'
                continue

            # ---- List item detection ----
            # Matches lines like: "1. text", "1) text", "- text", "* text"
            if re.match(r'^\s*(?:\d+[\.\)]|[-*])\s', stripped):
                if current_block_id is None:
                    block_id = self.store.add_block(stripped, 'list', None, owner_id)
                    current_block_id = block_id
                    all_block_ids.append(block_id)

                line_num = len(self.store.get_sentences_by_block(current_block_id))
                sid = self.store.add_sentence(
                    stripped, current_block_id, line_num, 'list_item',
                    score=0.0, label='unlabeled', owner_id=owner_id
                )
                all_sentence_ids.append(sid)

                # Update block text with new list item content
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + stripped if existing else stripped
                cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
                    (new_text, json.dumps(extract_top_words(new_text)), current_block_id))
                self.store.conn.commit()
                last_block_type = 'list'
                i += 1
                continue

            # ---- Table detection ----
            if stripped.startswith('|') and stripped.endswith('|'):
                table_lines = [lines[i]]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                    table_lines.append(lines[i])
                    i += 1

                table_text = '\n'.join(table_lines)
                block_id = self.store.add_block(table_text, 'table', current_block_id, owner_id)
                all_block_ids.append(block_id)

                for ln, tl in enumerate(table_lines):
                    sid = self.store.add_sentence(tl, block_id, ln, 'table_row', owner_id=owner_id)
                    all_sentence_ids.append(sid)

                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                last_block_type = 'table'
                continue

            # ---- Regular sentence (paragraph) ----
            # There are three cases:
            # A) current_block exists, line doesn't end with ":" -> append to block
            # B) current_block exists, line ends with ":" -> append, mark as parent
            # C) no current_block -> create new paragraph block
            
            if current_block_id is not None and not stripped.endswith(':'):
                # Case A: Append line to existing paragraph block
                for ln, s in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped)):
                        # Split on sentence boundaries. Regex:
                        # (?<=[.!?])  = lookbehind for punctuation ending sentence
                        # \s+         = whitespace gap
                        # (?=[A-Z"(\[]) = lookahead for next sentence's first letter
                        # This keeps punctuation attached to the first sentence.
                        # Split on sentence boundaries. Regex:
                        # (?<=[.!?])  = lookbehind for punctuation ending sentence
                        # \s+         = whitespace gap
                        # (?=[A-Z"(\[]) = lookahead for next sentence's first letter
                        # This keeps punctuation attached to the first sentence.
                    s = s.strip()
                    if not s:
                        continue
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence',
                                                   owner_id=owner_id)
                    all_sentence_ids.append(sid)

                # Update block text and top_words
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + stripped
                cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
                    (new_text, json.dumps(extract_top_words(new_text)),
                     current_block_id))
                self.store.conn.commit()
                last_block_type = 'paragraph'
                
            elif current_block_id is not None and stripped.endswith(':'):
                # Case B: Line ends with ":" — marks a parent paragraph
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped) if len(s.strip()) > 3]
                block_text = ' '.join(sents)
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + block_text
                cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
                    (new_text, json.dumps(extract_top_words(new_text)),
                     current_block_id))
                self.store.conn.commit()
                for ln, s in enumerate(sents):
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence',
                                                   owner_id=owner_id)
                    all_sentence_ids.append(sid)
                last_block_type = 'paragraph'
            else:
                # Case C: Start a new paragraph block
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped) if len(s.strip()) > 3]
                        # Split on sentence boundaries. Regex:
                        # (?<=[.!?])  = lookbehind for punctuation ending sentence
                        # \s+         = whitespace gap
                        # (?=[A-Z"(\[]) = lookahead for next sentence's first letter
                        # This keeps punctuation attached to the first sentence.
                
                block_text = ' '.join(sents)
                block_id = self.store.add_block(block_text, 'paragraph', None, owner_id)
                all_block_ids.append(block_id)
                for ln, s in enumerate(sents):
                    sid = self.store.add_sentence(s, block_id, ln, 'sentence',
                                                   owner_id=owner_id)
                    all_sentence_ids.append(sid)
                current_block_id = block_id
                last_block_type = 'paragraph'

            i += 1

        # After all blocks are parsed, propagate keywords
        self._propagate_top_words()

        # If embedding function and LDA model are available, classify sentences
        if self.embed and self.store.lda:
            self._classify_sentences(all_sentence_ids)

        return all_block_ids, all_sentence_ids

    def _propagate_top_words(self):
        """
        Propagate parent keywords to children: injection, merge, type seeding.
        
        For every parent-child relationship (from block_relations):
        
        1. BARE REFERENCE DETECTION: If parent has <3 content keywords
           (like "check this table:" after stopword filtering), it's a
           "bare reference" — a thin sentence that just introduces the
           child. inject_type_keywords() adds structural descriptors
           ("row", "column", "data") to give the parent more keywords.
        
        2. MERGE: merge_top_words() prepends parent keywords into each
           child's top_words list (deduplicated, capped at 12).
        
        3. TYPE SEEDING: Each child gets its structural type name
           appended to top_words (e.g., 'code' for a code child).
           This is separate from the seeding in add_block() which only
           applies to orphan blocks (no parent relationship).
        """
        cur = self.store.conn.cursor()
        # Get all parent-child relations with child block types
        cur.execute("""
            SELECT r.parent_id, r.child_id, b.type AS child_type
            FROM block_relations r
            JOIN blocks b ON b.id = r.child_id
            WHERE r.relation_type = 'child_of'
        """)
        parent_groups = defaultdict(list)
        for pid, cid, ctype in cur.fetchall():
            parent_groups[pid].append((cid, ctype))

        for pid, children in parent_groups.items():
            # Get parent's current top_words
            cur.execute("SELECT top_words FROM blocks WHERE id = ?", (pid,))
            row = cur.fetchone()
            if not row:
                continue
            parent_words = json.loads(row[0])
            child_types = list({ct for _, ct in children})

            # Bare reference detection: if parent has <3 keywords,
            # inject structural type descriptors
            if detect_bare_reference(parent_words):
                parent_words = inject_type_keywords(parent_words, child_types)
                cur.execute(
                    "UPDATE blocks SET top_words = ? WHERE id = ?",
                    (json.dumps(parent_words), pid)
                )

            for cid, ctype in children:
                cur.execute("SELECT top_words FROM blocks WHERE id = ?", (cid,))
                crow = cur.fetchone()
                if not crow:
                    continue
                child_words = json.loads(crow[0])
                
                # Merge parent keywords into child (preprend, dedup, cap at 12)
                merged = merge_top_words(parent_words, child_words)

                # Seed the structural type name
                type_name = ctype
                if type_name and type_name not in {w.lower() for w in merged}:
                    merged.append(type_name)
                    merged = merged[:12]

                cur.execute(
                    "UPDATE blocks SET top_words = ? WHERE id = ?",
                    (json.dumps(merged), cid)
                )
        self.store.conn.commit()

    def _classify_sentences(self, sentence_ids):
        """
        Run LDA classifier on each sentence.
        
        LDA (Linear Discriminant Analysis) is a pre-trained binary
        classifier that scores each sentence as "definitional" (positive
        score = 'flagged') or "mundane" (negative score = 'unlabeled').
        
        Code lines, equations, and table rows inherit their parent
        block's first sentence score and label — they're structural,
        not semantic, so their individual classification isn't useful.
        
        The LDA model was trained on 142 curated examples with 97.9%
        accuracy. It's loaded from models/lda_model.pkl.
        """
        cur = self.store.conn.cursor()
        for sid in sentence_ids:
            cur.execute("""
                SELECT s.text, s.type, b.id as block_id
                FROM sentences s
                JOIN blocks b ON s.block_id = b.id
                WHERE s.id = ?
            """, (sid,))
            row = cur.fetchone()
            if not row:
                continue
            text, stype, block_id = row[0], row[1], row[2]

            # Code lines, equations, and table rows inherit parent's label
            if stype in ('code_line', 'equation', 'table_row'):
                cur.execute("""
                    SELECT parent_id FROM block_relations
                    WHERE child_id = ? AND relation_type = 'child_of'
                """, (block_id,))
                parent_rel = cur.fetchone()
                if parent_rel:
                    cur.execute("""
                        SELECT id, score, label FROM sentences
                        WHERE block_id = ? AND type = 'sentence'
                        ORDER BY line_number LIMIT 1
                    """, (parent_rel[0],))
                    ps = cur.fetchone()
                    if ps:
                        cur.execute("UPDATE sentences SET score = ?, label = ? WHERE id = ?",
                            (ps[1], ps[2], sid))
                        continue
                cur.execute("UPDATE sentences SET score = 0.0, label = 'unlabeled' WHERE id = ?", (sid,))
                continue

            # Skip very short sentences (under 10 chars)
            if len(text.strip()) < 10:
                continue
                
            # Embed the sentence using the provided embedding function,
            # then score it with LDA
            emb = self.embed(text).reshape(1, -1)  # reshape to (1, 384) — LDA expects 2D input: (samples, features)
            score = float(self.store.lda.decision_function(emb)[0])
            label = 'flagged' if score > 0 else 'unlabeled'
            cur.execute(
                "UPDATE sentences SET score = ?, label = ? WHERE id = ?",
                (score, label, sid)
            )
        self.store.conn.commit()


if __name__ == "__main__":
    print("Testing MemoryStore + BlockParser...")
    store = MemoryStore(":memory:")

    parser = BlockParser(store)

    test = """A symmetric matrix equals its own transpose. Eigenvalues are always real for symmetric matrices.

Key properties of symmetric matrices:
1. A = A^T is the defining property
2. All eigenvalues are real

The spectral theorem guarantees:
A = Q * Lambda * Q^T

Implementation:
```python
def is_symmetric(A):
    return np.allclose(A, A.T)
```"""

    block_ids, sent_ids = parser.parse_and_store(test)
    stats = store.count_stats()
    print(f"\nStored: {stats['sentences']} sentences, {stats['blocks']} blocks, {stats['relations']} relations")

    print("\nBlocks tree:")
    for b in store.get_tree():
        print(f"\n  BLOCK {b['id']}: [{b['type']}] \"{b['text'][:60]}...\"")
        print(f"    top_words: {b['top_words']}")
        for c in b['children']:
            print(f"    ├── CHILD {c['id']}: [{c['type']}] \"{c['text'][:50]}...\"")
            print(f"    |   top_words: {c['top_words']}")
        for s in store.get_sentences_by_block(b['id']):
            print(f"    ├── SENT {s['id']}: [{s['type']}] \"{s['text'][:50]}...\"")

    store.close()
    print("\nDone!")
