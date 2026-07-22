"""
Hierarchical memory store — SQLite-backed block/sentence storage with LDA classification.

Parses markdown text into hierarchical blocks (paragraph, code, table, list, equation)
via BlockParser, classifies sentences with LDA, and propagates keywords for retrieval.
Used by StreamIntervention for mid-reasoning memory injection.
"""
import json
import pickle
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from text_utils import (
    extract_top_words,
    merge_top_words,
    detect_bare_reference,
    inject_type_keywords,
    filter_injected_keywords,
)


class MemoryStore:
    """SQLite-backed memory with blocks, sentences, and hierarchical relations.

    Stores parsed AI responses as block trees (parent paragraph -> child code/table).
    Provides keyword indexing, owner_id filtering, and LDA classification.
    """

    def __init__(self, db_path=None, lda_path=None):
        self.db_path = db_path or str(Path(__file__).parent / 'memory.db')
        self.db_path = db_path or str(Path(__file__).parent / "memory.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.analyzer = SentimentIntensityAnalyzer()
        self.lda = None
        if lda_path:
            self.load_lda(lda_path)
        self._create_tables()

    def _create_tables(self):
        '''Create SQLite tables for blocks, sentences, block_relations. Always safe to call (IF NOT EXISTS).'''
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'paragraph',
                parent_id INTEGER REFERENCES blocks(id),
                top_words TEXT DEFAULT '[]',
                sentiment REAL DEFAULT 0.0,
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
        self._migrate()

    def _migrate(self):
        """
        Add missing columns to existing databases.
        
        Tries to ALTER TABLE to add owner_id. If the column already
        exists, sqlite3.OperationalError is caught and ignored.
        This ensures existing databases work after schema changes.
        """
        for table in ('blocks', 'sentences'):
            try:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN owner_id INTEGER"
                )
                self.conn.commit()
            except sqlite3.OperationalError:
                pass

    # ===== INSERT =====

    def add_block(self, text, block_type='paragraph', parent_id=None, owner_id=None):
        """
        Store a parsed block in the database.
        
        Args:
            text: Block text content.
            block_type: 'paragraph', 'code', 'table', 'list', or 'equation'.
            parent_id: FK to parent block (for tree structure).
            owner_id: Discord user ID for privacy filtering (None = public).
        
        Returns:
            The new block's auto-increment ID.
        
        Structural type seeding: code, table, list, and equation blocks
        automatically get their type name appended to top_words (e.g., a
        code block gets 'code'). This enables the structural match tier
        in StreamIntervention.check_match() — the AI can reference a
        "code block" or "table" and the system finds the right object.
        """
        top_words = extract_top_words(text, content_type=block_type)
        # Seed structural type name for retrieval
        structural_types = {'code', 'table', 'list', 'equation'}
        if block_type in structural_types and block_type not in top_words:
            top_words.append(block_type)
            top_words = top_words[:12]
        top_words = json.dumps(top_words)
        sentiment = self.analyzer.polarity_scores(text)['compound']
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO blocks (text, type, parent_id, top_words, sentiment, owner_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (text.strip(), block_type, parent_id, top_words, sentiment, owner_id, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_sentence(self, text, block_id, line_number=0, sent_type='sentence',
                     embedding=None, score=0.0, label='unlabeled', strip=True,
                     owner_id=None):
        """
        Store an individual sentence linked to a block.
        
        Args:
            text: Sentence text.
            block_id: FK to the parent block.
            line_number: Position within the block (for ordering).
            sent_type: 'sentence', 'code_line', 'table_row', 'list_item', 'equation'.
            embedding: Optional numpy float32 array stored as BLOB.
            score: LDA classifier decision function value (positive = flagged).
            label: 'flagged' or 'unlabeled' — set by LDA classifier.
            strip: Whether to strip whitespace (False for code lines).
            owner_id: Discord user ID for privacy filtering.
        """
        text_content = text.strip() if strip else text
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sentences (text, block_id, line_number, type, embedding, score, label, owner_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (text_content, block_id, line_number, sent_type, emb_blob, float(score), label, owner_id, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_relation(self, parent_id, child_id, relation_type='child_of'):
        '''Create a block_relations link between parent and child blocks.'''
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO block_relations (parent_id, child_id, relation_type) VALUES (?, ?, ?)",
            (parent_id, child_id, relation_type)
        )
        self.conn.commit()
        return cur.lastrowid

    # ===== QUERY =====

    def get_block(self, block_id):
        '''Fetch a single block by ID.'''
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
        return cur.fetchone()

    def get_block_children(self, block_id):
        '''Fetch all child blocks for a parent.'''
        cur = self.conn.cursor()
        cur.execute("""
            SELECT b.* FROM blocks b
            JOIN block_relations r ON b.id = r.child_id
            WHERE r.parent_id = ? AND r.relation_type = 'child_of'
            ORDER BY b.id
        """, (block_id,))
        return cur.fetchall()

    def get_sentences_by_block(self, block_id):
        '''Fetch all sentences in a block, ordered by line_number.'''
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM sentences WHERE block_id = ? ORDER BY line_number",
            (block_id,)
        )
        return cur.fetchall()

    def get_tree(self, block_id=None):
        '''Fetch root blocks (or a single block) with nested children.'''
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

    # Keywords for structural element types (code, table, list, equation).
    # Used by build_match_index() to filter structural words.
    STRUCTURAL_NAMES = {'table', 'tables', 'code', 'list', 'lists', 'equation', 'equations'}

    def _block_matches_structural(self, word, block_id):
        """Check if structural keyword matches this block or any children."""
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
        """Match lowercase structural word to canonical block type."""
        return (
            (wl in ('table', 'tables') and block_type == 'table') or
            (wl == 'code' and block_type == 'code') or
            (wl in ('list', 'lists') and block_type == 'list') or
            (wl in ('equation', 'equations') and block_type == 'equation')
        )

    def build_match_index(self, owner_id=None):
        """Build {keyword: [sentence_info]} inverted index for intervention.
        Filters by owner_id and strips generic injected keywords."""
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
                    if not self._block_matches_structural(wl, block_id):
                        continue
                elif wl not in clean_words:
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
        """Return block with nested children for deep-tier injection."""
        block = self.get_block(block_id)
        if not block:
            return None
        node = dict(block)
        node['children'] = [dict(c) for c in self.get_block_children(block_id)]
        return node

    def load_lda(self, lda_path):
        '''Load a pre-trained LDA classifier from a pickle file.'''
        with open(lda_path, 'rb') as f:
            self.lda = pickle.load(f)

    def count_stats(self):
        '''Return dict of {sentences, blocks, flagged, relations} counts.'''
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
        '''Close the database connection.'''
        self.conn.close()


class BlockParser:
    """Line-by-line markdown parser that stores hierarchical blocks into MemoryStore.

    Detects code blocks (```), equations ($$), tables (|...|), list items (#. / -),
    and paragraphs. Propagates parent keywords to children and runs LDA classification.
    """

    def __init__(self, memory_store, embed_fn=None):
        self.store = memory_store
        self.embed = embed_fn

    def parse_and_store(self, text, owner_id=None):
        """Parse markdown text into blocks/sentences, store via add_block/add_sentence."""
        all_block_ids = []
        all_sentence_ids = []
        lines = text.split('\n')
        i = 0
        current_block_id = None
        last_block_type = None

        while i < len(lines):
            stripped = lines[i].strip()

            if not stripped:
                if last_block_type in ('table', 'code', 'list', 'equation'):
                    current_block_id = None
                # If last_block_type is 'paragraph' or None, keep current_block_id
                # so consecutive paragraphs merge into one block
                i += 1
                continue

            # --- Code block ---
            if stripped.startswith('```'):
                code_lines = [lines[i]]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])
                    i += 1

                code_text = '\n'.join(code_lines)
                block_id = self.store.add_block(code_text, 'code', current_block_id, owner_id)
                all_block_ids.append(block_id)

                for ln, code_line in enumerate(code_lines):
                    sid = self.store.add_sentence(
                        code_line, block_id, ln, 'code_line',
                        score=0.0, label='unlabeled', strip=False
                    , owner_id=owner_id)
                    all_sentence_ids.append(sid)

                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                last_block_type = 'code'
                continue

            # --- Equation block ---
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
                    sid = self.store.add_sentence(eql, block_id, ln, 'equation',
                                                   score=0.0, label='unlabeled', owner_id=owner_id)
                    all_sentence_ids.append(sid)

                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                last_block_type = 'equation'
                continue

            # --- List item ---
            if re.match(r'^\s*(?:\d+[\.\)]|[-*])\s', stripped):
                if current_block_id is None:
                    block_id = self.store.add_block(stripped, 'list', None, owner_id)
                    current_block_id = block_id
                    all_block_ids.append(block_id)

                line_num = len(self.store.get_sentences_by_block(current_block_id))
                sid = self.store.add_sentence(
                    stripped, current_block_id, line_num, 'list_item',
                    score=0.0, label='unlabeled'
                , owner_id=owner_id)
                all_sentence_ids.append(sid)

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

            # --- Table ---
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

            # --- Regular sentence ---
            if current_block_id is not None and not stripped.endswith(':'):
                for ln, s in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped)):
                    s = s.strip()
                    if not s:
                        continue
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence', owner_id=owner_id)
                    all_sentence_ids.append(sid)

                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + stripped
                cur.execute("UPDATE blocks SET text = ?, top_words = ?, sentiment = ? WHERE id = ?",
                    (new_text, json.dumps(extract_top_words(new_text)),
                     self.store.analyzer.polarity_scores(new_text)['compound'], current_block_id))
                self.store.conn.commit()
                last_block_type = 'paragraph'
            elif current_block_id is not None and stripped.endswith(':'):
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped) if len(s.strip()) > 3]
                block_text = ' '.join(sents)
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + block_text
                cur.execute("UPDATE blocks SET text = ?, top_words = ?, sentiment = ? WHERE id = ?",
                    (new_text, json.dumps(extract_top_words(new_text)),
                     self.store.analyzer.polarity_scores(new_text)['compound'], current_block_id))
                self.store.conn.commit()
                for ln, s in enumerate(sents):
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence', owner_id=owner_id)
                    all_sentence_ids.append(sid)
                last_block_type = 'paragraph'
            else:
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped) if len(s.strip()) > 3]
                block_text = ' '.join(sents)
                block_id = self.store.add_block(block_text, 'paragraph', None, owner_id)
                all_block_ids.append(block_id)
                for ln, s in enumerate(sents):
                    sid = self.store.add_sentence(s, block_id, ln, 'sentence', owner_id=owner_id)
                    all_sentence_ids.append(sid)
                current_block_id = block_id
                last_block_type = 'paragraph'

            i += 1

        self._propagate_top_words()

        if self.embed and self.store.lda:
            self._classify_sentences(all_sentence_ids)

        return all_block_ids, all_sentence_ids

    def _propagate_top_words(self):
        '''Merge parent keywords into children. Seed structural type names for code/table/equation blocks.'''
        """Propagate parent keywords to children: bare ref injection, merge, type seeding."""
        cur = self.store.conn.cursor()
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
            cur.execute("SELECT top_words FROM blocks WHERE id = ?", (pid,))
            row = cur.fetchone()
            if not row:
                continue
            parent_words = json.loads(row[0])
            child_types = list({ct for _, ct in children})

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
                merged = merge_top_words(parent_words, child_words)

                # Always seed the structural type name as a keyword
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
        '''Score each sentence with LDA: embeds -> decision_function -> label='flagged' or 'unlabeled'.'''
        '''Run LDA classifier on each sentence. Code lines, equations, and table rows inherit parent labels.'''
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

            if len(text.strip()) < 10:
                continue
            emb = self.embed(text).reshape(1, -1)
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
