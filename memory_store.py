import json
import pickle
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from text_utils import extract_top_words


class MemoryStore:
    def __init__(self, db_path=None, lda_path=None):
        self.db_path = db_path or str(Path(__file__).parent / "memory.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.analyzer = SentimentIntensityAnalyzer()
        self.lda = None
        if lda_path:
            self.load_lda(lda_path)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'paragraph',
                parent_id INTEGER REFERENCES blocks(id),
                top_words TEXT DEFAULT '[]',
                sentiment REAL DEFAULT 0.0,
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

    # ===== INSERT =====

    def add_block(self, text, block_type='paragraph', parent_id=None):
        top_words = json.dumps(extract_top_words(text, content_type=block_type))
        sentiment = self.analyzer.polarity_scores(text)['compound']
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO blocks (text, type, parent_id, top_words, sentiment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (text.strip(), block_type, parent_id, top_words, sentiment, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_sentence(self, text, block_id, line_number=0, sent_type='sentence',
                     embedding=None, score=0.0, label='unlabeled', strip=True):
        text_content = text.strip() if strip else text
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sentences (text, block_id, line_number, type, embedding, score, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (text_content, block_id, line_number, sent_type, emb_blob, float(score), label, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_relation(self, parent_id, child_id, relation_type='child_of'):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO block_relations (parent_id, child_id, relation_type) VALUES (?, ?, ?)",
            (parent_id, child_id, relation_type)
        )
        self.conn.commit()
        return cur.lastrowid

    # ===== QUERY =====

    def get_block(self, block_id):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
        return cur.fetchone()

    def get_block_children(self, block_id):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT b.* FROM blocks b
            JOIN block_relations r ON b.id = r.child_id
            WHERE r.parent_id = ? AND r.relation_type = 'child_of'
            ORDER BY b.id
        """, (block_id,))
        return cur.fetchall()

    def get_sentences_by_block(self, block_id):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM sentences WHERE block_id = ? ORDER BY line_number",
            (block_id,)
        )
        return cur.fetchall()

    def get_labeled(self, label='flagged', limit=50):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT s.*, b.type as block_type, b.top_words
            FROM sentences s
            JOIN blocks b ON s.block_id = b.id
            WHERE s.label = ?
            ORDER BY s.created_at DESC
            LIMIT ?
        """, (label, limit))
        return cur.fetchall()

    def search_by_keyword(self, keyword, label='flagged', limit=20):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT s.*, b.type as block_type
            FROM sentences s
            JOIN blocks b ON s.block_id = b.id
            WHERE s.label = ? AND s.text LIKE ?
            ORDER BY s.score DESC
            LIMIT ?
        """, (label, f'%{keyword}%', limit))
        return cur.fetchall()

    def get_tree(self, block_id=None):
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
        with open(lda_path, 'rb') as f:
            self.lda = pickle.load(f)

    def count_stats(self):
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
        self.conn.close()


class BlockParser:
    def __init__(self, memory_store, embed_fn=None):
        self.store = memory_store
        self.embed = embed_fn

    def parse_and_store(self, text):
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
                block_id = self.store.add_block(code_text, 'code', current_block_id)
                all_block_ids.append(block_id)

                for ln, code_line in enumerate(code_lines):
                    sid = self.store.add_sentence(
                        code_line, block_id, ln, 'code_line',
                        score=0.0, label='unlabeled', strip=False
                    )
                    all_sentence_ids.append(sid)

                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                last_block_type = 'code'
                continue

            # --- List item ---
            if re.match(r'^\s*(?:\d+[\.\)]|[-*])\s', stripped):
                if current_block_id is None:
                    block_id = self.store.add_block(stripped, 'list', None)
                    current_block_id = block_id
                    all_block_ids.append(block_id)

                line_num = len(self.store.get_sentences_by_block(current_block_id))
                sid = self.store.add_sentence(
                    stripped, current_block_id, line_num, 'list_item',
                    score=0.0, label='unlabeled'
                )
                all_sentence_ids.append(sid)

                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + stripped if existing else stripped
                cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
                    (new_text, json.dumps(extract_top_words(new_text)), current_block_id))
                self.store.conn.commit()
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
                block_id = self.store.add_block(table_text, 'table', current_block_id)
                all_block_ids.append(block_id)

                for ln, tl in enumerate(table_lines):
                    sid = self.store.add_sentence(tl, block_id, ln, 'table_row')
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
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence')
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
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence')
                    all_sentence_ids.append(sid)
                last_block_type = 'paragraph'
            else:
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[])', stripped) if len(s.strip()) > 3]
                block_text = ' '.join(sents)
                block_id = self.store.add_block(block_text, 'paragraph', None)
                all_block_ids.append(block_id)
                for ln, s in enumerate(sents):
                    sid = self.store.add_sentence(s, block_id, ln, 'sentence')
                    all_sentence_ids.append(sid)
                current_block_id = block_id

            i += 1

        if self.embed and self.store.lda:
            self._classify_sentences(all_sentence_ids)

        return all_block_ids, all_sentence_ids

    def _classify_sentences(self, sentence_ids):
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
