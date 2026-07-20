"""  
Memory Store — SQLite-backed hierarchical memory for the MAG-pal bot.

Schema:
  sentences: one row per sentence or line of text
    id | text | block_id | type | embedding | score | label | created_at

  blocks: groups of sentences forming a logical unit
    id | text | type | parent_id | top_words | sentiment | created_at

  block_relations: parent-child links between blocks
    id | parent_id | child_id | relation_type

Each sentence gets its own embedding and LDA score.
Blocks group related sentences and form parent-child hierarchies.
"""
import sqlite3, json, re, time, numpy as np
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DB_PATH = "/home/taya/Projects/MAG-pal/memory.db"

# ===== STOPWORDS =====
STOPS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'should', 'now',
    'let', 'explain', 'make', 'made', 'making', 'get', 'got',
    'use', 'using', 'used', 'way', 'thing', 'things', 'like',
    'also', 'well', 'even', 'much', 'many', 'still', 'already',
    'yet', 'may', 'might', 'could', 'would', 'shall', 'will',
    'can', 'need', 'take', 'took', 'taken', 'say', 'said',
    'go', 'went', 'gone', 'come', 'came', 'know', 'see',
])


class MemoryStore:
    """SQLite-backed store for sentences, blocks, and their relationships."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.analyzer = SentimentIntensityAnalyzer()

    def _create_tables(self):
        """Create the schema if it doesn't exist."""
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

    def extract_top_words(self, text, max_words=5):
        """Extract top words from text, excluding stopwords."""
        cleaned = re.sub(r'[^a-z\s]', ' ', text.lower())
        words = [w for w in cleaned.split() if w not in STOPS and len(w) > 2]
        if not words:
            return []
        return [w for w, _ in Counter(words).most_common(max_words)]

    # ===== INSERT =====

    def add_block(self, text, block_type='paragraph', parent_id=None):
        """Insert a block and return its ID.
        
        A block is a group of related lines: a paragraph, code block,
        list group, table, or equation.
        """
        top_words = json.dumps(self.extract_top_words(text))
        sentiment = self.analyzer.polarity_scores(text)['compound']
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO blocks (text, type, parent_id, top_words, sentiment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (text.strip(), block_type, parent_id, top_words, sentiment, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_sentence(self, text, block_id, line_number=0, sent_type='sentence',
                     embedding=None, score=0.0, label='unlabeled'):
        """Insert a single sentence or line into the sentences table.
        
        embedding: 384-dim numpy array, stored as binary blob.
        label: category assigned later (e.g. 'definition', 'rule', 'preference')
        """
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sentences (text, block_id, line_number, type, embedding, score, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (text.strip(), block_id, line_number, sent_type, emb_blob, float(score), label, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_relation(self, parent_id, child_id, relation_type='child_of'):
        """Link two blocks as parent-child."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO block_relations (parent_id, child_id, relation_type) VALUES (?, ?, ?)",
            (parent_id, child_id, relation_type)
        )
        self.conn.commit()
        return cur.lastrowid

    # ===== QUERY =====

    def get_block(self, block_id):
        """Get a block by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
        return cur.fetchone()

    def get_block_children(self, block_id):
        """Get all child blocks of a block."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT b.* FROM blocks b
            JOIN block_relations r ON b.id = r.child_id
            WHERE r.parent_id = ? AND r.relation_type = 'child_of'
            ORDER BY b.id
        """, (block_id,))
        return cur.fetchall()

    def get_sentences_by_block(self, block_id):
        """Get all sentences in a block, ordered by line_number."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM sentences WHERE block_id = ? ORDER BY line_number",
            (block_id,)
        )
        return cur.fetchall()

    def get_labeled(self, label='flagged', limit=50):
        """Get the most recent sentences with a given label."""
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
        """Find labeled sentences containing a keyword."""
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
        """Get the full block tree (root if None)."""
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

    def count_stats(self):
        """Return counts of stored data."""
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


# ===== BLOCK PARSER (integrated with MemoryStore) =====

class BlockParser:
    """Parses AI response text and stores directly into MemoryStore.
    
    Each paragraph/block becomes a 'block' row.
    Each sentence/line becomes a 'sentence' row linked to its block.
    Parent-child relationships use block_relations.
    
    Detection order: code → list → table → equation → paragraph
    """

    def __init__(self, memory_store, embed_fn=None, lda=None):
        self.store = memory_store
        self.embed = embed_fn
        self.lda = lda

    def parse_and_store(self, text):
        """Parse text and store everything into the database.
        
        Returns: (block_ids, sentence_ids) for the created records.
        """
        lines = text.split('\n')
        all_block_ids = []
        all_sentence_ids = []
        i = 0
        current_block_id = None  # block collecting children
        
        while i < len(lines):
            stripped = lines[i].strip()
            
            # --- Empty line: finalize current block ---
            if not stripped:
                current_block_id = None
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
                
                # Each line of code becomes a sentence
                for ln, code_line in enumerate(code_lines):
                    sid = self.store.add_sentence(
                        code_line, block_id, ln, 'code_line',
                        score=0.0, label='unlabeled'
                    )
                    all_sentence_ids.append(sid)
                
                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                current_block_id = None
                continue
            
            # --- List item ---
            if re.match(r'^\s*(?:\d+[\.\)]|[-*])\s', stripped):
                if current_block_id is None:
                    # Create a parent block for orphan list items
                    block_id = self.store.add_block('', 'list', None)
                    current_block_id = block_id
                    all_block_ids.append(block_id)
                    if current_block_id:
                        pass  # already set
                
                sid = self.store.add_sentence(
                    stripped, current_block_id, 0, 'list_item',
                    score=0.0, label='unlabeled'
                )
                all_sentence_ids.append(sid)
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
                continue
            
            # --- Equation (has =, not ending with .) ---
            if '=' in stripped and not stripped.endswith('.'):
                block_id = self.store.add_block(stripped, 'equation', current_block_id)
                all_block_ids.append(block_id)
                sid = self.store.add_sentence(stripped, block_id, 0, 'equation')
                all_sentence_ids.append(sid)
                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                i += 1
                continue
            
            # --- Paragraph (may end with ':' to introduce children) ---
            if stripped.endswith(':') and not stripped.startswith('#'):
                clean = stripped.rstrip(':').strip()
                block_id = self.store.add_block(clean, 'paragraph', current_block_id)
                all_block_ids.append(block_id)
                sid = self.store.add_sentence(clean, block_id, 0, 'sentence')
                all_sentence_ids.append(sid)
                current_block_id = block_id
                i += 1
                continue
            
            # --- Regular sentence (continuation or new paragraph) ---
            if current_block_id is not None and not stripped.endswith(':'):
                # Continuation: add each sentence individually to current block
                for ln, s in enumerate(re.split(r'(?<=[.!?])\s+', stripped)):
                    s = s.strip()
                    if not s:
                        continue
                    sid = self.store.add_sentence(s, current_block_id, ln, 'sentence')
                    all_sentence_ids.append(sid)
                # Update block text (full merged paragraph)
                cur = self.store.conn.cursor()
                cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
                existing = cur.fetchone()[0]
                new_text = existing + ' ' + stripped
                cur.execute("UPDATE blocks SET text = ?, top_words = ?, sentiment = ? WHERE id = ?",
                    (new_text, json.dumps(self.store.extract_top_words(new_text)),
                     self.store.analyzer.polarity_scores(new_text)['compound'], current_block_id))
                self.store.conn.commit()
            else:
                # New paragraph: split into individual sentences
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', stripped) if len(s.strip()) > 3]
                block_text = ' '.join(sents)
                block_id = self.store.add_block(block_text, 'paragraph', None)
                all_block_ids.append(block_id)
                for ln, s in enumerate(sents):
                    sid = self.store.add_sentence(s, block_id, ln, 'sentence')
                    all_sentence_ids.append(sid)
                current_block_id = block_id
            
            i += 1
        
        # Now classify each sentence with LDA if available
        if self.embed and self.lda:
            self._classify_sentences(all_sentence_ids)
        
        return all_block_ids, all_sentence_ids

    def _classify_sentences(self, sentence_ids):
        """Run LDA on all stored sentences and set their label.
        
        Currently labels as 'flagged' if LDA score > 0.
        You can change label thresholds later without reprocessing.
        """
        cur = self.store.conn.cursor()
        for sid in sentence_ids:
            cur.execute("SELECT text FROM sentences WHERE id = ?", (sid,))
            row = cur.fetchone()
            if not row or len(row[0].strip()) < 10:
                continue
            emb = self.embed(row[0]).reshape(1, -1)
            score = float(self.lda.decision_function(emb)[0])
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
