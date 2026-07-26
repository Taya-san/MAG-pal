from __future__ import annotations
import json

import re


from collections import defaultdict

from text_utils import (
    extract_top_words,
    merge_top_words,
    is_keyword_sparse,
    inject_type_keywords,
    filter_injected_keywords,
)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(\[])')



class BlockParser:
    """
    Line-by-line markdown parser that stores hierarchical blocks into MemoryStore.
    
    Parse order (first match wins):
      1. Code blocks (lines starting with ```)
      2. Equations (lines starting with $$)
      3. List items (numbered/bulleted lines)
      4. Tables (lines starting and ending with |)
      5. Paragraphs (everything else)
    
    After parsing, _enrich_child_block_keywords() runs to:
      - Merge parent keywords into child blocks
      - Detect "bare" parents and inject structural type keywords
      - Seed type names on code/table/list/equation child blocks
    
    If embed_fn and LDA model are available, _classify_sentences() then
    scores each sentence and labels it 'flagged' or 'unlabeled'.
    """
    
    def __init__(self, memory_store, embed_fn=None):
        self.store = memory_store
        self.embed = embed_fn

    def _parse_delimited_block(self, lines, i, start_delim, end_delim,
                                   block_type, sent_type, current_block_id,
                                   owner_id, strip_sentences=True,
                                   continue_fn=None):
        """Parse a delimited block (```, $$, table) and return (new_i, block_id, is_new).

        Args:
            continue_fn: Optional function(line, lines, i) returning bool.
                         For tables: continues while a line starts AND ends with |.
                         Default: continues until a line starts with end_delim.
        """
        collected = [lines[i]]
        i += 1
        while i < len(lines):
            if continue_fn:
                should_continue = continue_fn(lines[i], lines, i)
            else:
                should_continue = not lines[i].strip().startswith(end_delim)
            if not should_continue:
                break
            collected.append(lines[i])
            i += 1
        if not continue_fn and i < len(lines):
            # Include the closing delimiter line (``` or $$, not for tables)
            collected.append(lines[i])
            i += 1

        full_text = '\n'.join(collected)
        block_id, is_new = self.store.add_block(full_text, block_type, current_block_id, owner_id)

        if is_new:
            for ln, line in enumerate(collected):
                self.store.add_sentence(
                    line, block_id, ln, sent_type,
                    score=0.0, label='unlabeled', strip=strip_sentences,
                    owner_id=owner_id
                )
        return i, block_id, is_new

    def _parse_list_item(self, lines, i, stripped, current_block_id, owner_id,
                          all_block_ids, all_sentence_ids):
        """Parse a list item line and return the next index i."""
        if current_block_id is None:
            block_id, _ = self.store.add_block(stripped, 'list', None, owner_id)
            current_block_id = block_id
            all_block_ids.append(block_id)

        line_num = len(self.store.get_sentences_by_block(current_block_id))
        sid = self.store.add_sentence(
            stripped, current_block_id, line_num, 'list_item',
            score=0.0, label='unlabeled', owner_id=owner_id
        )
        all_sentence_ids.append(sid)

        cur = self.store.conn.cursor()
        cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
        existing = cur.fetchone()[0]
        new_text = existing + ' ' + stripped if existing else stripped
        cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
            (new_text, json.dumps(extract_top_words(new_text)), current_block_id))
        self.store.conn.commit()
        return i + 1

    def _parse_paragraph_line(self, lines, i, stripped, current_block_id, last_block_type,
                              last_structural_parent_id, owner_id, all_block_ids, all_sentence_ids):
        """Parse a regular sentence/paragraph line and return (i, last_block_type, current_block_id, last_structural_parent_id)."""
        if current_block_id is not None and not stripped.endswith(':'):
            for ln, s in enumerate(SENTENCE_SPLIT_RE.split(stripped)):
                s = s.strip()
                if not s:
                    continue
                self.store.add_sentence(s, current_block_id, ln, 'sentence', owner_id=owner_id)
                all_sentence_ids.append(s)

            cur = self.store.conn.cursor()
            cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
            existing = cur.fetchone()[0]
            new_text = existing + ' ' + stripped
            cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
                (new_text, json.dumps(extract_top_words(new_text)), current_block_id))
            self.store.conn.commit()
            last_block_type = 'paragraph'

        elif current_block_id is not None and stripped.endswith(':'):
            sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(stripped) if len(s.strip()) > 3]
            block_text = ' '.join(sents)
            cur = self.store.conn.cursor()
            cur.execute("SELECT text FROM blocks WHERE id = ?", (current_block_id,))
            existing = cur.fetchone()[0]
            new_text = existing + ' ' + block_text
            cur.execute("UPDATE blocks SET text = ?, top_words = ? WHERE id = ?",
                (new_text, json.dumps(extract_top_words(new_text)), current_block_id))
            self.store.conn.commit()
            for ln, s in enumerate(sents):
                self.store.add_sentence(s, current_block_id, ln, 'sentence', owner_id=owner_id)
                all_sentence_ids.append(s)
            last_block_type = 'paragraph'
        else:
            sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(stripped) if len(s.strip()) > 3]
            block_text = ' '.join(sents)

            is_multi = False
            if last_structural_parent_id is not None:
                peep = i + 1
                while peep < len(lines) and not lines[peep].strip():
                    peep += 1
                is_multi = (
                    peep < len(lines)
                    and lines[peep].strip()
                    and not lines[peep].strip().startswith('$$')
                    and not lines[peep].strip().startswith('```')
                    and not re.match(r'^\s*(?:\d+[\.\)]|[-*])\s', lines[peep].strip())
                    and not (lines[peep].strip().startswith('|') and lines[peep].strip().endswith('|'))
                )

            if last_structural_parent_id is not None and not is_multi:
                block_id, is_new = self.store.add_block(block_text, 'paragraph', last_structural_parent_id, owner_id)
                self.store.add_relation(last_structural_parent_id, block_id)
                last_structural_parent_id = None
            else:
                block_id, is_new = self.store.add_block(block_text, 'paragraph', None, owner_id)
                last_structural_parent_id = None

            all_block_ids.append(block_id)
            if is_new:
                for ln, s in enumerate(sents):
                    self.store.add_sentence(s, block_id, ln, 'sentence', owner_id=owner_id)
                    all_sentence_ids.append(s)
            current_block_id = block_id
            last_block_type = 'paragraph'

        return i + 1, last_block_type, current_block_id, last_structural_parent_id

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
        last_structural_parent_id = None  # paragraph that was parent before a structural block
        
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
                i, block_id, is_new = self._parse_delimited_block(
                    lines, i, '```', '```', 'code', 'code_line',
                    current_block_id, owner_id, strip_sentences=False
                )
                all_block_ids.append(block_id)
                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                    last_structural_parent_id = current_block_id
                current_block_id = None
                last_block_type = 'code'
                continue

            # ---- Equation block detection ----
            if stripped.startswith('$$'):
                i, block_id, is_new = self._parse_delimited_block(
                    lines, i, '$$', '$$', 'equation', 'equation',
                    current_block_id, owner_id
                )
                all_block_ids.append(block_id)
                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                    last_structural_parent_id = current_block_id
                current_block_id = None
                last_block_type = 'equation'
                continue

            # ---- List item detection ----
            if re.match(r'^\s*(?:\d+[\.\)]|[-*])\s', stripped):
                i = self._parse_list_item(lines, i, stripped, current_block_id, owner_id, all_block_ids, all_sentence_ids)
                last_block_type = 'list'
                continue

            # ---- Table detection ----
            if stripped.startswith('|') and stripped.endswith('|'):
                i, block_id, is_new = self._parse_delimited_block(
                    lines, i, '|', '|', 'table', 'table_row',
                    current_block_id, owner_id,
                    continue_fn=lambda line, _lines, _i: (
                        line.strip().startswith('|') and line.strip().endswith('|')
                    )
                )
                all_block_ids.append(block_id)
                if current_block_id:
                    self.store.add_relation(current_block_id, block_id)
                    last_structural_parent_id = current_block_id
                current_block_id = None
                last_block_type = 'table'
                continue

            # ---- Regular sentence (paragraph) ----
            i, last_block_type, current_block_id, last_structural_parent_id = self._parse_paragraph_line(
                lines, i, stripped, current_block_id, last_block_type, last_structural_parent_id,
                owner_id, all_block_ids, all_sentence_ids
            )

        # After all blocks are parsed, propagate keywords
        self._enrich_child_block_keywords()

        # If embedding function and LDA model are available, classify sentences
        if self.embed and self.store.lda:
            self._classify_sentences(all_sentence_ids)

        return all_block_ids, all_sentence_ids

    def _enrich_child_block_keywords(self):
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
            if is_keyword_sparse(parent_words):
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
    from memory_store import MemoryStore
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
