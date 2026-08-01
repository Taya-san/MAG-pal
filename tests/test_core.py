from __future__ import annotations
"""Tests for streaming intervention, memory store, and database operations."""

import sys
sys.path.insert(0, '/home/taya/Projects/MAG-pal')

import json
import asyncio
from memory_store import MemoryStore, BlockParser
from stream_intervention import StreamIntervention
from db import Database


# ============= stream_intervention.py tests =============

def test_buffer_token_extracts_words():
    """Sub-word streaming tokens are assembled into complete words."""
    si = StreamIntervention()
    si.reset_buffer()
    
    # Simulate streaming tokens: "sym" + "metric" → "symmetric"
    results = []
    for tok in [" sym", "metric", " matric", "es."]:
        results.extend(si.buffer_token(tok))
    
    assert "symmetric" in results, f"Expected 'symmetric', got {results}"
    assert "matrices" in results, f"Expected 'matrices', got {results}"
    assert si.word_buffer == ""

def test_buffer_token_requires_word_boundary():
    """Words with no trailing boundary character stay in buffer."""
    si = StreamIntervention()
    si.reset_buffer()
    
    results = si.buffer_token("symmetric")
    assert results == [], "No trailing boundary → no word emitted"
    assert si.word_buffer == "symmetric"

def test_buffer_token_filters_short_words():
    """Words shorter than 3 characters are not emitted."""
    si = StreamIntervention()
    si.reset_buffer()
    
    results = []
    for t in ["I ", "am "]:
        results.extend(si.buffer_token(t))
    
    assert results == [], f"Short words should be filtered, got {results}"

def test_flush_emits_trailing_word():
    """Flush extracts word at end of buffer without trailing boundary."""
    si = StreamIntervention()
    si.reset_buffer()
    for t in [" cons", "ider", " the"]:
        si.buffer_token(t)
    
    flushed = si.flush_buffer()
    assert "the" in flushed, f"Expected 'the' in flush, got {flushed}"
    assert si.word_buffer == ""

def test_build_index_with_flagged_sentences():
    """Index contains content words and structural type names."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("symmetric matrix:\n```python\ndef f(): pass\n```")
    c = s.conn.cursor()
    c.execute("UPDATE sentences SET label='flagged',score=0.8 WHERE text LIKE '%symmetric%'")
    s.conn.commit()
    
    si = StreamIntervention(memory_store=s)
    assert "symmetric" in si.keyword_index
    assert "code" in si.keyword_index
    s.close()

def test_empty_index_when_no_flagged():
    """Index is empty when no sentences are flagged."""
    s = MemoryStore(":memory:")
    BlockParser(s).parse_and_store("just some text.")
    si = StreamIntervention(memory_store=s)
    assert si.keyword_index == {}
    s.close()

def test_cooldown_prevents_re_match():
    """A sentence on cooldown cannot trigger again."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("symmetric matrix:\n```python\nx=1\n```")
    c = s.conn.cursor()
    c.execute("UPDATE sentences SET label='flagged',score=0.8")
    s.conn.commit()
    
    si = StreamIntervention(memory_store=s)
    m1 = asyncio.run(si.check_match("symmetric", ["symmetric"]))
    assert m1 is not None, "First match should fire"
    assert si.cooldowns.get(m1["sentence_id"], 0) > 0, "Cooldown should be set"
    
    m2 = asyncio.run(si.check_match("symmetric", ["symmetric"]))
    # Second match blocked OR different sentence (both valid)
    if m2 is not None:
        assert m2["sentence_id"] != m1["sentence_id"], "Should be different sentence"
    
    s.close()

def test_deep_tier_triggers_on_high_overlap():
    """Deep tier (>=70%) fires when most keywords match."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("The chain rule differentiates composite functions:\n$$f'(x) = ...$$\n\nRules:\n1. Power rule\n2. Product rule")
    c = s.conn.cursor()
    c.execute("UPDATE sentences SET label='flagged',score=0.8 WHERE text LIKE '%chain%'")
    s.conn.commit()
    
    si = StreamIntervention(memory_store=s)
    # Match enough keywords to exceed 70%
    m = asyncio.run(si.check_match("chain", ["chain", "rule", "differentiates", "composite", "functions"]))
    assert m is not None
    assert m["tier"] == "deep"
    assert m["ratio"] >= 0.7
    s.close()

def test_cooldown_decay():
    """Cooldowns expire after sufficient decrement."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("symmetric matrix:\n```python\nx=1\n```")
    c = s.conn.cursor()
    c.execute("UPDATE sentences SET label='flagged',score=0.8")
    s.conn.commit()
    
    si = StreamIntervention(memory_store=s)
    m1 = asyncio.run(si.check_match("symmetric", ["symmetric"]))
    sid = m1["sentence_id"]
    assert si.cooldowns.get(sid, 0) == 150  # default limit
    
    si.decrement_cooldowns(200)
    assert sid not in si.cooldowns
    
    m2 = asyncio.run(si.check_match("symmetric", ["symmetric"]))
    assert m2 is not None, "Should match after cooldown expires"
    s.close()

def test_output_decay():
    """apply_output_decay reduces cooldowns by 50% of output length."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("symmetric matrix:\n```python\nx=1\n```")
    c = s.conn.cursor()
    c.execute("UPDATE sentences SET label='flagged',score=0.8")
    s.conn.commit()
    
    si = StreamIntervention(memory_store=s)
    m = asyncio.run(si.check_match("symmetric", ["symmetric"]))
    remaining_before = si.cooldowns.get(m["sentence_id"], 0)
    
    si.apply_output_decay(200)  # 200 output words → decay = 100
    remaining_after = si.cooldowns.get(m["sentence_id"], 0)
    assert remaining_after <= remaining_before - 100
    s.close()

def test_rebuild_index_in_place():
    """Rebuild updates shared dict reference without replacing it."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("eigenvalues are real")
    c = s.conn.cursor()
    c.execute("UPDATE sentences SET label='flagged',score=0.8 WHERE text LIKE '%eigenvalues%'")
    s.conn.commit()
    
    si = StreamIntervention(memory_store=s)
    original_id = id(si.keyword_index)
    si.rebuild_index()
    assert id(si.keyword_index) == original_id, "Rebuild should update in-place"
    assert "eigenvalues" in si.keyword_index
    s.close()


# ============= memory_store.py tests =============

def test_type_names_seeded_on_blocks():
    """Code/table/list/equation blocks get their type name in top_words."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("text:\n```python\nx=1\n```\n\n|A|B|\n|-|-|\n|1|2|")
    c = s.conn.cursor()
    c.execute("SELECT top_words FROM blocks WHERE type='code'")
    cw = json.loads(c.fetchone()["top_words"])
    assert "code" in cw
    c.execute("SELECT top_words FROM blocks WHERE type='table'")
    tw = json.loads(c.fetchone()["top_words"])
    assert "table" in tw
    s.close()

def test_equation_block_detected():
    """$$ equations are parsed as equation blocks."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("Newton:\n$$F = ma$$")
    tree = s.get_tree()
    children = tree[0].get("children", [])
    assert len(children) >= 1
    assert children[0]["type"] == "equation"
    s.close()

def test_owner_id_stored_in_blocks():
    """owner_id is passed through parse_and_store to add_block."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("test text", owner_id=12345)
    c = s.conn.cursor()
    c.execute("SELECT owner_id FROM blocks LIMIT 1")
    row = c.fetchone()
    assert row["owner_id"] == 12345
    s.close()

def test_owner_id_filtering_in_index():
    """build_match_index filters by owner_id."""
    s = MemoryStore(":memory:")
    p = BlockParser(s)
    p.parse_and_store("gradient descent:\n```python\nx=1\n```", owner_id=999)
    c = s.conn.cursor()
    c.execute("UPDATE sentences SET label='flagged',score=0.8")
    s.conn.commit()
    
    # User 12345 should not see user 999's sentences
    index = s.build_match_index(owner_id=12345)
    assert "gradient" not in index
    
    # User 999 should see their own
    index2 = s.build_match_index(owner_id=999)
    assert "gradient" in index2
    
    # No owner filter returns all
    index3 = s.build_match_index()
    assert "gradient" in index3
    s.close()


# ============= db.py tests =============

async def test_db_alias_methods():
    """Alias CRUD operations work correctly."""
    db = Database(":memory:")
    await db.connect()
    
    # Add aliases
    await db.add_alias(123, "bintang", 456)
    await db.add_alias(123, "bro", 789)
    
    # List
    aliases = await db.get_aliases(123)
    assert len(aliases) == 2
    
    # Resolve
    target = await db.resolve_alias(123, "bintang")
    assert target == 456
    
    # Overwrite
    await db.add_alias(123, "bintang", 999)
    target = await db.resolve_alias(123, "bintang")
    assert target == 999
    
    # Different owner
    assert await db.resolve_alias(456, "bintang") is None
    
    # Remove
    await db.remove_alias(123, "bro")
    assert await db.resolve_alias(123, "bro") is None
    assert len(await db.get_aliases(123)) == 1
    
    await db.close()


# ============= Run all tests =============

if __name__ == "__main__":
    import traceback
    
    sync_tests = [
        test_buffer_token_extracts_words,
        test_buffer_token_requires_word_boundary,
        test_buffer_token_filters_short_words,
        test_flush_emits_trailing_word,
        test_build_index_with_flagged_sentences,
        test_empty_index_when_no_flagged,
        test_cooldown_prevents_re_match,
        test_deep_tier_triggers_on_high_overlap,
        test_cooldown_decay,
        test_output_decay,
        test_rebuild_index_in_place,
        test_type_names_seeded_on_blocks,
        test_equation_block_detected,
        test_owner_id_stored_in_blocks,
        test_owner_id_filtering_in_index,
    ]
    
    passed = 0
    for fn in sync_tests:
        try:
            fn()
            print(f"  + {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  - {fn.__name__}: {e}")
            traceback.print_exc()
    
    # Run async test
    try:
        asyncio.run(test_db_alias_methods())
        print(f"  + test_db_alias_methods")
        passed += 1
    except Exception as e:
        print(f"  - test_db_alias_methods: {e}")
    
    total = len(sync_tests) + 1
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
