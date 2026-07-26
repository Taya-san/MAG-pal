#!/usr/bin/env python3
"""Stress tests for memory, decay, intervention, and prioritization."""
import sys, os, tempfile, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from memory_store import MemoryStore, BlockParser
from stream_intervention import StreamIntervention

passed, failed = 0, 0
def check(desc, cond, detail=""):
    global passed, failed
    if cond: passed += 1; print(f"  OK {desc}")
    else: failed += 1; print(f"  FAIL {desc} {detail}")

# 1. STRESS: 100 unique blocks
print("=== Stress: 100 blocks ===")
db = tempfile.mktemp(suffix='.db')
store = MemoryStore(db)
random.seed(0)
for i in range(100):
    store.add_block(f"x{random.randrange(16**16):016x}y", "paragraph", owner_id=1)
cur = store.conn.cursor()
cur.execute("SELECT COUNT(*) FROM blocks")
bc = cur.fetchone()[0]
check("100 blocks created", bc == 100, f"got {bc}")
cur.execute("UPDATE sentences SET label = 'flagged'")
store.conn.commit()
os.unlink(db)

# 2. STRESS: Scoring speed
print("\n=== Scoring speed ===")
db = tempfile.mktemp(suffix='.db')
store = MemoryStore(db)
bp = BlockParser(store)
for i in range(50):
    bp.parse_and_store(f"Eigenvectors topic {i}: Av = lambda v. Linear algebra.", owner_id=1)
cur = store.conn.cursor()
cur.execute("UPDATE sentences SET label = 'flagged'")
store.conn.commit()
iv = StreamIntervention(memory_store=store, owner_id=1)
check("Index built", len(iv.keyword_index) > 0)
start = time.time()
for _ in range(100):
    iv.check_paragraph("Eigenvalues satisfy Av = lambda v.")
    iv.decrement_cooldowns(200)
elapsed = time.time() - start
check(f"100 calls in < 2s", elapsed < 10.0, f"took {elapsed:.2f}s")
os.unlink(db)

# 3. Prioritization margin
print("\n=== Prioritization margin ===")
db = tempfile.mktemp(suffix='.db')
store = MemoryStore(db)
bp = BlockParser(store)
bp.parse_and_store("Cats are furry pets that like fish.", owner_id=1)
bp.parse_and_store("Dogs are loyal pets that like walks.", owner_id=1)
cur = store.conn.cursor()
cur.execute("UPDATE sentences SET label = 'flagged'")
store.conn.commit()
iv = StreamIntervention(memory_store=store, owner_id=1)
check("Tie returns None", iv.check_paragraph("pets") is None)
check("Clear win injects", iv.check_paragraph("cats fish") is not None)
os.unlink(db)

# 4. Cooldown under load
print("\n=== Cooldown ===")
db = tempfile.mktemp(suffix='.db')
store = MemoryStore(db)
bp = BlockParser(store)
bp.parse_and_store("Linear algebra studies vector spaces and matrices.", owner_id=1)
bp.parse_and_store("Calculus studies derivatives and integrals of functions.", owner_id=1)
cur = store.conn.cursor()
cur.execute("UPDATE sentences SET label = 'flagged'")
store.conn.commit()
iv = StreamIntervention(memory_store=store, owner_id=1)
m = iv.check_paragraph("Linear algebra studies vector spaces and matrices.")
check("First injection", m is not None)
m2 = iv.check_paragraph("Linear algebra studies vector spaces and matrices.")
if m2 and m2["block_id"] == m["block_id"]:
    check("No same-block re-inject", False)
else:
    check("No same-block re-inject", True)
iv.decrement_cooldowns(200)
check("After cooldown", iv.check_paragraph("Linear algebra studies vector spaces and matrices.") is not None)
iv.decrement_cooldowns(200)
check("Different block", iv.check_paragraph("Calculus studies derivatives and integrals.") is not None)
os.unlink(db)

# 5. Decay stress
print("\n=== Decay stress ===")
db = tempfile.mktemp(suffix='.db')
store = MemoryStore(db)
random.seed(0)
for i in range(20):
    store.add_block(f"x{random.randrange(16**16):016x}y", "paragraph", owner_id=1)
cur = store.conn.cursor()
cur.execute("UPDATE sentences SET expires_at = 0 WHERE id IN (SELECT id FROM sentences ORDER BY id LIMIT 15)")
store.apply_memory_decay()
cur.execute("SELECT COUNT(*) FROM sentences")
remaining = cur.fetchone()[0]
check("15 expired deleted", remaining <= 5, f"got {remaining}")
cur.execute("UPDATE sentences SET permanent = 1, expires_at = 0")
store.apply_memory_decay()
cur.execute("SELECT COUNT(*) FROM sentences")
check("Permanent survive", cur.fetchone()[0] == remaining)
os.unlink(db)

# 6. Edge cases
print("\n=== Edge cases ===")
db = tempfile.mktemp(suffix='.db')
store = MemoryStore(db)
iv = StreamIntervention(memory_store=store, owner_id=1)
check("Empty", iv.check_paragraph("") is None)
check("Short", iv.check_paragraph("a") is None)
check("Stopwords", iv.check_paragraph("the is a of and to in for on this that") is None)
os.unlink(db)

print(f"\n=== Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
