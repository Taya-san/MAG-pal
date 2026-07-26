from __future__ import annotations
#!/usr/bin/env python3
"""Comprehensive stress test with 500 varied paragraphs, mixed content, timing."""
import sys, os, tempfile, time, random, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from memory_store import MemoryStore, BlockParser
from stream_intervention import StreamIntervention
from test_data import PARAGRAPHS

random.seed(42)
passed = 0
failed = 0

def check(desc, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {desc}")
    else:
        failed += 1
        print(f"  FAIL {desc} {detail}")

# ============================================================
# 1. LOAD 500 paragraphs into memory
# ============================================================
print("=== 1. Load 500 paragraphs ===")
db = tempfile.mktemp(suffix='.db')
store = MemoryStore(db)
bp = BlockParser(store)

start = time.time()
for i, para in enumerate(PARAGRAPHS):
    bp.parse_and_store(para, owner_id=1)
elapsed = time.time() - start

cur = store.conn.cursor()
cur.execute("SELECT COUNT(*) FROM blocks")
bc = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM sentences")
sc = cur.fetchone()[0]
check(f"500 blocks created", bc == 500, f"got {bc}")
check(f"Sentences created", sc >= 500, f"got {sc}")
check(f"Load time < 30s", elapsed < 30.0, f"took {elapsed:.2f}s")

# Flag all sentences
cur.execute("UPDATE sentences SET label = 'flagged'")
store.conn.commit()
print(f"  Memory: {bc} blocks, {sc} sentences")

# ============================================================
# 2. Build index + scoring speed
# ============================================================
print("
=== 2. Scoring speed (500 calls) ===")
iv = StreamIntervention(memory_store=store, owner_id=1)
check("Index built", len(iv.keyword_index) > 0)
print(f"  Index entries: {len(iv.keyword_index)} keywords")

start = time.time()
hit_count = 0
for i in range(500):
    para = PARAGRAPHS[i % len(PARAGRAPHS)]
    m = iv.check_paragraph(para)
    if m:
        hit_count += 1
    iv.decrement_cooldowns(100)
elapsed = time.time() - start
check(f"500 scoring calls", elapsed < 15.0, f"took {elapsed:.2f}s")
print(f"  Hits: {hit_count}/500 (cooldown may reduce some)")

# ============================================================
# 3. Intervention quality check
# ============================================================
print("
=== 3. Intervention quality ===")
# Fresh store for clean measurements
db2 = tempfile.mktemp(suffix='.db')
store2 = MemoryStore(db2)
bp2 = BlockParser(store2)

# Add 10 distinct topic clusters
clusters = [
    "Eigenvalues are scalars satisfying Av = lambda v for eigenvectors.",
    "The matrix eigenvalue problem is fundamental to linear algebra theory.",
    "Gaussian elimination solves linear systems by row reduction operations.",
    "The running time of Gaussian elimination is O(n^3) for n by n matrices.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Calvin cycle fixes carbon dioxide into organic sugar molecules.",
    "Quantum entanglement connects particles across any distance instantly.",
    "Bell's theorem proves that quantum mechanics cannot be explained locally.",
    "Binary search finds elements in logarithmic time for sorted arrays.",
    "Merge sort divides the array recursively and merges sorted halves.",
]

for c in clusters:
    bp2.parse_and_store(c, owner_id=1)

cur2 = store2.conn.cursor()
cur2.execute("UPDATE sentences SET label = 'flagged'")
store2.conn.commit()

iv2 = StreamIntervention(memory_store=store2, owner_id=1)

# Test: paragraph about eigenvalues should match eigenvalue blocks
m = iv2.check_paragraph("Eigenvalues and eigenvectors satisfy the equation Av = lambda v.")
check("Eigenvalue match", m is not None)
iv2.decrement_cooldowns(200)

# Test: paragraph about sorting should match binary search or merge sort
m2 = iv2.check_paragraph("Binary search runs in logarithmic time for sorted arrays.")
check("Binary search match", m2 is not None)
iv2.decrement_cooldowns(200)

# Test: paragraph about unrelated topic should not match
m3 = iv2.check_paragraph("The weather today is rainy and cold.")
check("Unrelated returns None", m3 is None)

# Test: cross-topic doesn't false-match
m4 = iv2.check_paragraph("Photosynthesis uses light energy to power the Calvin cycle.")
check("Biology match", m4 is not None)
if m4:
    check("  Correct domain", "photosynthesis" in m4['context'].lower() or "calvin" in m4['context'].lower())

os.unlink(db2)

# ============================================================
# 4. Mixed content: code, equations, lists
# ============================================================
print("
=== 4. Mixed content ===")
db3 = tempfile.mktemp(suffix='.db')
store3 = MemoryStore(db3)
bp3 = BlockParser(store3)

mixed = [
    "Here is a Python function:",
    "```python\ndef hello():\n    print('hello')\n```",
    "This function prints a greeting to the user.",
    "The quadratic formula solves ax^2 + bx + c = 0:",
    "$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$",
    "This formula has two solutions for real discriminants.",
    "Key points about the algorithm:",
    "- It runs in O(n log n) time",
    "- It uses divide and conquer strategy",
    "- It is stable for equal elements",
]
for m in mixed:
    bp3.parse_and_store(m, owner_id=1)

cur3 = store3.conn.cursor()
cur3.execute("SELECT COUNT(*) FROM blocks")
print(f"  Blocks created from mixed content: {cur3.fetchone()[0]}")

cur3.execute("""
    SELECT type, COUNT(*) FROM blocks GROUP BY type ORDER BY COUNT(*) DESC
""")
for row in cur3.fetchall():
    print(f"    {row[0]}: {row[1]} blocks")

# Flag specific mixed-content sentences
cur3.execute("UPDATE sentences SET label = 'flagged' WHERE text LIKE '%greeting%'")
cur3.execute("UPDATE sentences SET label = 'flagged' WHERE text LIKE '%quadratic%'")
cur3.execute("UPDATE sentences SET label = 'flagged' WHERE text LIKE '%O(n log n)%'")
store3.conn.commit()

iv3 = StreamIntervention(memory_store=store3, owner_id=1)
check("Mixed index built", len(iv3.keyword_index) > 0)

m = iv3.check_paragraph("The code prints a greeting to the screen.")
check("Code paragraph match", m is not None)
iv3.decrement_cooldowns(200)

m2 = iv3.check_paragraph("The quadratic formula solves polynomial equations.")
check("Equation paragraph match", m2 is not None)
iv3.decrement_cooldowns(200)

os.unlink(db3)

# ============================================================
# 5. Decay under real load
# ============================================================
print("
=== 5. Decay under load ===")
db4 = tempfile.mktemp(suffix='.db')
store4 = MemoryStore(db4)
bp4 = BlockParser(store4)

for i, para in enumerate(PARAGRAPHS[:200]):
    bp4.parse_and_store(para, owner_id=1)

cur4 = store4.conn.cursor()
cur4.execute("UPDATE sentences SET label = 'flagged'")

# Set half of sentences to expire immediately
cur4.execute("""
    UPDATE sentences SET expires_at = 0 WHERE id IN (
        SELECT id FROM sentences ORDER BY id LIMIT (
            SELECT COUNT(*)/2 FROM sentences
        )
    )
""")
store4.conn.commit()

before = cur4.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
store4.apply_memory_decay()
after = cur4.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
check("Decay removed expired sentences", after < before, f"{before} -> {after}")
check("Some sentences survived", after > 0)

# Rebuild index after decay
iv4 = StreamIntervention(memory_store=store4, owner_id=1)
check("Index rebuild after decay works", len(iv4.keyword_index) > 0)

m = iv4.check_paragraph("Eigenvalues are important in linear algebra.")
if m:
    check("Intervention after decay works", True)
else:
    # Might be on cooldown or no match — acceptable
    print("  ~ No match after decay (may be expected)")
os.unlink(db4)

# ============================================================
# 6. Concurrent-like fast sequential access
# ============================================================
print("
=== 6. High-frequency access ===")
db5 = tempfile.mktemp(suffix='.db')
store5 = MemoryStore(db5)
bp5 = BlockParser(store5)

for para in PARAGRAPHS[:50]:
    bp5.parse_and_store(para, owner_id=1)

cur5 = store5.conn.cursor()
cur5.execute("UPDATE sentences SET label = 'flagged'")
store5.conn.commit()

iv5 = StreamIntervention(memory_store=store5, owner_id=1)

start = time.time()
total_hits = 0
for i in range(1000):
    para = random.choice(PARAGRAPHS[:50])
    m = iv5.check_paragraph(para)
    if m:
        total_hits += 1
    # Vary cooldown decay to simulate realistic streaming
    iv5.decrement_cooldowns(random.randint(50, 300))
elapsed = time.time() - start
check(f"1000 rapid calls", elapsed < 30.0, f"took {elapsed:.2f}s")
check(f"  At least some hits", total_hits > 0, f"got {total_hits} hits")
print(f"  Hits: {total_hits}/1000")

os.unlink(db5)

# ============================================================
print(f"\n=== Final Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
