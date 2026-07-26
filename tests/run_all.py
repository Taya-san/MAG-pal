#!/usr/bin/env python3
"""Run all tests: core + decay + integration checks."""
import sys, os, tempfile, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

passed = 0
failed = 0

def run_test(name, func):
    global passed, failed
    try:
        func()
        passed += 1
        print(f"  OK {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL {name}: {e}")

# --- Core tests ---
spec = importlib.util.spec_from_file_location("test_core", 
    os.path.join(os.path.dirname(__file__), "test_core.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("=== Core Tests ===")
for name in dir(mod):
    if name.startswith("test_") and callable(getattr(mod, name)):
        run_test(name, getattr(mod, name))

# --- Decay tests ---
print("\n=== Decay Tests ===")
from memory_store import MemoryStore, BlockParser
from stream_intervention import StreamIntervention

def test_decay_schema():
    db = tempfile.mktemp(suffix='.db')
    s = MemoryStore(db)
    cur = s.conn.cursor()
    cur.execute("PRAGMA table_info(sentences)")
    cols = {r[1] for r in cur.fetchall()}
    assert "access_count" in cols
    assert "expires_at" in cols
    assert "permanent" in cols
    os.unlink(db)

def test_decay_ttl_set():
    db = tempfile.mktemp(suffix='.db')
    s = MemoryStore(db)
    BlockParser(s).parse_and_store("Eigenvalues are scalars.", owner_id=1)
    cur = s.conn.cursor()
    cur.execute("SELECT expires_at FROM sentences")
    assert cur.fetchone()[0] is not None
    os.unlink(db)

def test_decay_touch():
    db = tempfile.mktemp(suffix='.db')
    s = MemoryStore(db)
    BlockParser(s).parse_and_store("Test.", owner_id=1)
    cur = s.conn.cursor()
    cur.execute("SELECT id FROM sentences")
    sid = cur.fetchone()[0]
    s._touch_sentences({sid})
    cur.execute("SELECT access_count FROM sentences WHERE id = ?", (sid,))
    assert cur.fetchone()[0] == 1
    os.unlink(db)

def test_decay_permanent_survives():
    db = tempfile.mktemp(suffix='.db')
    s = MemoryStore(db)
    BlockParser(s).parse_and_store("Test.", owner_id=1)
    cur = s.conn.cursor()
    cur.execute("SELECT id FROM sentences")
    sid = cur.fetchone()[0]
    cur.execute("UPDATE sentences SET permanent = 1, expires_at = 0 WHERE id = ?", (sid,))
    s.apply_memory_decay()
    cur.execute("SELECT id FROM sentences WHERE id = ?", (sid,))
    assert cur.fetchone() is not None
    os.unlink(db)

def test_decay_expired_deleted():
    db = tempfile.mktemp(suffix='.db')
    s = MemoryStore(db)
    BlockParser(s).parse_and_store("Test.", owner_id=1)
    cur = s.conn.cursor()
    cur.execute("SELECT id FROM sentences")
    sid = cur.fetchone()[0]
    cur.execute("UPDATE sentences SET expires_at = 0 WHERE id = ?", (sid,))
    s.apply_memory_decay()
    cur.execute("SELECT id FROM sentences WHERE id = ?", (sid,))
    assert cur.fetchone() is None
    os.unlink(db)

def test_decay_intervention_works():
    db = tempfile.mktemp(suffix='.db')
    s = MemoryStore(db)
    bp = BlockParser(s)
    bp.parse_and_store("Gaussian elimination solves systems.", owner_id=1)
    cur = s.conn.cursor()
    cur.execute("UPDATE sentences SET label = 'flagged'")
    cur.execute("UPDATE sentences SET permanent = 1")
    s.conn.commit()
    iv = StreamIntervention(memory_store=s, owner_id=1)
    m = iv.check_paragraph("Gaussian elimination solves linear systems.")
    assert m is not None
    os.unlink(db)

run_test("decay_schema", test_decay_schema)
run_test("decay_ttl_set", test_decay_ttl_set)
run_test("decay_touch", test_decay_touch)
run_test("decay_permanent_survives", test_decay_permanent_survives)
run_test("decay_expired_deleted", test_decay_expired_deleted)
run_test("decay_intervention_works", test_decay_intervention_works)


# --- Stress tests ---
print("\n=== Stress Tests ===")
import subprocess
result = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), "stress_test.py")],
    capture_output=True, text=True
)
for line in result.stdout.split("\n"):
    if "OK" in line or "FAIL" in line:
        print(line)
    elif "Results" in line:
        print(line)
        import re
        m = re.search(r"(\d+) passed, (\d+) failed", line)
        if m:
            passed += int(m.group(1))
            failed += int(m.group(2))
if result.returncode != 0:
    for line in result.stderr.split("\n"):
        if "FAIL" in line:
            print(line)

print(f"\n=== Final Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
