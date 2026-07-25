from collections import Counter
from stopwords import STOPWORDS
import re


OBJECT_TYPE_KEYWORDS = {
    'table':    ['row', 'column', 'data', 'value', 'comparison', 'entry', 'feature', 'property'],
    'code':     ['code', 'function', 'class', 'implementation', 'example', 'syntax', 'program'],
    'list':     ['step', 'item', 'point', 'example', 'property', 'key'],
    'equation': ['formula', 'expression', 'variable', 'function', 'symbol', 'math'],
}

MIN_WORDS_FOR_RICH = 3


def merge_top_words(parent_words, child_words, max_words=12):
    seen = set(parent_words)
    merged = list(parent_words)
    for w in child_words:
        if w not in seen:
            seen.add(w)
            merged.append(w)
    return merged[:max_words]


def detect_bare_reference(top_words):
    return len(top_words) < MIN_WORDS_FOR_RICH


def inject_type_keywords(top_words, child_types, max_words=5):
    seen = set(top_words)
    result = list(top_words)
    for ctype in child_types:
        for kw in OBJECT_TYPE_KEYWORDS.get(ctype, []):
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
    return result[:max_words]


def filter_injected_keywords(words):
    """Remove generic structural keywords (function, example, row, column, etc.)
    from a word list, keeping only content-bearing topical keywords."""
    injected = set()
    for kws in OBJECT_TYPE_KEYWORDS.values():
        injected.update(kws)
    return [w for w in words if w not in injected]


def extract_top_words(text, content_type='paragraph', max_words=5):
    if content_type == 'table':
        return _extract_table_top_words(text, max_words)

    # Code and equation content is mostly symbols — not useful as keywords.
    # The structural type name (e.g. 'code', 'equation') is seeded separately
    # in add_block().
    if content_type in ('code', 'equation'):
        return []

    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    words = [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 2]
    if not words:
        return []
    return [w for w, _ in Counter(words).most_common(max_words)]


def _extract_table_top_words(text, max_words=5):
    rows = text.strip().split('\n')
    if not rows:
        return []

    grid = []
    for row in rows:
        cells = [c.strip().lower() for c in row.split('|')]
        # Remove leading/trailing empty cells (from outer ||)
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        if not cells:
            continue
        if all(c.replace('-', '') == '' for c in cells):
            continue
        grid.append(cells)

    if len(grid) < 2:
        return []

    header, data = grid[0], grid[1:]

    def is_numeric(val):
        # Pure numbers: ints (1), floats (0.95), negatives (-5), scientific (1e-3)
        # with optional commas (1,000,000), units (100M, 350GB), percent signs (95%)
        try:
            cleaned = val.replace(',', '').replace('%', '').replace(' ', '')
            # Strip trailing unit letters (B, M, K, GB, MB, KB, Hz, etc.)
            cleaned = re.sub(r'[a-z%]+$', '', cleaned)
            if cleaned:
                float(cleaned)
                return True
        except ValueError:
            pass
        # Ordinals: 1st, 2nd, 3rd, 4th, 5th
        if re.match(r'^\d+(?:st|nd|rd|th)$', val):
            return True
        # Pure non-alpha strings: 1.0, -5, ---, |||
        if val and not any(c.isalpha() for c in val):
            return True
        # Math symbols
        if any(c in val for c in {'^', '×', '÷', '√', '∫', '∑', 'π', 'Σ', 'μ'}):
            return True
        # Equation-like: O(n^3), sin(x), etc.
        if any(c in val for c in {'(', ')'}) and not any(c.isalpha() for c in re.sub(r'[osn]', '', val)):
            return True
        return False

    cols = min(len(header), max(len(r) for r in data))
    candidates = []

    # Generic classifier words that describe row categories, not entities.
    # Columns where ALL values are from this set are classification labels
    # (Positive/Negative, High/Low, True/False) — useless for retrieval.
    # Entity names (Python, Rust, SGD, Adam, Transformer) are NOT here.
    for ci in range(cols):
        vals = [r[ci] for r in data if ci < len(r)]
        all_numeric = vals and all(is_numeric(v) for v in vals)
        string_vals = [v for v in vals if v and not is_numeric(v)]
        
        # Classifier detection: if a column has many rows but very few unique
        # string values, it's likely a classification label column (Positive/
        # Negative, High/Low) — not useful for retrieval.
        # Threshold: >=10 rows AND unique/rows < 0.3 → classifier
        # This high bar ensures entity-name columns (Python/Rust with 2 uniq
        # across 8 rows = 0.25) are NOT caught. Only extreme cases like
        # 20 rows of True/False (2/20 = 0.1) get filtered.
        uniq = len(set(string_vals))
        is_classifier = len(string_vals) >= 20 and uniq / max(len(string_vals), 1) < 0.3
        
        if not is_classifier:
            candidates.append(header[ci])
            if vals and not all_numeric:
                for v in string_vals:
                    candidates.append(v)

    all_words = []
    for c in candidates:
        words = re.sub(r'[^\w\s]', ' ', c).split()
        all_words.extend([w for w in words if w not in STOPWORDS and len(w) >= 2])

    seen = set()
    result = []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            result.append(w)

    return result[:max_words]
