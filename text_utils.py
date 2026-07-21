from collections import Counter
from stopwords import STOPWORDS
import re


def extract_top_words(text, max_words=5, content_type='paragraph'):
    if content_type == 'table':
        return _extract_table_top_words(text, max_words)

    cleaned = re.sub(r'[^a-z\s]', ' ', text.lower())
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
        cells = [c.strip().lower() for c in row.split('|') if c.strip()]
        if not cells:
            continue
        if all(c.replace('-', '') == '' for c in cells):
            continue
        grid.append(cells)

    if len(grid) < 2:
        return []

    header, data = grid[0], grid[1:]

    def is_numeric(val):
        try:
            float(val.replace(',', ''))
            return True
        except ValueError:
            pass
        if val and not any(c.isalpha() for c in val):
            return True
        if any(c in val for c in {'^', '×', '÷', '√', '∫', '∑', 'π'}):
            return True
        if any(c in val for c in {'(', ')'}) and not any(c.isalpha() for c in re.sub(r'[osn]', '', val)):
            return True
        return False

    cols = min(len(header), max(len(r) for r in data))
    candidates = []

    for ci in range(cols):
        vals = [r[ci] for r in data if ci < len(r)]
        if vals and all(is_numeric(v) for v in vals):
            continue
        candidates.append(header[ci])
        for v in vals:
            if not is_numeric(v):
                candidates.append(v)

    all_words = []
    for c in candidates:
        words = re.sub(r'[^a-z\s]', ' ', c).split()
        all_words.extend([w for w in words if w not in STOPWORDS and len(w) >= 2])

    seen = set()
    result = []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            result.append(w)

    return result[:max_words]
