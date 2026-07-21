from collections import Counter
from stopwords import STOPWORDS
import re


def extract_top_words(text, content_type='paragraph', max_words=5):
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

    for ci in range(cols):
        vals = [r[ci] for r in data if ci < len(r)]
        # Skip non-empty values that are all numeric
        all_numeric = vals and all(is_numeric(v) for v in vals)
        # Skip classifier columns: 2-3 labels (Positive/Negative, High/Low)
        # across many rows aren't useful for retrieval.
        # Heuristic: unique_strings / total_rows < 0.5 → likely a classifier
        string_vals = [v for v in vals if v and not is_numeric(v)]
        uniq = len(set(string_vals))
        is_classifier = len(string_vals) >= 4 and uniq / len(string_vals) < 0.5
        
        if not is_classifier:
            # Keep the header — it describes the column's content
            candidates.append(header[ci])
            # Keep values only if not all-numeric and not a classifier column
            if vals and not all_numeric:
                for v in string_vals:
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
