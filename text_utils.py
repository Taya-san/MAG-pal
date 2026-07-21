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

    # Generic classifier words that describe row categories, not entities.
    # Columns where ALL values are from this set are classification labels
    # (Positive/Negative, High/Low, True/False) — useless for retrieval.
    # Entity names (Python, Rust, SGD, Adam, Transformer) are NOT here.
    # Generic classifier words describing row categories.
    # Columns where ALL values are from this set are classification labels
    # (Positive/Negative, True/False, High/Low) — useless for retrieval.
    # Entity names (Python, Rust, GPU, CNN, Transformer) are NOT here.
    _CLASSIFIER = {
        'positive', 'negative', 'neutral',
        'high', 'medium', 'low', 'mid',
        'true', 'false',
        'yes', 'no',  # 'no' is also a stopword — fine here
        'good', 'bad', 'poor', 'average', 'excellent', 'better', 'worse',
        'pass', 'fail', 'passed', 'failed',
        'success', 'failure', 'successful', 'unsuccessful',
        'active', 'inactive', 'activated', 'deactivated',
        'enabled', 'disabled',
        'on', 'off',
        'detected', 'undetected',
        'present', 'absent',
        'valid', 'invalid',
        'correct', 'incorrect', 'right', 'wrong',
        'accept', 'reject', 'accepted', 'rejected',
        'approve', 'deny', 'approved', 'denied',
        'include', 'exclude', 'included', 'excluded',
        'required', 'optional',
        'static', 'dynamic',
        'local', 'global',
        'simple', 'complex',
        'fast', 'slow', 'faster', 'slower',
        'small', 'large', 'big',
        'new', 'old',
        'single', 'multiple',
        'all', 'none', 'some', 'any', 'every',
        'other', 'another', 'same', 'different',
        'known', 'unknown',
        'normal', 'abnormal', 'anomaly',
        'safe', 'unsafe', 'danger',
        'clean', 'dirty',
        'full', 'empty',
        'visible', 'hidden',
        'public', 'private',
        'read', 'write',
        'input', 'output',
        'source', 'target',
        'easy', 'hard', 'difficult',
        'cheap', 'expensive',
        'free', 'paid', 'premium',
        'basic', 'pro',
        'stable', 'unstable', 'experimental',
        'todo', 'fixme', 'hack', 'xxx',
        'n/a', 'na',
    }

    for ci in range(cols):
        vals = [r[ci] for r in data if ci < len(r)]
        all_numeric = vals and all(is_numeric(v) for v in vals)
        string_vals = [v for v in vals if v and not is_numeric(v)]
        
        # Classifier detection: a column is a classifier if ALL its string values
        # are generic classifier words (Positive, True, High, etc.)
        # Entity names (Python, Rust, SGD, Transformer) are NOT classifier words.
        all_classifier = bool(string_vals) and all(v in _CLASSIFIER for v in string_vals)
        is_classifier = len(string_vals) >= 4 and all_classifier
        
        if not is_classifier:
            candidates.append(header[ci])
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
