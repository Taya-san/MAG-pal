from __future__ import annotations
from __future__ import annotations

import re
from collections import Counter

from stopwords import STOPWORDS


def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    important = [w for w in words if w not in STOPWORDS]
    counter = Counter(important)
    return [word for word, _ in counter.most_common(max_keywords)]
