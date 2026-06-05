# keywords.py
# Extracts important words from your messages for the keyword learning system.
#
# Why no spaCy?
#   spaCy is a 50MB NLP library. For extracting keywords from
#   10-word Discord messages, a simple regex + stopword filter
#   gives 80% of the value at 0% of the dependency cost.
#
# How it works:
#   1. Find all 3+ letter words using regex
#   2. Filter out common English words (stopwords)
#   3. Count remaining words, return top N
#
# The extracted keywords are stored in the database and fed
# into the AI prompt so it knows what topics you discuss.

import re
from collections import Counter

# STOPWORDS: Common English words that aren't useful as keywords.
# Words shorter than 3 letters are already filtered by the regex.
# This list includes contractions and internet slang (lol, idk, tbh).
# If you find a word is being incorrectly filtered, add it here.
# If you find a word is NOT being filtered that should be, add it here.

STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "some", "any", "no", "not", "none",
    "and", "or", "but", "if", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most",
    "other", "same", "so", "than", "too", "very", "just",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "get", "got", "getting", "go", "goes", "went", "going",
    "say", "says", "said",
    "make", "makes", "made",
    "know", "knows", "knew",
    "think", "thinks", "thought",
    "take", "takes", "took",
    "see", "sees", "saw",
    "come", "comes", "came",
    "want", "wants", "wanted",
    "look", "looks", "looked",
    "would", "could", "should", "might", "must", "shall", "will",
    "can", "may", "like", "need", "let",
    "even", "still", "also", "well", "really", "actually",
    "yeah", "yes", "no", "oh", "ok", "okay", "hey", "hi", "hello",
    "thanks",
    "lol", "lmao", "idk", "imo", "tbh", "btw", "omg", "wtf",
    "dont", "doesnt", "didnt", "wont", "wouldnt", "couldnt",
    "shouldnt", "cant", "isnt", "arent", "wasnt", "werent",
    "havent", "hasnt", "hadnt",
    "im", "ive", "id", "youre", "youve", "youll", "theyll",
    "its", "thats", "whats", "whos", "wheres", "theres", "heres",
    "hes", "shes",
}


def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    # Step 1: Find all alphabetic words that are 3+ characters long
    # The regex matches word boundaries with \b and requires [a-zA-Z]{3,}
    # This automatically filters out "a", "an", "to", "of", "my", etc.
    # Text is lowered so "Rust" and "rust" are treated the same.
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    # Step 2: Remove stopwords (common words that aren't meaningful topics)
    important = [w for w in words if w not in STOPWORDS]

    # Step 3: Count occurrences and return the most common
    # Counter.most_common() handles sorting by frequency
    # Multiple mentions in one message = stronger signal
    counter = Counter(important)
    return [word for word, _ in counter.most_common(max_keywords)]
