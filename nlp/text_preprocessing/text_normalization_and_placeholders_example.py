import html
import re
import unicodedata


# ============================================================
# Text Normalization And Semantic Placeholders
# ============================================================
#
# Normalization makes inconsistent raw inputs more uniform. The correct choices
# depend on the task: sentiment analysis may need punctuation and negation,
# while document matching may benefit from stronger normalization.


raw_documents = [
    "Café prices increased by £5. Visit https://example.com/prices",
    "Email Help@Example.com — I can't access order ORD-1042!",
    "<p>The delivery isn't late; it arrives on 25/07/2026.</p>",
]


# Expanding common contractions can make negation explicit. This tiny mapping is
# suitable for teaching only; real English has many ambiguous contractions.
contractions = {
    "can't": "can not",
    "isn't": "is not",
    "wasn't": "was not",
    "don't": "do not",
    "didn't": "did not",
}


def normalize_text(text):
    """Return one normalized string while preserving useful information."""
    # Convert HTML entities such as &amp; back to their characters.
    normalized = html.unescape(text)

    # Remove HTML tags. A real HTML document should use an HTML parser because
    # regular expressions cannot reliably understand every HTML structure.
    normalized = re.sub(r"<[^>]+>", " ", normalized)

    # NFKC normalizes compatible Unicode forms, such as full-width characters.
    # It does not remove accents; café remains café.
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.lower()

    for contraction, expanded_form in contractions.items():
        normalized = normalized.replace(contraction, expanded_form)

    # Replace variable values with stable semantic placeholders. This reduces
    # vocabulary size while preserving the type of information that occurred.
    normalized = re.sub(r"https?://\S+|www\.\S+", " <URL> ", normalized)
    normalized = re.sub(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        " <EMAIL> ",
        normalized,
    )
    normalized = re.sub(r"\bORD-\d+\b", " <ORDER_ID> ", normalized, flags=re.I)
    normalized = re.sub(
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        " <DATE> ",
        normalized,
    )
    normalized = re.sub(r"[£$€]\s*\d+(?:\.\d+)?", " <MONEY> ", normalized)

    # Keep letters, digits, apostrophes, angle brackets used by placeholders,
    # and whitespace. Replace other punctuation with spaces.
    normalized = re.sub(r"[^\w\s'<>-]", " ", normalized)

    # Collapse repeated whitespace and remove whitespace at both ends.
    return re.sub(r"\s+", " ", normalized).strip()


for document in raw_documents:
    print("Raw:")
    print(document)
    print("Normalized:")
    print(normalize_text(document))
    print()


print(
    "Important: fit preprocessing decisions on the task, not habit. For example,"
    " removing accents may help search matching but can damage names and meaning"
    " in multilingual text."
)

