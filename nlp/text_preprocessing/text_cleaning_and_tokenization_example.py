import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# Text Cleaning And Tokenization
# ============================================================
#
# Raw text often contains inconsistent case, URLs, punctuation, and spacing.
# Preprocessing makes those inputs more consistent before feature extraction.
# There is no universal cleaning recipe: removing punctuation, numbers, or stop
# words can destroy useful meaning for some tasks.


documents = [
    "I LOVED the new camera! Photos were sharp. https://example.com/review",
    "Delivery wasn't fast, but the product is very good.",
    "Support emailed me at Help@Example.com -- no reply after 3 days.",
]


# Keep negation words because "not good" means something different from "good".
# sklearn's default stop-word set includes some words that may matter, so a real
# project should review this list for its domain.
custom_stop_words = set(ENGLISH_STOP_WORDS) - {"no", "not"}


def clean_and_tokenize(text):
    """Normalize one document and return a list of useful word tokens."""
    # lower() makes Camera and camera share one vocabulary entry.
    normalized = text.lower()

    # Replace URLs and email addresses with semantic placeholder tokens. A
    # placeholder preserves the fact that an address was present.
    normalized = re.sub(r"https?://\S+|www\.\S+", " URL ", normalized)
    normalized = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", " EMAIL ", normalized)

    # This tokenizer keeps words, apostrophes inside words, and numbers.
    # \b marks word boundaries; [a-z]+ matches one or more letters.
    tokens = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b|\b\d+\b", normalized)

    # Stop-word removal can reduce noise in bag-of-words models. Do not apply it
    # automatically to every task; modern neural models often keep stop words.
    filtered_tokens = [
        token
        for token in tokens
        if token not in custom_stop_words or token in {"url", "email"}
    ]
    return filtered_tokens


for document_number, document in enumerate(documents, start=1):
    tokens = clean_and_tokenize(document)
    print(f"Document {document_number}")
    print("Raw:", document)
    print("Tokens:", tokens)
    print("Normalized text:", " ".join(tokens))
    print()


print(
    "Note: stemming shortens words using rules, while lemmatization uses"
    " vocabulary and grammar to find dictionary forms. Libraries such as NLTK"
    " or spaCy provide those tools; avoid inventing crude suffix rules in a"
    " production pipeline."
)

