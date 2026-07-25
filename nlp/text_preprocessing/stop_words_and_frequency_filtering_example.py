import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# Stop Words And Frequency-Based Token Filtering
# ============================================================
#
# Stop words are common words such as "the" and "is". Removing them can reduce
# noise for some classical models, but can also remove grammar and negation.


documents = [
    "The product is easy to use and the product is reliable.",
    "The interface is easy but the documentation is not clear.",
    "Reliable support makes the service easy to recommend.",
    "The product is not reliable when the connection is slow.",
]


def tokenize(text):
    return re.findall(r"\b[a-z]+\b", text.lower())


tokenized_documents = [tokenize(document) for document in documents]
all_tokens = [
    token
    for document_tokens in tokenized_documents
    for token in document_tokens
]
token_frequencies = Counter(all_tokens)


print("Token frequencies:")
for token, count in token_frequencies.most_common():
    print(f"{token}: {count}")


# Keep negation because "not reliable" has the opposite meaning of "reliable".
domain_stop_words = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor"}

# A minimum frequency removes tokens seen only once in this small corpus.
# Real projects choose this threshold using training data only.
minimum_frequency = 2


def filter_tokens(tokens):
    return [
        token
        for token in tokens
        if token not in domain_stop_words
        and token_frequencies[token] >= minimum_frequency
    ]


print("\nFiltered documents:")
for original_document, tokens in zip(documents, tokenized_documents):
    filtered = filter_tokens(tokens)
    print("Original:", original_document)
    print("Filtered tokens:", filtered)
    print()


print(
    "Caution: frequency filtering and stop-word lists must be learned or chosen"
    " from training data. Looking at the final test set while choosing them is"
    " a form of data leakage."
)
