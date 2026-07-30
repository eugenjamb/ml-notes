"""Predict spam by averaging Word2Vec word embeddings."""

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    from gensim.models import Word2Vec
except ImportError as error:
    raise SystemExit(
        "This example needs gensim. Install it with: py -3 -m pip install gensim"
    ) from error


# ============================================================
# Why use Word2Vec here?
# ============================================================
#
# BoW and TF-IDF treat every word as a separate feature. Word2Vec instead
# learns dense vectors in which words used in similar contexts can have
# similar representations. This can help connect related words such as
# "prize", "reward", and "voucher".
#
# A real project normally uses much more text or pretrained embeddings. This
# tiny dataset is intentionally only a demonstration of the workflow.


# ============================================================
# 1. Create and tokenize a tiny dataset
# ============================================================

messages = [
    "win cash prize now",
    "claim free reward today",
    "collect cash voucher now",
    "urgent prize offer today",
    "team meeting at ten",
    "review project report today",
    "lunch with Sarah tomorrow",
    "appointment confirmed at ten",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]

# Word2Vec expects a list of token lists rather than complete strings.
tokenized_messages = [message.lower().split() for message in messages]


# ============================================================
# 2. Train the Word2Vec model
# ============================================================

# vector_size=20 keeps this teaching example small.
# min_count=1 keeps every word because the dataset is tiny.
# workers=1 and seed=42 make the result more reproducible.
word_model = Word2Vec(
    sentences=tokenized_messages,
    vector_size=20,
    window=3,
    min_count=1,
    workers=1,
    seed=42,
    epochs=100,
)


def average_word_vectors(tokens):
    """Represent one message with the mean vector of its known words."""
    known_vectors = [
        word_model.wv[token] for token in tokens if token in word_model.wv
    ]

    # Return zeros if none of the message's words are in the vocabulary.
    if not known_vectors:
        return np.zeros(word_model.vector_size)
    return np.mean(known_vectors, axis=0)


# ============================================================
# 3. Train a classifier using the averaged vectors
# ============================================================

message_features = np.vstack(
    [average_word_vectors(tokens) for tokens in tokenized_messages]
)

classifier = LogisticRegression(random_state=42)
classifier.fit(message_features, labels)


# ============================================================
# 4. Predict new messages
# ============================================================

new_messages = ["claim cash reward now", "project meeting tomorrow"]
new_features = np.vstack(
    [average_word_vectors(message.lower().split()) for message in new_messages]
)
predictions = classifier.predict(new_features)

print("Predictions:")
for message, prediction in zip(new_messages, predictions):
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"{label:8} | {message}")
