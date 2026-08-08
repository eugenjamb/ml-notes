"""Predict spam using CountVectorizer followed by TfidfTransformer."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression


# ============================================================
# Why use TfidfTransformer here?
# ============================================================
#
# TfidfTransformer is useful when we already have a matrix of word counts and
# want to convert those counts into TF-IDF weights as a separate stage.
#
# TfidfVectorizer performs both operations in one step:
#     raw text -> TF-IDF features
#
# This example keeps the operations separate:
#     raw text -> word counts -> TF-IDF features
#
# Keeping them separate is helpful for learning, inspecting count features, or
# adding TF-IDF to an existing count-based workflow.


# ============================================================
# 1. Create a tiny labelled dataset
# ============================================================

messages = [
    "win a cash prize now",
    "claim your free reward",
    "urgent offer click today",
    "winner collect your cash",
    "free voucher available now",
    "exclusive prize claim today",
    "team meeting at ten",
    "please review the report",
    "lunch with Sarah tomorrow",
    "your appointment is confirmed",
    "project notes are attached",
    "can we call this afternoon",
]

# 1 means spam; 0 means not spam.
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


# ============================================================
# 2. Convert the messages into raw word counts
# ============================================================

# CountVectorizer learns the vocabulary and counts each word.
count_vectorizer = CountVectorizer(lowercase=True)
count_features = count_vectorizer.fit_transform(messages)

print("Word-count matrix shape:", count_features.shape)
print("A few vocabulary words:")
print(count_vectorizer.get_feature_names_out()[:10])


# ============================================================
# 3. Transform word counts into TF-IDF weights
# ============================================================

# fit_transform() learns how common each word is across the training messages,
# then reduces the weight of common words and highlights distinctive words.
tfidf_transformer = TfidfTransformer()
tfidf_features = tfidf_transformer.fit_transform(count_features)

print("\nTF-IDF matrix shape:", tfidf_features.shape)


# ============================================================
# 4. Train the spam classifier
# ============================================================

# Logistic regression learns which TF-IDF features indicate spam.
model = LogisticRegression(random_state=42)
model.fit(tfidf_features, labels)


# ============================================================
# 5. Predict new messages
# ============================================================

new_messages = [
    "claim your exclusive cash prize",
    "please review the attached project report",
]

# New messages must pass through both fitted stages in the same order.
# transform() reuses the training vocabulary and learned TF-IDF weights.
new_count_features = count_vectorizer.transform(new_messages)
new_tfidf_features = tfidf_transformer.transform(new_count_features)
predictions = model.predict(new_tfidf_features)

print("\nPredictions:")
for message, prediction in zip(new_messages, predictions):
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"{label:8} | {message}")
