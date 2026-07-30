"""Predict spam with TF-IDF features and a small classifier."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# Why use TF-IDF here?
# ============================================================
#
# Bag of Words gives every word a raw count. TF-IDF also lowers the importance
# of words that appear in many messages and highlights more distinctive words.
# That makes terms such as "prize" and "voucher" more useful than common terms.


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
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


# ============================================================
# 2. Create TF-IDF features
# ============================================================

# fit_transform() learns the vocabulary and calculates a TF-IDF weight for
# each word. A larger weight means the word is more distinctive in a message.
vectorizer = TfidfVectorizer(lowercase=True)
message_features = vectorizer.fit_transform(messages)

print("TF-IDF matrix shape:", message_features.shape)


# ============================================================
# 3. Train the classifier
# ============================================================

# Logistic regression learns which TF-IDF weights indicate each class.
model = LogisticRegression(random_state=42)
model.fit(message_features, labels)


# ============================================================
# 4. Predict new messages
# ============================================================

new_messages = [
    "claim your exclusive cash prize",
    "please review the attached project report",
]
new_features = vectorizer.transform(new_messages)
predictions = model.predict(new_features)

print("\nPredictions:")
for message, prediction in zip(new_messages, predictions):
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"{label:8} | {message}")
