"""Small Bag-of-Words example that predicts whether a message is spam."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# Why use Bag of Words here?
# ============================================================
#
# These messages contain strong individual clues such as "prize", "cash",
# "meeting", and "report". Bag of Words (BoW) is a good simple choice because
# it counts words without needing to understand their order.
#
# Limitation: "free prize" and "prize free" have the same BoW features because
# word order is ignored. The n-gram example shows how to preserve local order.


# ============================================================
# 1. Create a tiny labelled dataset
# ============================================================

# 1 means spam; 0 means not spam.
training_messages = [
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
training_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


# ============================================================
# 2. Convert words into count features
# ============================================================

# CountVectorizer creates one column per known word.
# Each value says how often that word appears in a message.
vectorizer = CountVectorizer(lowercase=True)
training_features = vectorizer.fit_transform(training_messages)

print("Bag-of-Words vocabulary:")
print(vectorizer.get_feature_names_out())
print("\nFeature matrix shape:", training_features.shape)


# ============================================================
# 3. Train the classifier
# ============================================================

# Multinomial Naive Bayes is a small, common baseline for count-based text
# features. It learns which words occur more often in spam and non-spam.
model = MultinomialNB()
model.fit(training_features, training_labels)


# ============================================================
# 4. Predict new messages
# ============================================================

new_messages = [
    "claim your cash reward now",
    "the meeting notes are attached",
]

# transform(), rather than fit_transform(), keeps the training vocabulary.
new_features = vectorizer.transform(new_messages)
predictions = model.predict(new_features)

print("\nPredictions:")
for message, prediction in zip(new_messages, predictions):
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"{label:8} | {message}")
