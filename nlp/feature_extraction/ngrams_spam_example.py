"""Small n-gram example that predicts whether a message is spam."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# Why use n-grams here?
# ============================================================
#
# In this example, short phrases are more useful than isolated words.
# Phrases such as "click here", "limited offer", and "claim now" are common
# spam patterns. Bigrams keep pairs of neighbouring words, so the model can
# distinguish phrases even when their individual words appear elsewhere.
#
# Trade-off: n-grams create more features and need more data than simple BoW.


# ============================================================
# 1. Create a tiny labelled dataset
# ============================================================

# 1 means spam; 0 means not spam.
training_messages = [
    "click here to win",
    "click here for cash",
    "limited offer ends today",
    "limited offer claim now",
    "claim now for reward",
    "claim now your prize",
    "click the meeting link",
    "the offer was accepted",
    "limited seats were booked",
    "claim the travel expenses",
    "reward the team today",
    "win the match today",
]
training_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


# ============================================================
# 2. Convert phrases into bigram count features
# ============================================================

# ngram_range=(2, 2) means: create features from exactly two adjacent words.
# For example, "click here today" contains "click here" and "here today".
vectorizer = CountVectorizer(ngram_range=(2, 2), lowercase=True)
training_features = vectorizer.fit_transform(training_messages)

print("Bigram vocabulary:")
print(vectorizer.get_feature_names_out())
print("\nFeature matrix shape:", training_features.shape)


# ============================================================
# 3. Train the classifier
# ============================================================

# The classifier now learns from phrase counts instead of single-word counts.
model = MultinomialNB()
model.fit(training_features, training_labels)


# ============================================================
# 4. Predict new messages
# ============================================================

new_messages = [
    "limited offer click here",
    "please click the meeting link",
]

# Apply the already learned bigram vocabulary to the new messages.
new_features = vectorizer.transform(new_messages)
predictions = model.predict(new_features)

print("\nPredictions:")
for message, prediction in zip(new_messages, predictions):
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"{label:8} | {message}")
