from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ============================================================
# Multinomial Naive Bayes For Text Classification
# ============================================================
#
# What this method does:
# - turns text into word-count features
# - estimates how strongly each word is associated with each class
# - predicts the most likely class for a new piece of text
#
# Why we use it:
# - it is fast and simple
# - it works surprisingly well as a baseline for text tasks
# - it introduces the bag-of-words idea clearly


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "generated_reviews_multinomial_nb.csv"


# ============================================================
# Load the review dataset
# ============================================================

# The CSV file contains two columns:
# 1. review -> the text that a customer wrote
# 2. sentiment -> the label we want to predict
reviews_df = pd.read_csv(DATA_PATH)

print("First 5 rows of the dataset:")
print(reviews_df.head())

print("\nClass distribution:")
print(reviews_df["sentiment"].value_counts())


# ============================================================
# Prepare the text and labels
# ============================================================

# X contains the review text.
# y contains the correct sentiment label for each review.
X = reviews_df["review"]
y = reviews_df["sentiment"]


# ============================================================
# Split the dataset into training and testing parts
# ============================================================

# The model learns from X_train and y_train.
# We keep X_test and y_test separate so we can measure performance fairly.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.2 reserves 20% of the reviews for testing.
    test_size=0.2,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y preserves class balance in train and test.
    stratify=y,
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))


# ============================================================
# Convert text into numbers with CountVectorizer
# ============================================================

# Machine learning models cannot use raw text directly.
# CountVectorizer builds a vocabulary and counts how many times each word appears.
# stop_words="english" removes common filler words from the vocabulary.
vectorizer = CountVectorizer(stop_words="english")

# fit_transform() learns the vocabulary from the training reviews
# and converts the training text into a matrix of word counts.
X_train_vectorized = vectorizer.fit_transform(X_train)

# transform() uses the same learned vocabulary on the test reviews.
X_test_vectorized = vectorizer.transform(X_test)

print("\nVectorized training matrix shape:", X_train_vectorized.shape)
print("Vectorized testing matrix shape:", X_test_vectorized.shape)

feature_names = vectorizer.get_feature_names_out()
print("\nA few vocabulary words:")
print(feature_names[:20])


# ============================================================
# Train the Multinomial Naive Bayes model
# ============================================================

# MultinomialNB works well for word-count features,
# which makes it a common baseline model for text classification.
# The default smoothing helps avoid zero-probability issues for unseen words.
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


# ============================================================
# Predict the sentiments of the test reviews
# ============================================================

# The model looks at the word counts in each review
# and predicts whether the review is positive or negative.
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest accuracy:", round(accuracy, 3))
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# ============================================================
# Predict completely new reviews
# ============================================================

# These are new reviews that the model has never seen before.
new_reviews = [
    "The design is excellent and the product worked exactly as described.",
    "I regret buying this because the quality feels cheap and frustrating.",
    "Setup was simple, performance was smooth, and I would buy it again.",
    "The experience was annoying, unreliable, and worse than I expected.",
]

# Transform the new review text into the same word-count format.
new_reviews_vectorized = vectorizer.transform(new_reviews)

# Predict the sentiment label for each new review.
new_predictions = model.predict(new_reviews_vectorized)

print("\nPredictions for new reviews:")
for review_text, prediction in zip(new_reviews, new_predictions):
    print(f"- {prediction:8} | {review_text}")
