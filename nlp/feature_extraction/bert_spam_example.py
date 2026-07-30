"""Predict spam with pretrained BERT sentence features."""

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as error:
    raise SystemExit(
        "This example needs transformers and PyTorch. Install them with: "
        "py -3 -m pip install transformers torch"
    ) from error


# ============================================================
# Why use BERT here?
# ============================================================
#
# BERT creates contextual features: the vector for a word depends on the words
# around it. This helps when meaning and word order matter. Unlike BoW, TF-IDF,
# and this tiny Word2Vec model, pretrained BERT already learned language
# patterns from a very large text collection.
#
# BERT is more powerful, but it is also slower and uses more memory. We freeze
# it and train only a small classifier, avoiding a complicated fine-tuning
# example. The first run downloads the pretrained model from Hugging Face.


# ============================================================
# 1. Create a tiny labelled dataset
# ============================================================

messages = [
    "click here to claim your prize",
    "you have won a free voucher",
    "urgent cash reward waiting",
    "limited offer collect now",
    "please review the project report",
    "our team meeting starts at ten",
    "your appointment is confirmed",
    "can we have lunch tomorrow",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]


# ============================================================
# 2. Load a small pretrained BERT model
# ============================================================

# DistilBERT is a smaller and faster BERT variant, suitable for this note.
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.eval()


def create_bert_features(texts):
    """Turn messages into one contextual vector per message."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    # no_grad() saves memory because BERT is used only as a feature extractor.
    with torch.no_grad():
        output = bert_model(**encoded)

    # The first token is BERT's summary token. Its vector represents the
    # complete message and becomes input to the classifier.
    return output.last_hidden_state[:, 0, :].cpu().numpy()


# ============================================================
# 3. Train a classifier on BERT features
# ============================================================

message_features = create_bert_features(messages)
classifier = LogisticRegression(random_state=42)
classifier.fit(message_features, labels)


# ============================================================
# 4. Predict new messages
# ============================================================

new_messages = [
    "claim your free cash reward",
    "the meeting report is attached",
]
new_features = create_bert_features(new_messages)
predictions = classifier.predict(new_features)

print("BERT feature matrix shape:", np.shape(message_features))
print("\nPredictions:")
for message, prediction in zip(new_messages, predictions):
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"{label:8} | {message}")
