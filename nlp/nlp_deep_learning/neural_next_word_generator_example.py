"""Train a tiny embedding-based neural network to generate text."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras


# ============================================================
# Is this an LLM?
# ============================================================
#
# This uses the same basic learning idea as a neural language model:
#   word IDs -> learned embeddings -> recurrent neural network -> next word
#
# However, it is a tiny language model, not a Large Language Model (LLM).
# Modern LLMs contain vastly more data and parameters and normally use
# Transformer layers. This small GRU model makes the core idea easy to inspect.


keras.utils.set_random_seed(42)


# ============================================================
# 1. Prepare a small training story
# ============================================================

# Repeated sentence patterns give the small model enough examples to learn
# relationships such as "robot learns" and "writes helpful code".
training_text = """
the small robot learns from code
the small robot learns from words
the clever robot writes helpful code
the clever robot writes clear notes
the ai developer learns from examples
the ai developer writes helpful notes
good examples make learning clear
clear notes make coding easier
helpful code makes developers happy
the robot reads code and writes notes
the developer reads notes and writes code
"""

tokens = training_text.lower().split()

# Zero is reserved for padding. The remaining IDs represent real words.
vocabulary = sorted(set(tokens))
word_to_id = {word: index + 1 for index, word in enumerate(vocabulary)}
id_to_word = {index: word for word, index in word_to_id.items()}


# ============================================================
# 2. Create next-word training examples
# ============================================================

CONTEXT_SIZE = 3
contexts = []
targets = []

# A sliding window turns "the small robot learns" into:
# context = [the, small, robot], target = learns
for position in range(len(tokens) - CONTEXT_SIZE):
    context_words = tokens[position : position + CONTEXT_SIZE]
    target_word = tokens[position + CONTEXT_SIZE]
    contexts.append([word_to_id[word] for word in context_words])
    targets.append(word_to_id[target_word])

contexts = np.asarray(contexts, dtype="int32")
targets = np.asarray(targets, dtype="int32")

print("Vocabulary size:", len(vocabulary))
print("Training examples:", len(contexts))


# ============================================================
# 3. Build the neural language model
# ============================================================

model = keras.Sequential(
    [
        keras.layers.Input(shape=(CONTEXT_SIZE,)),

        # Embedding starts with random values. Training adjusts these weights
        # so words useful in similar contexts develop useful representations.
        keras.layers.Embedding(
            input_dim=len(vocabulary) + 1,
            output_dim=16,
            name="learned_word_embeddings",
        ),

        # A GRU reads the ordered embedding sequence and learns context. It is
        # a simpler recurrent relative of the LSTM used in the seq2seq example.
        keras.layers.GRU(32, name="context_gru"),

        # One output probability is produced for each vocabulary word.
        keras.layers.Dense(
            len(vocabulary) + 1,
            activation="softmax",
            name="next_word_probabilities",
        ),
    ],
    name="tiny_neural_language_model",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# 4. Train the word embeddings and neural-network weights
# ============================================================

print("Training the tiny language model...")
history = model.fit(
    contexts,
    targets,
    epochs=250,
    batch_size=16,
    verbose=0,
)
print("Final training accuracy:", round(history.history["accuracy"][-1], 3))

# These are the learned word-vector weights. Each vocabulary word now has a
# vector of 16 numbers that changed while the model learned next-word patterns.
embedding_weights = model.get_layer("learned_word_embeddings").get_weights()[0]
print("Learned embedding matrix shape:", embedding_weights.shape)


# ============================================================
# 5. Generate text autoregressively
# ============================================================

def generate_text(seed_text, words_to_generate=8):
    """Predict a word, append it, and use it to predict the next word."""
    generated_words = seed_text.lower().split()

    if len(generated_words) < CONTEXT_SIZE:
        raise ValueError(f"Seed text needs at least {CONTEXT_SIZE} words.")
    if any(word not in word_to_id for word in generated_words):
        raise ValueError("Every seed word must occur in the training story.")

    for _ in range(words_to_generate):
        recent_words = generated_words[-CONTEXT_SIZE:]
        context_ids = np.asarray(
            [[word_to_id[word] for word in recent_words]],
            dtype="int32",
        )
        probabilities = model.predict(context_ids, verbose=0)[0]

        # Greedy decoding chooses the most probable next word. Temperature or
        # random sampling could produce more varied, but less stable, results.
        probabilities[0] = 0  # Padding must never be generated.
        next_word_id = int(np.argmax(probabilities))
        generated_words.append(id_to_word[next_word_id])

    return " ".join(generated_words)


print("\nGenerated text:")
print(generate_text("the clever robot", words_to_generate=8))

