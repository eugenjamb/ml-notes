"""A small encoder-decoder model that translates English into Spanish."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras


# ============================================================
# Why use a seq2seq model for translation?
# ============================================================
#
# Translation changes one sequence of words into another sequence whose length
# and word order may differ. A sequence-to-sequence (seq2seq) model has:
# - an encoder that reads the complete English sentence;
# - a decoder that generates the Spanish sentence one word at a time.
#
# This deliberately tiny dataset teaches the mechanics. It is able to memorise
# these examples, but a useful translator needs a large bilingual dataset.


keras.utils.set_random_seed(42)


# ============================================================
# 1. Create a tiny parallel-text dataset
# ============================================================

# Each tuple contains the same message in English and Spanish.
sentence_pairs = [
    ("hello", "hola"),
    ("good morning", "buenos dias"),
    ("good night", "buenas noches"),
    ("thank you", "gracias"),
    ("see you soon", "hasta pronto"),
    ("i am happy", "estoy feliz"),
    ("i am tired", "estoy cansado"),
    ("i like music", "me gusta musica"),
    ("we like coffee", "nos gusta cafe"),
    ("she reads books", "ella lee libros"),
    ("he drinks water", "el bebe agua"),
    ("where is home", "donde esta casa"),
]

# Special decoder tokens tell the model when generation starts and ends.
START_TOKEN = "<start>"
END_TOKEN = "<end>"


def build_vocabulary(sentences, special_tokens=()):
    """Give every distinct word a positive integer ID; zero is padding."""
    words = sorted({word for sentence in sentences for word in sentence.split()})
    id_to_word = ["<pad>", *special_tokens, *words]
    word_to_id = {word: index for index, word in enumerate(id_to_word)}
    return word_to_id, id_to_word


english_sentences = [english for english, _ in sentence_pairs]
spanish_sentences = [spanish for _, spanish in sentence_pairs]

english_to_id, english_id_to_word = build_vocabulary(english_sentences)
spanish_to_id, spanish_id_to_word = build_vocabulary(
    spanish_sentences,
    special_tokens=(START_TOKEN, END_TOKEN),
)


def encode(sentence, vocabulary):
    """Replace the words in one sentence with their vocabulary IDs."""
    return [vocabulary[word] for word in sentence.lower().split()]


# pad_sequences makes every batch row the same length by appending zeroes.
encoder_input = keras.utils.pad_sequences(
    [encode(sentence, english_to_id) for sentence in english_sentences],
    padding="post",
)

# Teacher forcing supplies the correct previous Spanish word to the decoder.
# Input:  <start> buenos dias
# Target: buenos dias <end>
decoder_input = keras.utils.pad_sequences(
    [
        encode(f"{START_TOKEN} {sentence}", spanish_to_id)
        for sentence in spanish_sentences
    ],
    padding="post",
)
decoder_target = keras.utils.pad_sequences(
    [
        encode(f"{sentence} {END_TOKEN}", spanish_to_id)
        for sentence in spanish_sentences
    ],
    padding="post",
)

# Padding is not a real target word, so give padded positions zero loss weight.
target_weights = (decoder_target != 0).astype("float32")


# ============================================================
# 2. Build the encoder
# ============================================================

EMBEDDING_SIZE = 32
STATE_SIZE = 64

encoder_words = keras.Input(shape=(None,), name="english_words")

# Embedding learns a compact vector for each English word.
encoder_embedding = keras.layers.Embedding(
    input_dim=len(english_id_to_word),
    output_dim=EMBEDDING_SIZE,
    mask_zero=True,
    name="english_embedding",
)(encoder_words)

# Only the final hidden and cell states are needed. Together they form the
# context passed from the encoder to the decoder.
_, encoder_hidden, encoder_cell = keras.layers.LSTM(
    STATE_SIZE,
    return_state=True,
    name="encoder_lstm",
)(encoder_embedding)


# ============================================================
# 3. Build the decoder
# ============================================================

decoder_words = keras.Input(shape=(None,), name="previous_spanish_words")
decoder_embedding = keras.layers.Embedding(
    input_dim=len(spanish_id_to_word),
    output_dim=EMBEDDING_SIZE,
    mask_zero=True,
    name="spanish_embedding",
)(decoder_words)

# return_sequences=True produces an output for every decoder timestep.
decoder_sequence = keras.layers.LSTM(
    STATE_SIZE,
    return_sequences=True,
    name="decoder_lstm",
)(decoder_embedding, initial_state=[encoder_hidden, encoder_cell])

# Softmax creates a probability for every possible next Spanish word.
word_probabilities = keras.layers.Dense(
    len(spanish_id_to_word),
    activation="softmax",
    name="next_word",
)(decoder_sequence)

model = keras.Model(
    inputs=[encoder_words, decoder_words],
    outputs=word_probabilities,
    name="tiny_english_to_spanish_seq2seq",
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# 4. Train the complete seq2seq model
# ============================================================

print("Training the tiny translation model...")
model.fit(
    [encoder_input, decoder_input],
    decoder_target,
    sample_weight=target_weights,
    epochs=300,
    batch_size=len(sentence_pairs),
    verbose=0,
)


# ============================================================
# 5. Generate a translation one word at a time
# ============================================================

def translate(english_sentence):
    """Autoregressively decode one sentence from English into Spanish."""
    unknown_words = [
        word
        for word in english_sentence.lower().split()
        if word not in english_to_id
    ]
    if unknown_words:
        return f"Cannot translate unknown word(s): {', '.join(unknown_words)}"

    encoded_english = keras.utils.pad_sequences(
        [encode(english_sentence, english_to_id)],
        maxlen=encoder_input.shape[1],
        padding="post",
    )
    generated_ids = [spanish_to_id[START_TOKEN]]

    # At every step, feed the words generated so far back into the decoder.
    for _ in range(decoder_target.shape[1]):
        partial_decoder_input = keras.utils.pad_sequences(
            [generated_ids],
            maxlen=decoder_input.shape[1],
            padding="post",
        )
        probabilities = model.predict(
            [encoded_english, partial_decoder_input],
            verbose=0,
        )
        current_position = len(generated_ids) - 1
        next_id = int(np.argmax(probabilities[0, current_position]))

        if next_id in (0, spanish_to_id[END_TOKEN]):
            break
        generated_ids.append(next_id)

    return " ".join(spanish_id_to_word[word_id] for word_id in generated_ids[1:])


print("\nTranslations:")
for example in ["good morning", "i like music", "he drinks water"]:
    print(f"{example:16} -> {translate(example)}")

