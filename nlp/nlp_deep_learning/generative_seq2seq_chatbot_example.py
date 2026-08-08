"""A tiny generative chatbot built with an encoder-decoder neural network."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras


# ============================================================
# What makes this chatbot generative?
# ============================================================
#
# A rule-based chatbot selects a response written in an if statement. This
# model instead learns prompt-response patterns and generates its response one
# word at a time. Its architecture is a sequence-to-sequence (seq2seq) model:
#
# user message -> encoder LSTM -> context states -> decoder LSTM -> response
#
# This tiny dataset is suitable for learning the workflow, not for creating a
# production chatbot. A useful chatbot requires far more varied conversations.


keras.utils.set_random_seed(42)


# ============================================================
# 1. Create a tiny conversation dataset
# ============================================================

# Each tuple contains a user message and the response the model should learn.
conversation_pairs = [
    ("hello", "hello how can i help"),
    ("hi", "hello how can i help"),
    ("good morning", "good morning how can i help"),
    ("how are you", "i am doing well"),
    ("what is your name", "i am note bot"),
    ("who are you", "i am note bot"),
    ("what can you do", "i can discuss coding"),
    ("can you help me", "yes i can help"),
    ("i like python", "python is great for coding"),
    ("tell me about python", "python is great for coding"),
    ("i am learning ai", "ai is fun to learn"),
    ("is ai interesting", "ai is fun to learn"),
    ("thank you", "you are welcome"),
    ("thanks", "you are welcome"),
    ("goodbye", "goodbye and keep learning"),
    ("see you", "goodbye and keep learning"),
]

START_TOKEN = "<start>"
END_TOKEN = "<end>"


def build_vocabulary(sentences, special_tokens=()):
    """Create mappings between words and integer token IDs."""
    words = sorted({word for sentence in sentences for word in sentence.split()})
    id_to_word = ["<pad>", *special_tokens, *words]
    word_to_id = {word: index for index, word in enumerate(id_to_word)}
    return word_to_id, id_to_word


user_messages = [message for message, _ in conversation_pairs]
bot_responses = [response for _, response in conversation_pairs]

input_word_to_id, input_id_to_word = build_vocabulary(user_messages)
output_word_to_id, output_id_to_word = build_vocabulary(
    bot_responses,
    special_tokens=(START_TOKEN, END_TOKEN),
)


def encode(sentence, vocabulary):
    """Convert a space-separated sentence into integer token IDs."""
    return [vocabulary[word] for word in sentence.lower().split()]


# ============================================================
# 2. Prepare encoder and decoder sequences
# ============================================================

# Padding adds zeroes so every sequence in a batch has the same length.
encoder_input = keras.utils.pad_sequences(
    [encode(message, input_word_to_id) for message in user_messages],
    padding="post",
)

# Teacher forcing shifts the decoder sequences by one position:
# decoder input:  <start> i am note bot
# decoder target: i am note bot <end>
decoder_input = keras.utils.pad_sequences(
    [
        encode(f"{START_TOKEN} {response}", output_word_to_id)
        for response in bot_responses
    ],
    padding="post",
)
decoder_target = keras.utils.pad_sequences(
    [
        encode(f"{response} {END_TOKEN}", output_word_to_id)
        for response in bot_responses
    ],
    padding="post",
)

# Padded positions contain no real target, so exclude them from the loss.
target_weights = (decoder_target != 0).astype("float32")


# ============================================================
# 3. Build the encoder
# ============================================================

EMBEDDING_SIZE = 32
STATE_SIZE = 64

encoder_words = keras.Input(shape=(None,), name="user_message")

# This embedding layer learns one vector for each user-vocabulary word.
encoder_vectors = keras.layers.Embedding(
    input_dim=len(input_id_to_word),
    output_dim=EMBEDDING_SIZE,
    mask_zero=True,
    name="input_word_embeddings",
)(encoder_words)

# The encoder reads the user message. Its final hidden and cell states provide
# a compact context representation for the response decoder.
_, encoder_hidden, encoder_cell = keras.layers.LSTM(
    STATE_SIZE,
    return_state=True,
    name="encoder_lstm",
)(encoder_vectors)


# ============================================================
# 4. Build the response decoder
# ============================================================

decoder_words = keras.Input(shape=(None,), name="previous_response_words")
decoder_vectors = keras.layers.Embedding(
    input_dim=len(output_id_to_word),
    output_dim=EMBEDDING_SIZE,
    mask_zero=True,
    name="response_word_embeddings",
)(decoder_words)

# The decoder begins with the encoder states, connecting the generated reply
# to the meaning learned from the user's message.
decoder_sequence = keras.layers.LSTM(
    STATE_SIZE,
    return_sequences=True,
    name="decoder_lstm",
)(decoder_vectors, initial_state=[encoder_hidden, encoder_cell])

# Softmax returns a probability for every possible next response word.
next_word_probabilities = keras.layers.Dense(
    len(output_id_to_word),
    activation="softmax",
    name="next_response_word",
)(decoder_sequence)

chatbot = keras.Model(
    inputs=[encoder_words, decoder_words],
    outputs=next_word_probabilities,
    name="tiny_generative_chatbot",
)
chatbot.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# 5. Train all embedding, LSTM, and output weights
# ============================================================

print("Training the tiny generative chatbot...")
history = chatbot.fit(
    [encoder_input, decoder_input],
    decoder_target,
    sample_weight=target_weights,
    epochs=350,
    batch_size=len(conversation_pairs),
    verbose=0,
)
print("Final training accuracy:", round(history.history["accuracy"][-1], 3))


# ============================================================
# 6. Generate a response autoregressively
# ============================================================

def generate_response(message):
    """Generate response words until the model predicts the end token."""
    message_words = message.lower().split()
    unknown_words = [word for word in message_words if word not in input_word_to_id]

    # This basic example has no unknown-word token. A production tokenizer
    # would use subwords so it could safely process unfamiliar vocabulary.
    if unknown_words:
        return f"I do not know the word(s): {', '.join(unknown_words)}"

    encoded_message = keras.utils.pad_sequences(
        [encode(message, input_word_to_id)],
        maxlen=encoder_input.shape[1],
        padding="post",
    )
    generated_ids = [output_word_to_id[START_TOKEN]]

    # Feed each generated word back into the model to obtain the next word.
    for _ in range(decoder_target.shape[1]):
        partial_response = keras.utils.pad_sequences(
            [generated_ids],
            maxlen=decoder_input.shape[1],
            padding="post",
        )
        probabilities = chatbot.predict(
            [encoded_message, partial_response],
            verbose=0,
        )
        current_position = len(generated_ids) - 1
        next_id = int(np.argmax(probabilities[0, current_position]))

        if next_id in (0, output_word_to_id[END_TOKEN]):
            break
        generated_ids.append(next_id)

    return " ".join(output_id_to_word[token_id] for token_id in generated_ids[1:])


# ============================================================
# 7. Test the trained chatbot
# ============================================================

print("\nChatbot examples:")
for user_message in ["hello", "what is your name", "i am learning ai", "goodbye"]:
    print(f"You: {user_message}")
    print(f"Bot: {generate_response(user_message)}\n")

