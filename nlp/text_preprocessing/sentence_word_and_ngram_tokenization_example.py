import re


# ============================================================
# Sentence, Word, And N-Gram Tokenization
# ============================================================
#
# Tokenization divides text into units. Different models may consume sentences,
# words, subwords, or characters. This example uses simple regular expressions
# so it runs without downloading NLTK or spaCy language resources.


text = (
    "The camera is not bad. Photos are very sharp! "
    "Would I recommend it? Yes, definitely."
)


# This simple sentence splitter looks for whitespace after ., !, or ?.
# Production tokenizers handle abbreviations such as "Dr." and decimal numbers.
sentences = re.split(r"(?<=[.!?])\s+", text.strip())


def word_tokenize(sentence):
    """Extract lowercase words while keeping apostrophes inside words."""
    return re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", sentence.lower())


def create_ngrams(tokens, n):
    """Return consecutive token groups of size n."""
    # For N tokens, there are N - n + 1 complete n-grams.
    return [
        tuple(tokens[start : start + n])
        for start in range(len(tokens) - n + 1)
    ]


sentence_token_lists = []
print("Sentences:")
for sentence_number, sentence in enumerate(sentences, start=1):
    tokens = word_tokenize(sentence)
    sentence_token_lists.append(tokens)
    print(f"{sentence_number}. {sentence}")
    print("   tokens:", tokens)


# Build n-grams inside each sentence. This avoids artificial pairs such as the
# last word of sentence 1 followed by the first word of sentence 2.
unigrams = [
    ngram
    for tokens in sentence_token_lists
    for ngram in create_ngrams(tokens, n=1)
]
bigrams = [
    ngram
    for tokens in sentence_token_lists
    for ngram in create_ngrams(tokens, n=2)
]
trigrams = [
    ngram
    for tokens in sentence_token_lists
    for ngram in create_ngrams(tokens, n=3)
]

print("\nUnigrams:")
print(unigrams)
print("\nBigrams:")
print(bigrams)
print("\nFirst five trigrams:")
print(trigrams[:5])


# Bigrams preserve local meaning that separate words may lose.
print("\nNegation bigram present:", ("not", "bad") in bigrams)
print(
    "A bag-of-unigrams sees 'not' and 'bad' separately. Adding bigrams lets a"
    " model learn that 'not bad' can differ from the word 'bad' alone."
)
