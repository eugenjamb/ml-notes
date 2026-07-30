"""A complete NLTK preprocessing path for one English document.

Install NLTK before running this file:

    python -m pip install nltk

NLTK code and NLTK language datasets are installed separately. The
download_required_resources() function downloads only the datasets used here.
NLTK stores them in the user's NLTK data directory, not in this project.
"""

import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# Each package supplies data used by an NLTK operation below:
# - punkt and punkt_tab: sentence and word tokenization rules.
# - stopwords: lists of common words for different languages.
# - averaged_perceptron_tagger_eng: English part-of-speech tagging model.
# - wordnet: English lexical database used by the lemmatizer.
# - omw-1.4: additional mappings used by WordNet.
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": (
        "taggers/averaged_perceptron_tagger_eng"
    ),
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def download_required_resources():
    """Make this example runnable on a new NLTK installation."""
    for package_name, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            # quiet=True hides progress messages. raise_on_error=True prevents
            # the script from continuing with an incomplete installation.
            nltk.download(
                package_name,
                quiet=True,
                raise_on_error=True,
            )


def to_wordnet_pos(treebank_tag):
    """Convert an NLTK Penn Treebank POS tag to a WordNet POS constant."""
    # WordNetLemmatizer assumes a noun when no POS is supplied. Mapping the
    # first letter lets it correctly reduce verbs and adjectives as well.
    first_letter = treebank_tag[0].upper()
    tag_map = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
        "V": wordnet.VERB,
    }
    return tag_map.get(first_letter, wordnet.NOUN)


download_required_resources()

raw_text = (
    "The striped bats were hanging on their feet and were not flying. "
    "Researchers studied them carefully because the bats' movements were "
    "better at night! Visit https://example.com/bat-study for details."
)


# ============================================================
# 1. Basic normalization
# ============================================================

# Lowercasing prevents "Researchers" and "researchers" from becoming separate
# vocabulary entries. Whether this is appropriate depends on the task because
# capitalization can carry information in named-entity recognition.
normalized_text = raw_text.lower()

# Replace URLs with a stable token. Removing every URL completely would discard
# the useful fact that the document contained a link.
normalized_text = re.sub(
    pattern=r"https?://\S+|www\.\S+",
    repl=" urltoken ",
    string=normalized_text,
)

# Expand a small set of negation contractions before tokenization. Production
# systems normally use a more complete contraction library or language model.
normalized_text = re.sub(r"n't\b", " not", normalized_text)

# Collapse repeated whitespace introduced by cleaning.
normalized_text = re.sub(r"\s+", " ", normalized_text).strip()


# ============================================================
# 2. Sentence and word tokenization
# ============================================================

# sent_tokenize() uses learned punctuation rules, so it handles sentence
# boundaries more reliably than splitting text on every period.
sentences = sent_tokenize(normalized_text, language="english")

# word_tokenize() separates words and punctuation. preserve_line=False tells
# NLTK to perform its normal sentence-aware tokenization.
all_tokens = word_tokenize(
    normalized_text,
    language="english",
    preserve_line=False,
)

# isalpha() removes punctuation and numeric-only tokens. Keep the URL
# placeholder because it consists only of letters.
word_tokens = [token for token in all_tokens if token.isalpha()]


# ============================================================
# 3. Part-of-speech tagging
# ============================================================

# pos_tag() assigns grammatical labels such as NN (noun), VBG (verb), and JJR
# (comparative adjective). POS information improves lemmatization.
tagged_tokens = nltk.pos_tag(word_tokens, lang="eng")


# ============================================================
# 4. Stop-word removal
# ============================================================

# stopwords.words("english") loads NLTK's English stop-word list.
english_stop_words = set(stopwords.words("english"))

# Negation often changes sentiment, so preserve no/not/nor instead of blindly
# removing every default stop word.
english_stop_words -= {"no", "not", "nor"}

# Keep each token's POS tag for the lemmatization stage.
content_tagged_tokens = [
    (token, tag)
    for token, tag in tagged_tokens
    if token not in english_stop_words
]
content_tokens = [token for token, _ in content_tagged_tokens]


# ============================================================
# 5. Stemming
# ============================================================

# PorterStemmer applies suffix-removal rules. It is fast and does not require a
# dictionary, but its results may not be real words: "studies" can become
# "studi". Stemming is useful when rough word grouping is sufficient.
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in content_tokens]


# ============================================================
# 6. POS-aware lemmatization
# ============================================================

# WordNetLemmatizer returns dictionary base forms. Supplying WordNet POS values
# allows "were" -> "be", "flying" -> "fly", and "better" -> "good".
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [
    lemmatizer.lemmatize(token, pos=to_wordnet_pos(tag))
    for token, tag in content_tagged_tokens
]

# Stemming and lemmatization are shown as alternative outputs. Do not normally
# stem a token and then lemmatize that stem because stems may not be real words.


# ============================================================
# 7. Display every preprocessing stage
# ============================================================

print("RAW TEXT")
print(raw_text)
print("\nNORMALIZED TEXT")
print(normalized_text)
print("\nSENTENCES")
for sentence_number, sentence in enumerate(sentences, start=1):
    print(f"{sentence_number}. {sentence}")
print("\nWORD TOKENS")
print(word_tokens)
print("\nPART-OF-SPEECH TAGS")
print(tagged_tokens)
print("\nTOKENS AFTER STOP-WORD REMOVAL")
print(content_tokens)
print("\nSTEMMED TOKENS")
print(stemmed_tokens)
print("\nLEMMATIZED TOKENS")
print(lemmatized_tokens)
print("\nFINAL LEMMATIZED DOCUMENT")
print(" ".join(lemmatized_tokens))
