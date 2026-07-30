from itertools import chain

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import bigrams


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
    """Download the small NLTK datasets required by this example."""
    for package_name, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(
                package_name,
                quiet=True,
                raise_on_error=True,
            )


def to_wordnet_pos(treebank_tag):
    """Map a Penn Treebank POS tag to the tag format WordNet expects."""
    tag_map = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
        "V": wordnet.VERB,
    }
    return tag_map.get(treebank_tag[0].upper(), wordnet.NOUN)


download_required_resources()

# A corpus is a collection of documents. Each string could represent a review,
# article, support message, or another independent piece of text.
documents = [
    "The banks approved loans for growing businesses.",
    "A fisherman sat on the river bank and watched the water.",
    "Customers were not enjoying the slower banking application.",
    "The improved application runs quickly and gives better predictions.",
]

english_stop_words = set(stopwords.words("english")) - {"no", "not", "nor"}
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_document(document):
    """Return tokens, stems, and POS-aware lemmas for one document."""
    # word_tokenize splits punctuation from words. lower() unifies case before
    # vocabulary creation.
    raw_tokens = word_tokenize(
        document.lower(),
        language="english",
        preserve_line=False,
    )

    # Keep alphabetic words, then tag them before stop-word removal. Tagging the
    # complete sentence gives the POS model more grammatical context.
    word_tokens = [token for token in raw_tokens if token.isalpha()]
    tagged_tokens = nltk.pos_tag(word_tokens, lang="eng")

    content_tagged_tokens = [
        (token, tag)
        for token, tag in tagged_tokens
        if token not in english_stop_words
    ]

    # Stems are rough rule-based forms; lemmas are dictionary base forms.
    stems = [
        stemmer.stem(token)
        for token, _ in content_tagged_tokens
    ]
    lemmas = [
        lemmatizer.lemmatize(token, pos=to_wordnet_pos(tag))
        for token, tag in content_tagged_tokens
    ]

    return {
        "tokens": [token for token, _ in content_tagged_tokens],
        "stems": stems,
        "lemmas": lemmas,
    }


processed_documents = [
    preprocess_document(document)
    for document in documents
]

for document_number, (original, processed) in enumerate(
    zip(documents, processed_documents),
    start=1,
):
    print(f"\nDOCUMENT {document_number}")
    print("Original:", original)
    print("Content tokens:", processed["tokens"])
    print("Stems:", processed["stems"])
    print("Lemmas:", processed["lemmas"])


# ============================================================
# Corpus vocabulary and frequency distribution
# ============================================================

# chain.from_iterable() flattens all per-document lemma lists into one corpus.
corpus_lemmas = list(
    chain.from_iterable(
        document["lemmas"]
        for document in processed_documents
    )
)

# FreqDist counts token occurrences. most_common(10) returns no more than the
# ten most frequent (token, count) pairs.
lemma_frequencies = FreqDist(corpus_lemmas)

print("\nCORPUS LEMMAS")
print(corpus_lemmas)
print("\nTEN MOST COMMON LEMMAS")
print(lemma_frequencies.most_common(10))

# bigrams() creates adjacent two-token sequences. N-grams preserve some local
# context that a plain bag-of-words representation loses.
corpus_bigrams = list(bigrams(corpus_lemmas))
print("\nFIRST TEN CORPUS BIGRAMS")
print(corpus_bigrams[:10])


# ============================================================
# Explore corpus vocabulary with the WordNet corpus
# ============================================================

def describe_word_with_wordnet(word, maximum_senses=3):
    """Print WordNet senses, definitions, examples, and synonyms."""
    # wordnet.synsets() returns groups of synonymous words. A word may have
    # several synsets because its meaning depends on context.
    synsets = wordnet.synsets(word)

    print(f"\nWORDNET RESULTS FOR '{word}'")
    if not synsets:
        print("No WordNet entry was found.")
        return

    # [:maximum_senses] limits output; WordNet can contain many meanings.
    for sense_number, synset in enumerate(
        synsets[:maximum_senses],
        start=1,
    ):
        # lemma_names() contains synonyms associated with this particular
        # meaning. Replace underscores to make multi-word terms readable.
        synonyms = sorted(
            {
                lemma.name().replace("_", " ")
                for lemma in synset.lemmas()
            }
        )

        # Antonyms belong to individual WordNet lemmas rather than the synset.
        antonyms = sorted(
            {
                antonym.name().replace("_", " ")
                for lemma in synset.lemmas()
                for antonym in lemma.antonyms()
            }
        )

        print(f"Sense {sense_number}: {synset.name()}")
        print("  Definition:", synset.definition())
        print("  Examples:", synset.examples() or "No example available")
        print("  Synonyms:", synonyms)
        print("  Antonyms:", antonyms or "No antonyms listed")


# "bank" demonstrates lexical ambiguity: WordNet contains financial and river
# meanings. Choosing synsets[0] automatically would not guarantee the correct
# meaning for every sentence; word-sense disambiguation requires context.
describe_word_with_wordnet("bank", maximum_senses=3)
describe_word_with_wordnet("good", maximum_senses=2)
