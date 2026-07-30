from collections import Counter

import nltk
from nltk import RegexpParser
from nltk.tokenize import sent_tokenize, word_tokenize


# punkt and punkt_tab provide sentence and word tokenization rules.
# averaged_perceptron_tagger_eng provides the English POS-tagging model.
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger_eng": (
        "taggers/averaged_perceptron_tagger_eng"
    ),
}


def download_required_resources():
    """Download NLTK datasets only when they are not already available."""
    for package_name, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            # Download only missing resources and suppress progress messages.
            nltk.download(
                package_name,
                quiet=True,
                raise_on_error=True,
            )


download_required_resources()


# This small corpus replaces the course-specific pos_tagged_oz variable. Using
# local text makes the example runnable without additional Python modules.
text = (
    "The developer quickly tested the new model. "
    "The analyst will carefully review the results. "
    "Our system is learning rapidly. "
    "The engineer did not give up the difficult task. "
    "A small robot moved very slowly."
)


# ============================================================
# 1. Tokenize and part-of-speech tag every sentence
# ============================================================

sentences = sent_tokenize(text, language="english")
pos_tagged_sentences = []

for sentence in sentences:
    # word_tokenize() converts one sentence into words and punctuation.
    tokens = word_tokenize(
        sentence,
        language="english",
        preserve_line=True,
    )

    # pos_tag() adds Penn Treebank grammatical tags to every token.
    # Examples: VB=verb, VBD=past-tense verb, RB=adverb, and NN=noun.
    tagged_sentence = nltk.pos_tag(tokens, lang="eng")
    pos_tagged_sentences.append(tagged_sentence)


# ============================================================
# 2. Define the verb-phrase chunk grammar
# ============================================================

# RegexpParser does not match text characters. It matches sequences of POS tags:
#
# <MD>?       zero or one modal verb: will, can, might
# <VB.*>?     zero or one auxiliary verb before an adverb: did in "did not give"
# <RB.*>*     zero or more adverbs: carefully, not, quickly
# <VB.*>+     one or more main/linked verbs: test, tested, is learning
# <RB.*>*     zero or more adverbs after the verb: very slowly
# <RP>?       zero or one particle used by a phrasal verb: up in "give up"
# <DT>?       zero or one determiner: the, a
# <JJ.*>*     zero or more adjectives: new, difficult
# <NN.*>*     zero or more nouns: model, task, results
#
# "." means "any characters", "*" means "zero or more", "+" means "one or
# more", and "?" means "zero or one".
chunk_grammar = r"""
    VP: {<MD>?<VB.*>?<RB.*>*<VB.*>+<RB.*>*<RP>?<DT>?<JJ.*>*<NN.*>*}
"""


# loop=1 applies the grammar once to each sentence. More loops can be useful
# when later rules depend on chunks produced by earlier rules.
# trace=0 prevents RegexpParser from logging every internal matching step.
chunk_parser = RegexpParser(
    chunk_grammar,
    loop=1,
    trace=0,
)


# ============================================================
# 3. Parse every POS-tagged sentence
# ============================================================

vp_chunked_sentences = []

for tagged_sentence in pos_tagged_sentences:
    # parse() returns an nltk.Tree. Matching token groups become VP subtrees;
    # tokens not captured by the grammar remain outside those subtrees.
    chunked_sentence = chunk_parser.parse(tagged_sentence)
    vp_chunked_sentences.append(chunked_sentence)


# ============================================================
# 4. Extract and count VP chunks
# ============================================================

def verb_phrase_counter(chunked_sentences):
    """Return verb phrases ordered from most to least frequent."""
    verb_phrases = []

    for chunked_sentence in chunked_sentences:
        # subtrees() walks through every nested subtree in the parse tree.
        for subtree in chunked_sentence.subtrees():
            if subtree.label() == "VP":
                # leaves() returns (word, POS-tag) pairs from the VP subtree.
                phrase = " ".join(word for word, _ in subtree.leaves())
                verb_phrases.append(phrase)

    # Counter counts duplicate phrases. most_common() sorts by frequency.
    return Counter(verb_phrases).most_common()


most_common_vp_chunks = verb_phrase_counter(vp_chunked_sentences)


# ============================================================
# 5. Display tags, chunk trees, and extracted phrases
# ============================================================

for sentence_number, (sentence, tagged, chunked) in enumerate(
    zip(sentences, pos_tagged_sentences, vp_chunked_sentences),
    start=1,
):
    print(f"\nSENTENCE {sentence_number}")
    print("Text:", sentence)
    print("POS tags:", tagged)
    print("Chunk tree:", chunked)

print("\nVERB PHRASES AND COUNTS")
for phrase, count in most_common_vp_chunks:
    print(f"{phrase!r}: {count}")


# For an interactive graphical tree, call chunked.draw(). It is intentionally
# not used here because it opens a separate Tkinter window.
