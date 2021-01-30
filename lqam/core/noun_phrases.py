import spacy.tokens


def create_spacy_model(prefer_gpu: bool = False) -> spacy.language.Language:
    if prefer_gpu:
        spacy.prefer_gpu()
    # This model works pretty well.  Caveat: there's a bug with multiprocessing with this model. So don't use
    # `n_process` with the model's function `pipe`. See https://github.com/explosion/spaCy/issues/6672
    return spacy.load("en_core_web_trf")


def is_phrase(span: spacy.tokens.Span) -> bool:
    """Checks if a text span it's a phrase (a constituent) given its context document."""
    # Checking for a phrase in a dependency graph is the same as checking if the nodes (tokens) form a tree.
    # As there are as many edges (heads) as vertices, and because a tree has as many edges as the vertex count minus 1,
    # then all the phrase neighbors should be inside the phrase except for one (the root).
    #
    # There's an exception: the coordination. They are problematic because in "cat and dog", the heads are:
    # [cat, cat, cat] with labels (root, cc, conj). So "cat and" looks like a constituent but it actually isn't.
    # This is actually fixed in UD v2. See https://universaldependencies.org/v2/coordination.html for more info.
    return all(t.head in span or t == span.root for t in span) and not (span[-1].dep_ == "cc")


def is_noun_phrase_or_n_bar(span: spacy.tokens.Span) -> bool:
    """Checks if a text span forms a noun phrase or an N-bar given its context document."""
    # Just for the record, a better way to do this would be to check that the probability of the span of being an NP
    # passes some threshold, because there are ambiguous cases. The fact that multiple parsing trees are likely for a
    # given sentence means that we don't have to look at the most likely one but rather to only consider the probability
    # of the span of being an NP in the given context.
    root = span.root
    return (is_phrase(span)
            and (root.pos_ in {"NOUN", "PRON", "PROPN"}
                 or (root.tag_ == "VBG" and all(child.dep_ != "aux" for child in root.children))
                 # Example with gerund: "*Eating in the morning* is a great way to stay healthy."
                 or (root.tag_ == "VB" and root.i > 0 and root.nbor(-1).tag_ == "TO")  # Infinitive.
                 # Example with infinitive: "*To eat in the morning* is a great way to stay healthy."
                 or span[0].tag_.startswith("W"))  # Wh-word. Example: "They describe *how it works*."
            and not (len(span) > 2
                     and span[0].lower_ == "order"
                     and span[1].lower_ == "to"
                     and span[0].i > 0
                     and span[0].nbor(-1).lower_ == "in"
                     and span[1] in span[0].subtree))  # Avoid NP-like behavior of an "in order to" usage.
    # The "in order to" needs fixing because it actually acts as a whole preposition but in multiple words.
    # I believe it's pretty exceptional.
