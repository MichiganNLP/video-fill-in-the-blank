import spacy.tokens


def create_spacy_model(prefer_gpu: bool = False) -> spacy.language.Language:
    if prefer_gpu:
        spacy.prefer_gpu()
    # This model works pretty well.  Caveat: there's a bug with multiprocessing with this model. So don't use
    # `n_process` with the model's function `pipe`. See https://github.com/explosion/spaCy/issues/6672
    return spacy.load("en_core_web_trf")


def is_noun_phrase_like(span: spacy.tokens.Span) -> bool:
    """FIXME Checks that the text between `start` and `end` form a noun phrase or an N-bar, given the parsed text
    `doc`."""
    # Checking for a phrase in a dependency graph is the same as checking if the nodes (tokens) form a tree.
    # As there are as many edges (heads) as vertices, and because a tree has as many edges as the vertex count minus 1,
    # then all the phrase neighbors should be inside the phrase except for one (the root).
    root = span.root
    return (all(t.head in span or t == root for t in span)
            and (root.pos_ in {"NOUN", "PRON", "PROPN"}
                 or root.tag_ == "VBG"  # Gerund. Example: "*Eating in the morning* is a great way to stay healthy."
                 or (root.tag_ == "VB" and root.i > 0 and root.nbor(-1).tag_ == "TO")  # Infinitive.
                 or span[0].tag_.startswith("W")))  # Wh-word. Example: "They describe *how it works*."
    # Example with infinitive: "*To eat in the morning* is a great way to stay healthy."
