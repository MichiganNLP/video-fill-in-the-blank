import spacy.tokens


def create_spacy_model(prefer_gpu: bool = False) -> spacy.language.Language:
    if prefer_gpu:
        spacy.prefer_gpu()
    # This model works pretty well.  Caveat: there's a bug with multiprocessing with this model. So don't use
    # `n_process` with the model's function `pipe`. See https://github.com/explosion/spaCy/issues/6672
    return spacy.load("en_core_web_trf")


def is_noun_phrase_like(doc: spacy.tokens.Doc, start: int, end: int) -> bool:
    """Checks that the text between `start` and `end` form a noun phrase or an N-bar, given the parsed text `doc`."""
    assert start < end
    # Checking for a phrase in a dependency graph is the same as checking if the nodes (tokens) form a tree.
    # As there are as many edges (heads) as vertices, and because a tree has as many edges as the vertex count minus 1,
    # then all the phrase neighbors should be inside the phrase except for one (the root).
    phrase_token_gen = (t for t in doc if start <= t.idx < end)
    root = next((t for t in phrase_token_gen if t.head.idx < start or end <= t.head.idx or t.head is t), False)
    return root and all(start <= t.head.idx < end for t in phrase_token_gen) and root.pos_ in {"NOUN", "PRON", "PROPN"}
