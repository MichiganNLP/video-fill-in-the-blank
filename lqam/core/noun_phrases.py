import spacy
import spacy.tokens


def create_spacy_model_for_noun_phrase_check() -> spacy.language.Language:
    return spacy.load("en_core_web_lg")  # We detected fewer errors with it than with "en_core_web_sm".


def is_noun_phrase_like(spacy_doc: spacy.tokens.Doc) -> bool:
    """Checks that there's exactly one sentence, and that it's a Noun Phrase or one without a specifier."""
    sentences_iter = iter(spacy_doc.sents)
    sent = next(sentences_iter, None)
    return sent and not next(sentences_iter, None) and (
            (root := sent.root).pos_ in {"NOUN", "PRON", "PROPN"}
            or root.tag_ in {"VBG", "VBN"}  # VBN, e.g.: "the objects being applied".
            or sent[0].tag_.startswith("W"))  # We also admit phrases that start with a wh-word.
    # E.g., "They explain [how to make a cake]."
