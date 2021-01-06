from typing import Optional, List
import spacy.tokens
import torch


def compute_label_prob(logits: torch.Tensor, label_ids: torch.Tensor,
                       pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None) -> torch.Tensor:
    """Computes the joint probability of the given labels using the logits.

    :param logits: has shape (N, L, V) and dtype float.
    :param label_ids: has shape (N, L) and dtype int.
    :param pad_token_id: padding token ID, to ignore it. Optional.
    :param eos_token_id: end-of-stream token ID. Optional. Provide it only if you want to ignore it.

    :return: joint probabilities with shape (N,).
    """
    probs = logits.softmax(dim=-1)
    N, L = label_ids.shape
    probs_label_ids = probs[torch.arange(N)[:, None], torch.arange(L)[None], label_ids]
    if pad_token_id is not None:
        probs_label_ids[label_ids == pad_token_id] = 1
    if eos_token_id is not None:
        probs_label_ids[label_ids == eos_token_id] = 1
    # There should be just a few factors, so the product should be numerically stable.
    return probs_label_ids.prod(dim=-1)


def compute_noun_phrase_indices(nlp, generated_answers: List[str], batch_size: int, num_return_sequences: int, device):
    """
    Computes index of the first noun phrase for each num_return_sequences answers if exists
    otherwise return the index of the first answer.
    
    :param nlp: spacy model
    :param generated_answers: has shape (batch_size * num_return_sequences, )
    
    :return: noun phrase indices with shape (batch_size, )
    """
    generated_docs = nlp.pipe(generated_answers)
    noun_chunks_mask = torch.zeros(batch_size * num_return_sequences, dtype=torch.bool, device=device)
    # each instance, if all sequences are not noun phrases,
    # then we mark the first sequence as the only answer
    for batch_idx in range(batch_size):
        start_index = batch_idx * num_return_sequences
        # mark the first answer as the chosen one
        noun_chunks_mask[start_index] = True
        for seq_idx in range(num_return_sequences):
            curr_index = start_index + seq_idx
            if is_noun_phrase_like(next(generated_docs)):
                noun_chunks_mask[start_index] = False
                noun_chunks_mask[curr_index] = True
                break
    return noun_chunks_mask.nonzero(as_tuple=False).squeeze()


def is_noun_phrase_like(spacy_doc: spacy.tokens.Doc) -> bool:
    """Checks that there's exactly one sentence, and that it's a Noun Phrase or one without a specifier."""
    sentences_iter = iter(spacy_doc.sents)
    sent = next(sentences_iter, None)
    return sent and not next(sentences_iter, None) and (
            (root := sent.root).pos_ in {"NOUN", "PRON", "PROPN"}
            or root.tag_ in {"VBG", "VBN"}  # VBN, e.g.: "the objects being applied".
            or sent[0].tag_.startswith("W"))  # We also admit phrases that start with a wh-word.
    # E.g., "They explain [how to make a cake]."
