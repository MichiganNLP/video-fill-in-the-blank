from typing import Optional, List
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
    """Computes the boolean mask for returned sequences for each question.
    
    :param nlp: Language object from spacy
    :param generated_answers: has shape (batch_size * num_return_sequences) and type str
    :param batch_size: type int
    :param num_return_sequences: type int
    :param device: device of tensor
    
    :return: noun phrase indices with shape (batch_size, )
    """
    noun_chunks_mask = torch.zeros(batch_size * num_return_sequences, dtype=torch.bool, device=device)
    # each instance, if all sequences are not noun phrases,
    # then we mark the first sequence as the only answer
    for batch_idx in range(batch_size):
        start_index = batch_idx * num_return_sequences
        # mark the first answer as the chosen one
        noun_chunks_mask[start_index] = True
        for seq_idx in range(num_return_sequences):
            curr_index = start_index + seq_idx
            if generated_answers[curr_index] in set(nlp(generated_answers[curr_index]).noun_chunks):
                noun_chunks_mask[start_index] = False
                noun_chunks_mask[curr_index] = True
                break
    return noun_chunks_mask.nonzero(as_tuple=False).squeeze()
