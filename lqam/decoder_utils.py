from typing import Optional

import torch


def compute_label_prob(logits: torch.Tensor, label_ids: torch.Tensor,
                       pad_token_id: Optional[int] = None) -> torch.Tensor:
    """Computes the joint probability of the given labels using the logits.

    :param logits: has shape (N, L, V) and dtype float.
    :param label_ids: has shape (N, L) and dtype int.
    :param pad_token_id: padding token ID. Optional.

    :return: joint probabilities with shape (N,).
    """
    probs = logits.softmax(dim=-1)
    N, L = label_ids.shape
    probs_label_ids = probs[torch.arange(N)[:, None], torch.arange(L)[None], label_ids]
    if pad_token_id is not None:
        probs_label_ids[label_ids == pad_token_id] = 1
    # There should be just a few factors, so the product should be numerically stable.
    return probs_label_ids.prod(dim=-1)
