#!/usr/bin/env python
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterator, List, Mapping, Sequence, Tuple, Union

import pandas as pd
import torch
from transformers import Pipeline, PreTrainedTokenizerBase, pipeline

from lqam import iterable_utils
from lqam.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.file_utils import cached_path
from lqam.metric_utils import exact_match

RE_EXTRA_ID = re.compile(r"<extra_id_\d+>")

OUTPUT_PATH = Path("output")


def _compute_mask_values(generated_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> Mapping[str, List[str]]:
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    extra_id_indices = {token: i for i, token in enumerate(tokens) if RE_EXTRA_ID.match(token)}
    extra_id_indices["</s>"] = len(tokens)

    return {extra_id_token: tokens[extra_id_indices[extra_id_token] + 1:extra_id_indices[next_extra_id_token]]
            for extra_id_token, next_extra_id_token in iterable_utils.pairwise(extra_id_indices)}


def _clean_mask_values(input_tokens: Sequence[str], mask_values: Mapping[str, List[str]]) -> Mapping[str, List[str]]:
    return {k: v for k, v in mask_values.items() if k in set(input_tokens)}


def _substitute_mask_values(input_text: str, generated_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) \
        -> Tuple[str, str]:
    mask_values = _compute_mask_values(generated_ids, tokenizer)

    new_tokens = [new_token
                  for token in tokenizer.tokenize(input_text)
                  for new_token in mask_values.get(token, [token])]

    # FIXME: It works only when having a single blank.
    return tokenizer.convert_tokens_to_string(new_tokens), \
        tokenizer.convert_tokens_to_string(mask_values["<extra_id_0>"])


def _fill_in_the_blanks(blanked_captions: Union[str, Sequence[str]], gen_pipeline: Pipeline,
                        batch_size: int = sys.maxsize, beam_size: int = 1) -> Iterator[Tuple[str, str]]:
    # The generator checks explicitly for a list, so we need one.
    blanked_captions = [blanked_captions] if isinstance(blanked_captions, str) else list(blanked_captions)

    with torch.no_grad():
        outputs = (output
                   for batch in iterable_utils.chunks(blanked_captions, batch_size)
                   for output in gen_pipeline(batch, return_tensors=True, return_text=False, num_beams=beam_size))
        for blanked_caption, output in zip(blanked_captions, outputs):
            yield _substitute_mask_values(blanked_caption, output["generated_token_ids"], gen_pipeline.tokenizer)


def _compute_label_prob(logits: Sequence[torch.Tensor], label: Sequence[torch.Tensor]) -> torch.Tensor:
    N = len(logits)
    results = torch.zeros(N, device=logits[0].device, dtype=logits[0].dtype)
    for i in range(N):
        probs = logits[i].softmax(dim=-1)
        results[i] = probs[torch.arange(len(label[i])), label[i]].prod()
    return results  # There should be just a few factors, so the product should be numerically stable.


def _compute_pred(input_ids: List[torch.Tensor], gen_pipeline: Pipeline,
                  beam_size: int) -> Tuple[Sequence[str], Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    # `generate` does it, however it doesn't return the logits.
    # So we do one more forward to get the logits.

    generated_ids = gen_pipeline.model.generate(input_ids, num_beams=beam_size, no_repeat_ngram_size=2,
                                                early_stopping=True)

    for i, beam_output in enumerate(generated_ids):
        print(f"{i}: {gen_pipeline.tokenizer.decode(beam_output)}")

    pred_list, pred_logits_list, pred_labels_list = [], [], []

    for ids in generated_ids:
        pred_tokens = _compute_mask_values(ids, gen_pipeline.tokenizer)["<extra_id_0>"]

        print(pred_tokens)

        pred = gen_pipeline.tokenizer.convert_tokens_to_string(pred_tokens)

        # `pred_tokens` could have some extra gibberish, so we clean it.
        pred_label_tokens = ["<extra_id_0>"] + pred_tokens + ["<extra_id_1>"]

        pred_label = torch.tensor(gen_pipeline.tokenizer.convert_tokens_to_ids(pred_label_tokens),
                                  device=gen_pipeline.device)

        pred_logits = gen_pipeline.model(ids.unsqueeze(0), labels=pred_label.unsqueeze(0))[1].squeeze(0)

        pred_list.append(pred)
        pred_logits_list.append(pred_logits)
        pred_labels_list.append(pred_label)

    return pred_list, pred_logits_list, pred_labels_list


def _compute_prob_and_pred(texts: List[str], gen_pipeline: Pipeline,
                           ground_truth: List[str], beam_size: int) -> Tuple[torch.Tensor, torch.Tensor, Sequence[str]]:
    input_ids = gen_pipeline.tokenizer(texts, add_special_tokens=True, padding=True,
                                       return_tensors=gen_pipeline.framework).input_ids.to(gen_pipeline.device)
    ground_truth_labels = gen_pipeline.tokenizer(
        ground_truth, add_special_tokens=False, padding=True,
        return_tensors=gen_pipeline.framework).input_ids.to(gen_pipeline.device)

    with torch.no_grad():
        ground_truth_logits = gen_pipeline.model(input_ids, labels=ground_truth_labels)[1]
        pred, pred_logits, pred_label = _compute_pred(input_ids, gen_pipeline, beam_size=beam_size)
        ground_truth_prob = _compute_label_prob(ground_truth_logits, ground_truth_labels)
        pred_prob = _compute_label_prob(pred_logits, pred_label)

    return ground_truth_prob, pred_prob, pred


def _evaluate_exact_match(df: pd.DataFrame, gen_pipeline: Pipeline) -> None:
    """Evaluate the text-only baseline on the training set using a greedy approach."""
    predictions = 0
    pred_values = []

    for masked_cap, ground_truth, caption, (pred_cap, pred_seq) in zip(
            df["masked caption"], df["label"], df["caption"], _fill_in_the_blanks(df["masked caption"], gen_pipeline)):
        pred_values.append([masked_cap, caption, pred_cap, ground_truth, pred_seq])
        predictions += exact_match(pred_seq, ground_truth)

    train_pd_out = pd.DataFrame(pred_values,
                                columns=["masked_caption", "caption", "pred_caption", "ground_truth", "pred"])
    csv_output_path = OUTPUT_PATH / "greedy-output.csv"
    train_pd_out.to_csv(csv_output_path)

    print(f"Exact Match Accuracy with {len(df['caption'])} questions: {predictions / len(df['caption'])}.")
    print(f"{csv_output_path} is saved.")
    print(train_pd_out)


def _evaluate_beam_search(df: pd.DataFrame, gen_pipeline: Pipeline, beam_size: int) -> None:
    labels = df["label"].tolist()
    masked_captions = df["masked caption"].tolist()

    beam_predictions = 0
    output = []
    new_labels = [f"<extra_id_0> {label} <extra_id_1>" for label in labels[:10]]  # FIXME: 10?
    ground_truth_probs, pred_probs, preds = _compute_prob_and_pred(masked_captions[:10], gen_pipeline, new_labels,
                                                                   beam_size=beam_size)

    for masked_caption, pred, label, pred_prob, ground_truth_prob in zip(masked_captions, preds, labels, pred_probs,
                                                                         ground_truth_probs):
        beam_predictions += exact_match(pred, label)
        output.append([masked_caption, label, pred, ground_truth_prob.item(), pred_prob.item()])

    train_df = pd.DataFrame(output,
                            columns=["masked_caption", "ground_truth", "pred", "ground_truth_prob", "pred_prob"])
    csv_output_path = OUTPUT_PATH / "beam-search-output.csv"
    train_df.to_csv(csv_output_path)

    print(f"Beam size = {beam_size}, Accuracy with {len(output)} questions: {beam_predictions / len(output)}")
    print(f"{csv_output_path} is saved.")
    print(train_df)


def _evaluate(df: pd.DataFrame, gen_pipeline: Pipeline, beam_size: int) -> None:
    if beam_size == 1:
        _evaluate_exact_match(df, gen_pipeline)
    else:
        _evaluate_beam_search(df, gen_pipeline, beam_size=beam_size)


def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Evaluate the text-only baseline.")
    parser.add_argument("csv_file_path")

    parser.add_argument("--device", type=int, default=0, help="-1 is CPU, otherwise the GPU device ID.")
    parser.add_argument("--framework", default="pt")
    parser.add_argument("--max-data-points", type=int, default=100)
    # The only models that work with the used pipelines are the ones from `MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING`.
    # The model config names can't be obtained easily. You can obtain all the officially supported ones, of all types,
    # but then it's hard to know which ones are in this list.
    # Also, note you still can't easily get the user-uploaded models, as they're resolved dynamically.
    # So we can't provide model name choices.
    # I guess we can check the options from the URL below, though I'm not sure if that's the exact filter tag.
    parser.add_argument("--model", default="t5-base",
                        help="pipeline model. Check the options in https://huggingface.co/models?filter=seq2seq")

    parser.add_argument("--beam-size", type=int, default=1)  # TODO: change to 5

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    df = pd.read_csv(cached_path(args.csv_file_path))[:args.max_data_points]

    gen_pipeline = pipeline("text2text-generation", model=args.model, framework=args.framework, device=args.device)
    gen_pipeline.model.eval()

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    _evaluate(df, gen_pipeline, args.beam_size)


if __name__ == "__main__":
    main()
