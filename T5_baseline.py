import torch
import itertools
import re
import os
import argparse
import pandas as pd
import string

from typing import Iterable, Mapping, Tuple, TypeVar, Union, List

from tqdm.auto import tqdm
from transformers import Pipeline, pipeline
from transformers import PreTrainedTokenizerBase

RE_EXTRA_ID = re.compile(r"<extra_id_\d+>")

T = TypeVar("T")

FRAMEWORK = "pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NUMBER = -1 if torch.cuda.is_available() else 0

regex = re.compile(r'\b(a|an|the)\b|[%s]' % re.escape(string.punctuation))
NUM_BEAMS = 5


def equal(pred_label, ground_truth):
    pred_label = re.sub(regex, '', pred_label.lower()).strip()
    ground_truth = re.sub(regex, '', ground_truth.lower()).strip()
    return 1 if pred_label == ground_truth else 0


# From https://stackoverflow.com/a/5434936/1165181
def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def compute_mask_values(generated_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> Mapping[str, Iterable[str]]:
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    
    extra_id_indices = {token: i for i, token in enumerate(tokens) if RE_EXTRA_ID.match(token)}
    extra_id_indices["</s>"] = len(tokens)
    
    return {extra_id_token: tokens[extra_id_indices[extra_id_token] + 1:extra_id_indices[next_extra_id_token]]
            for extra_id_token, next_extra_id_token in pairwise(extra_id_indices)}


def substitute_mask_values(input_text: str, generated_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) \
        -> Tuple[str, str]:
    mask_values = compute_mask_values(generated_ids, tokenizer)
    
    tokens = tokenizer.tokenize(input_text)
    
    new_tokens = []
    
    for token in tokens:
        pred_tokens = mask_values.get(token)
        if pred_tokens:
            new_tokens.extend(pred_tokens)
        else:
            new_tokens.append(token)
    
    # for only one blank
    return tokenizer.convert_tokens_to_string(new_tokens), \
           tokenizer.convert_tokens_to_string(mask_values['<extra_id_0>'])


def fill_in_the_blanks_multi(texts: Union[str, list], nlp_pipeline: Pipeline) -> str:
    with torch.no_grad():
        output = nlp_pipeline(texts, return_tensors=True, return_text=False)
        if isinstance(texts, list):
            generated_ids_list = [x['generated_token_ids'] for x in output]
            preds = [substitute_mask_values(texts[i], ids, nlp_pipeline.tokenizer)
                     for i, ids in enumerate(generated_ids_list)]
        elif isinstance(texts, str):
            generated_ids = output[0]['generated_token_ids']
            preds = [substitute_mask_values(texts, generated_ids, nlp_pipeline.tokenizer)]
    
    return preds


def compute_label_prob(logits: List[torch.Tensor], label: List[torch.Tensor]) -> float:
    N = len(logits)
    results = torch.zeros(N, device=logits[0].device, dtype=logits[0].dtype)
    for i in range(N):
        probs = logits[i].softmax(dim=-1)
        results[i] = probs[torch.arange(len(label[i])), label[i]].prod()
    # There should be just a few factors, so the product should be numerically stable.
    return results


def compute_pred(input_ids: List[torch.Tensor], nlp_pipeline: Pipeline) -> Tuple[str, torch.Tensor, torch.Tensor]:
    # `generate` does it, however it doesn't return the logits.
    # So we do one more forward to get the logits.
    pred_list, pred_logits_list, pred_labels_list = [], [], []
    
    generated_ids = nlp_pipeline.model.generate(input_ids,
                                                num_beams=NUM_BEAMS,
                                                no_repeat_ngram_size=2,
                                                early_stopping=True)

    for i, beam_output in enumerate(generated_ids):
        print("{}: {}".format(i, nlp_pipeline.tokenizer.decode(beam_output)))
    
    for ids in generated_ids:
        pred_tokens = compute_mask_values(ids, nlp_pipeline.tokenizer)["<extra_id_0>"]
        
        print(pred_tokens)
        
        pred = nlp_pipeline.tokenizer.convert_tokens_to_string(pred_tokens)
        
        # `pred_tokens` could have some extra gibberish, so we clean it.
        pred_label_tokens = ["<extra_id_0>"] + pred_tokens + ["<extra_id_1>"]
        
        pred_label = torch.tensor(nlp_pipeline.tokenizer.convert_tokens_to_ids(pred_label_tokens), device=DEVICE)
        
        pred_logits = nlp_pipeline.model(ids.unsqueeze(0), labels=pred_label.unsqueeze(0))[1].squeeze(0)
        
        pred_list.append(pred)
        pred_logits_list.append(pred_logits)
        pred_labels_list.append(pred_label)
    
    return pred_list, pred_logits_list, pred_labels_list


def compute_prob_and_pred(texts: List[str], nlp_pipeline: Pipeline, ground_truth: List[str]) -> Tuple[float, float, str]:
    input_ids = nlp_pipeline.tokenizer(texts, add_special_tokens=True, padding=True,
                                       return_tensors=FRAMEWORK).input_ids.to(DEVICE)
    ground_truth_labels = nlp_pipeline.tokenizer(ground_truth, add_special_tokens=False, padding=True,
                                                 return_tensors=FRAMEWORK).input_ids.to(DEVICE)
    
    with torch.no_grad():
        ground_truth_logits = nlp_pipeline.model(input_ids, labels=ground_truth_labels)[1]
        pred, pred_logits, pred_label = compute_pred(input_ids, nlp_pipeline)
        ground_truth_prob = compute_label_prob(ground_truth_logits, ground_truth_labels)
        pred_prob = compute_label_prob(pred_logits, pred_label)
    
    return ground_truth_prob, pred_prob, pred


def run_exact_match(csv_file_path: str, nlp_pipeline: Pipeline):
    # run T5 on training set using greedy approach
    
    predictions = 0
    pred_values = []
    
    data_frame = pd.read_csv(open(csv_file_path))
    labels = data_frame['label'][:100].tolist()
    captions = data_frame['caption'][:100].tolist()
    masked_captions = data_frame['masked caption'][:100].tolist()
    
    # predictions
    pred_captions = fill_in_the_blanks_multi(masked_captions, nlp_pipeline)
    
    for i, (pred_cap, pred_seq) in enumerate(pred_captions):
        masked_cap = masked_captions[i]
        ground_truth = labels[i]
        pred_values.append([masked_cap, captions[i], pred_cap, ground_truth, pred_seq])
        predictions += equal(pred_seq, ground_truth)
    
    train_pd_out = pd.DataFrame(
        pred_values, columns=["masked_caption", "caption", "pred_caption", "ground_truth", "pred"])
    csv_output_path = os.path.join('./output', 'greedy-output.csv')
    train_pd_out.to_csv(csv_output_path)
    
    if not os.path.exists('./output'):
        os.mkdir('./output')
    print(f"Exact Match Accuracy with {len(captions)} questions: {predictions / len(captions)}.")
    print(f"{csv_output_path} is saved.")
    print(train_pd_out)


def run_beam_search(csv_file_path: str, nlp_pipeline: Pipeline, num_return_sequences: int):
    
    data_frame = pd.read_csv(open(csv_file_path))
    labels = data_frame['label'][:100].tolist()
    captions = data_frame['caption'][:100].tolist()
    masked_captions = data_frame['masked caption'][:100].tolist()
    
    beam_5_predictions = 0
    output = []
    new_labels = ["<extra_id_0> " + x + " <extra_id_1>" for x in labels[:10]]
    ground_truth_prob, pred_prob, pred = compute_prob_and_pred(masked_captions[:10], nlp_pipeline, new_labels)

    for i in range(len(pred)):
        beam_5_predictions += equal(pred[i], labels[i])
        output.append([masked_captions[i], labels[i], pred[i], ground_truth_prob[i].item(), pred_prob[i].item()])

    train_pd_out = pd.DataFrame(output,
                                columns=["masked_caption", "ground_truth", "pred", "ground_truth_prob", "pred_prob"])
    csv_output_path = os.path.join('./output', 'beam-search-output.csv')
    train_pd_out.to_csv(csv_output_path)

    if not os.path.exists('./output'):
        os.mkdir('./output')
    print(f'Beam Size = {NUM_BEAMS}, Accuracy with {len(output)} questions: {beam_5_predictions / len(output)}')
    print(f"{csv_output_path} is saved.")
    print(train_pd_out)


def main():
    parser = argparse.ArgumentParser(description="run T5 model.")
    parser.add_argument('--csv_file_path', type=str, required=True,
                        help='csv file path you want to run T5 model on')
    parser.add_argument('--metric', type=str, required=True,
                        help='"greedy" or "beam_search"')
    
    metric = parser.parse_args().metric
    csv_file_path = parser.parse_args().csv_file_path

    t5_pipeline = pipeline("text2text-generation", framework=FRAMEWORK)
    t5_pipeline.model.eval()
    # run evaluation mode
    if metric == 'greedy':
        run_exact_match(csv_file_path, t5_pipeline)
    elif metric == 'beam_search':
        run_beam_search(csv_file_path, t5_pipeline)


if __name__ == '__main__':
    main()
