#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lqam.methods.dataset import QGenDataModule, URL_DATA_VAL
from lqam.methods.t5_filler_model import T5FillerModel
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults


def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Evaluate the T5 text-only baseline.")

    parser.add_argument("--data-path", default=URL_DATA_VAL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", "-j", type=int, default=0,
                        help="data loader workers. Each worker batch-tokenizes in parallel, "
                             "so maybe don't make this number equal to the number of CPU cores but just a small "
                             "natural number.")

    parser.add_argument("--gpus", type=int)

    # The only models that work with the used pipelines are the ones from `MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING`.
    # The model config names can't be obtained easily. You can obtain all the officially supported ones, of all types,
    # but then it's hard to know which ones are in this list.
    # Also, note you still can't easily get the user-uploaded models, as they're resolved dynamically.
    # So we can't provide model name choices.
    # I guess we can check the options from the URL below, though I'm not sure if that's the exact filter tag.
    parser.add_argument("--model", default="t5-base",
                        help="pipeline model. Check the options in https://huggingface.co/models?filter=seq2seq")
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--generation-early-stopping", action="store_true")
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument("--only-noun-phrases", action="store_true")

    # enable reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
<<<<<<< HEAD
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
=======
    parser.add_argument("--no-deterministic", dest="deterministic" action="store_false")
>>>>>>> 627e3e5 (Update scripts/evaluate_text_only_baseline.py)
    
    parser.add_argument("--predictions-output-path", default="predictions.csv")

    return parser.parse_args()


def _pandas_float_format(x: float) -> str:
    if x == 0:
        return "0"
    elif abs(x) < 0.005:
        return np.format_float_scientific(x, exp_digits=1, precision=0, trim="-")
    else:
        return f"{x:.2f}"


def main() -> None:
    args = _parse_args()

    pl.seed_everything(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_module = QGenDataModule(tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers)

    t5_like_pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    filler = T5FillerModel(t5_like_pretrained_model=t5_like_pretrained_model, tokenizer=tokenizer,
                           only_noun_phrases=args.only_noun_phrases,
                           generate_kwargs={"max_length": args.max_length,
                                            "num_beams": args.beam_size,
                                            "early_stopping": args.generation_early_stopping,
                                            "no_repeat_ngram_size": args.no_repeat_ngram_size})

    trainer = pl.Trainer(gpus=args.gpus, benchmark=args.benchmark, deterministic=args.deterministic)
    trainer.test(filler, test_dataloaders=data_module.val_dataloader(args.data_path))

    predictions = {k: v.tolist() if isinstance(v, torch.Tensor) else v
                   for k, v in next(iter(trainer.evaluation_loop.predictions.predictions.values())).items()}
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(args.predictions_output_path, index=False)
    print(f"Predictions saved in {args.predictions_output_path}. First rows:")
    print()
    pd.options.display.float_format = _pandas_float_format
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0,
                           "display.max_colwidth", None):
        print(df.head(10))


if __name__ == "__main__":
    main()
