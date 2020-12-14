#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lqam.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.data_module import QGenDataModule
from lqam.t5_module import T5FillerModel


def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Evaluate the T5 text-only baseline.")

    parser.add_argument("--data-path", default="https://drive.google.com/uc?id=1-JRsjFzP3Qmjti_w8ILV06msXjw4OXoB"
                                               "&export=download")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", "-j", type=int, default=0)

    parser.add_argument("--gpus", type=int)

    # The only models that work with the used pipelines are the ones from `MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING`.
    # The model config names can't be obtained easily. You can obtain all the officially supported ones, of all types,
    # but then it's hard to know which ones are in this list.
    # Also, note you still can't easily get the user-uploaded models, as they're resolved dynamically.
    # So we can't provide model name choices.
    # I guess we can check the options from the URL below, though I'm not sure if that's the exact filter tag.
    parser.add_argument("--model", default="t5-base",
                        help="pipeline model. Check the options in https://huggingface.co/models?filter=seq2seq")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--generation-early-stopping", type=bool)
    parser.add_argument("--no-repeat-ngram-size", type=int)

    parser.add_argument("--predictions-output-path", default="predictions.csv")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_module = QGenDataModule(tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers)

    t5_like_pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    filler = T5FillerModel(t5_like_pretrained_model=t5_like_pretrained_model, tokenizer=tokenizer,
                           generate_kwargs={"num_beams": args.beam_size,
                                            "early_stopping": args.generation_early_stopping,
                                            "no_repeat_ngram_size": args.no_repeat_ngram_size})

    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(filler, test_dataloaders=data_module.val_dataloader(args.data_path))

    predictions = {k: v.tolist() if isinstance(v, torch.Tensor) else v
                   for k, v in next(iter(trainer.evaluation_loop.predictions.predictions.values())).items()}
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(args.predictions_output_path, index=False)
    print(f"Predictions saved in {args.predictions_output_path}. First rows:")
    print()
    pd.options.display.float_format = \
        lambda x: np.format_float_scientific(x, exp_digits=1, precision=0, trim="-") if abs(x) < 0.005 else f"{x:.2f}"
    print(df.head(10))


if __name__ == "__main__":
    main()
