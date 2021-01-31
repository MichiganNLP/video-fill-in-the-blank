#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.methods.dataset import QGenDataModule
from lqam.methods.t5_filler_model import T5FillerModel
from lqam.methods.t5_multi_modal_module import T5AndVisual

def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Evaluate the T5 multi-modal baseline.")

    parser.add_argument("--data-path", default="https://drive.google.com/uc?id=1-JRsjFzP3Qmjti_w8ILV06msXjw4OXoB"
                                               "&export=download")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", "-j", type=int, default=0,
                        help="data loader workers. Each worker batch-tokenizes in parallel, "
                             "so maybe don't make this number equal to the number of CPU cores but just a small "
                             "natural number.")

    parser.add_argument("--gpus", type=int)
    parser.add_argument("--visual-size", type=int, default=1024)

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

    parser.add_argument("--predictions-output-path", default="predictions.csv")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1 for the Adam optimizer")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2 for the Adam optimizer")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--lr-scheduling", choices=("", "linear_with_warmup"), default="linear_with_warmup")
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--default-root-dir", type=str, default="/scratch/mihalcea_root/mihalcea1/shared_data/qgen/VATEX/multimodal_model/")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--has-visual", action="store_true")
    parser.add_argument("--visual-data-path", default="/scratch/mihalcea_root/mihalcea1/shared_data/qgen/VATEX/I3D_video_features")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_module = QGenDataModule(tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, hasVisual=args.has_visual)

    if args.has_visual:
        t5_like_pretrained_model = T5AndVisual.from_pretrained(args.model, visual_size=args.visual_size)
        t5_like_pretrained_model.set_encoder()
    else:
        t5_like_pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    filler = T5FillerModel(t5_like_pretrained_model=t5_like_pretrained_model, tokenizer=tokenizer,
                           only_noun_phrases=args.only_noun_phrases,
                           optimizer_args = {'lr': args.lr,
                                             'beta1': args.beta1,
                                             'beta2': args.beta2,
                                             'lr_scheduling': args.lr_scheduling,
                                             'epochs': args.epochs,
                                             'weight_decay': args.weight_decay
                                            },
                           generate_kwargs={"max_length": args.max_length,
                                            "num_beams": args.beam_size,
                                            "early_stopping": args.generation_early_stopping,
                                            "no_repeat_ngram_size": args.no_repeat_ngram_size})

    visual_data_path = None
    if args.has_visual:
        visual_data_path = args.visual_data_path

    train_dataloaders = data_module.train_dataloader(visual_data_path = visual_data_path)
    val_dataloaders = data_module.val_dataloader(visual_data_path = visual_data_path)
    test_dataloaders=data_module.test_dataloader( visual_data_path = visual_data_path)
    trainer = pl.Trainer(gpus=args.gpus, default_root_dir=args.default_root_dir, fast_dev_run = args.fast_dev_run, max_epochs = args.epochs)
    if not args.test:
        trainer.fit(filler, train_dataloaders, val_dataloaders)
    trainer.test(filler, test_dataloaders=test_dataloaders)

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
