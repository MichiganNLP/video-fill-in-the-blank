#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
import traceback

import boto3
import pandas as pd
from tqdm.auto import tqdm

from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("path_or_url", metavar="BONUS_CSV_FILE", nargs="?", default="-")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    args.input = sys.stdin if args.path_or_url == "-" else cached_path(args.path_or_url)

    return args


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    kwargs = {}
    if not args.production:
        kwargs["endpoint_url"] = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"

    client = boto3.client("mturk", region_name="us-east-1", **kwargs)  # Always use this region for MTurk.

    for row in tqdm(df.itertuples(), total=len(df)):
        try:
            client.send_bonus(WorkerId=row["worker_id"], BonusAmount=row["bonus_amount"],
                              AssignmentId=row["assignment_id"], Reason=row["reason"],
                              UniqueRequestToken=row.get("uuid"))  # Useful to avoid paying the same twice.
        except Exception:  # noqa
            traceback.print_exc()
            print(row, file=sys.stderr)


if __name__ == "__main__":
    main()
