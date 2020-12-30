#!/usr/bin/env python
# coding: utf-8
"""Usage: ./txt2csv.py INPUT_FILE > OUTPUT_FILE."""
import csv
import fileinput
import itertools
import sys
from typing import Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")


def grouper(n: int, iterable: Iterable[T]) -> Iterator[Tuple[T, ...]]:
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def main() -> None:
    csv_writer = csv.writer(sys.stdout)
    csv_writer.writerow(["video_id", "question", "answer", "pos_tag", "video_start_time", "video_end_time"])
    for chunk in grouper(6, fileinput.input()):
        meta = [line[:-1] for line in chunk[:4]]  # "video_id", "question", "answer", "pos_tag"
        meta[1] = meta[1].replace("'", "\\'")
        meta.extend(eval(meta[5])[:2])  # Start and end times.
        if len(meta) == 6:
            csv_writer.writerow(meta)


if __name__ == '__main__':
    main()
