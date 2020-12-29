#!/usr/bin/env python
# coding: utf-8
import itertools
import json
import operator
import re
import sys
from collections import defaultdict
from typing import Mapping, Sequence

import pandas as pd

# TODO: remove the old format.
RE_ANSWER_KEY = re.compile(r"^answer-?(?P<question_index>\d+)-(?P<answer_index>\d+)$")
RE_ANSWER_INPUT_KEY = re.compile(r"^(?:in-answer-box|answer-input-)(?P<question_index>\d+)$")


def format_answer(answer: str) -> str:
    return answer.strip().lower().replace("  ", " ")


def order_worker_answers_by_question(answers: Mapping[str, str]) -> Sequence[Sequence[str]]:
    """Orders a worker's answers by question."""
    answers_by_question = defaultdict(list)
    for name in answers:
        # noinspection PyUnusedLocal
        if match := RE_ANSWER_KEY.match(name) or (match := RE_ANSWER_INPUT_KEY.match(name)):
            question_i = int(match.group("question_index"))
            i = match.groupdict().get("answer_index", sys.maxsize)
            answers_by_question[question_i].append((i, format_answer(answers[name])))

    return [[sorted_i_and_question_answers[1]
             for sorted_i_and_question_answers in sorted(i_and_question_answers, key=operator.itemgetter(0))]
            for _, i_and_question_answers in sorted(answers_by_question.items(), key=operator.itemgetter(0))]


def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else sys.stdin
    mturk_results = pd.read_csv(input_path, converters={"Answer.taskAnswers": json.loads})

    hits = (mturk_results
            .groupby(["HITId"] + [c for c in mturk_results.columns if c.startswith("Input.")])
            .agg({"Answer.taskAnswers": lambda lists: [x for list_ in lists for x in list_],
                  "WorkerId": lambda lists: list(lists)})
            .reset_index()
            .to_dict("records"))

    for hit in hits:
        print(f"*** HIT {hit['HITId']} ***")
        print()
        print()

        answers = [order_worker_answers_by_question(worker_answers_map)
                   for worker_answers_map in hit["Answer.taskAnswers"]]
        worker_ids = hit["WorkerId"]

        for i in itertools.count(start=1):
            if f"Input.question{i}" not in hit:
                break

            df = pd.DataFrame((worker_answers[i - 1] for worker_answers in answers),
                              index=pd.Index(worker_ids, name="Worker ID"))
            df.columns = [f"Ans. {j + 1}" for j in range(len(df.columns))]
            df[df.isna()] = ""
            # Convert the index into a column. Otherwise, the index name and column names are output in different lines.
            df = df.reset_index()
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
                formatted_question_answers = df.to_string(index=False)

            # This YouTube URL format (embed) supports specifying an end time.
            print(f"""\
Question: {hit[f"Input.question{i}"].replace("[MASK]", "_____")}
Video: https://www.youtube.com/embed/{hit[f"Input.video{i}_id"]}?start={hit[f"Input.video{i}_start_time"]}\
&end={hit[f"Input.video{i}_end_time"]}
Std. answer: {hit[f"Input.label{i}"]}
Worker answers:
{formatted_question_answers}
""")

        comments = (worker_answers_map.get("comments") for worker_answers_map in hit["Answer.taskAnswers"])
        print("Comments:")
        for worker_id, comment in zip(worker_ids, comments):
            if comment:
                print(f"{worker_id:>14}: {comment}")
        print()
        print()


if __name__ == "__main__":
    main()
