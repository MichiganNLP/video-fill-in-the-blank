#!/usr/bin/env python
# coding: utf-8
import json
import sys
import textwrap
from typing import List, Mapping, Tuple

import pandas as pd

QUESTIONS_PER_HIT = 5


def arrange_answers(answers: Mapping[str, str]) -> Tuple[List[List[str]], str]:  # organize answers in one instance
    arranged_answers = [[] for _ in range(QUESTIONS_PER_HIT)]
    comment = ""
    for name in answers:
        if name.startswith("less-2-checkbox"):  # TODO: remove later
            continue
        answer = answers[name].strip().lower()
        for i in range(QUESTIONS_PER_HIT):
            if f"answer{i + 1}-" in name:
                arranged_answers[i].append(answer)
            if f"in-answer-box{i + 1}" in name:
                arranged_answers[i].append(answer)
            if name == "comments":
                comment = answer
    return [sorted(set(t), key=t.index) for t in
            arranged_answers], comment  # remove repeated answer and keep the answer order


def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else sys.stdin
    mturk_results = pd.read_csv(input_path, converters={"Answer.taskAnswers": json.loads})

    # TODO: add some assert to check on the format.

    hits = mturk_results \
        .groupby(["HITId"] + [c for c in mturk_results.columns if c.startswith("Input.")]) \
        .agg({"Answer.taskAnswers": lambda lists: [x for list_ in lists for x in list_]}) \
        .reset_index().to_dict('records')

    for hit in hits:
        print(f"*** HIT {hit['HITId']} ***")
        print()
        print()

        answers, comments = zip(*(arrange_answers(worker_answers) for worker_answers in hit["Answer.taskAnswers"]))

        i = 1
        while f"Input.question{i}" in hit:
            # This URL format (embed) supports specifying an end time.
            print(textwrap.dedent(f"""\
            question: {hit[f"Input.question{i}"].replace("[MASK]", "_____")}
            video: https://www.youtube.com/embed/{hit[f"Input.video{i}_id"]}?start={hit[f"Input.video{i}_start_time"]}\
&end={hit[f"Input.video{i}_end_time"]}
            std. answer: {hit[f"Input.label{i}"]}
            answers:"""))
            for worker_answers in answers:
                print(worker_answers[i - 1])
            print()

            i += 1

        print("Comments:")
        for comment in comments:
            if comment:
                print(comment)
        print()
        print()


if __name__ == '__main__':
    main()
