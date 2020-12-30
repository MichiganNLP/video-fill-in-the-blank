import json
import operator
import re
import sys
from collections import defaultdict
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd
from pandas._typing import FilePathOrBuffer  # noqa

QUESTIONS_PER_HIT = 5

# TODO: remove the old format.
RE_ANSWER_KEY = re.compile(r"^answer-?(?P<question_index>\d+)-(?P<answer_index>\d+)$")
RE_ANSWER_INPUT_KEY = re.compile(r"^(?:in-answer-box|answer-input-)(?P<question_index>\d+)$")

RE_A_AN_THE = re.compile(r"\b(?:an?|the)\b")
RE_PUNCTUATION = re.compile(r"[.,/#!$%^&*;:{}=\-_`~()]")  # TODO: change for something more native? but also update js
RE_MULTIPLE_SPACES = re.compile(r"\s{2,}")


def format_answer(answer: str) -> str:
    """Useful when wanting to print the raw worker answers, but disregarding casing and spacing."""
    return answer.strip().lower().replace("  ", " ")


def normalize_answer(answer: str) -> str:
    """Should correspond to the JavaScript function `normalizeAnswerToLookForRepetitions`.

    Useful when looking for repetitions or computing measures.
    """
    return RE_MULTIPLE_SPACES.sub(" ", RE_A_AN_THE.sub("", RE_PUNCTUATION.sub("", answer.lower()))).strip()


def order_worker_answers_by_question(worker_answers: Mapping[str, str]) -> Sequence[Sequence[str]]:
    """Orders a worker's answers by question."""
    answers_by_question = defaultdict(list)
    for k in worker_answers:
        # noinspection PyUnusedLocal
        if match := RE_ANSWER_KEY.match(k) or (match := RE_ANSWER_INPUT_KEY.match(k)):
            question_i = int(match.group("question_index"))
            i = match.groupdict().get("answer_index", sys.maxsize)
            answers_by_question[question_i].append((i, worker_answers[k]))

    return [[sorted_i_and_question_answers[1]
             for sorted_i_and_question_answers in sorted(i_and_question_answers, key=operator.itemgetter(0))]
            for _, i_and_question_answers in sorted(answers_by_question.items(), key=operator.itemgetter(0))]


def parse_hits(filepath_or_buffer: FilePathOrBuffer) -> Mapping[str, Mapping[str, Any]]:
    df = pd.read_csv(filepath_or_buffer, converters={"Answer.taskAnswers": json.loads})
    hits = (df
            .groupby(["HITId"] + [c for c in df.columns if c.startswith("Input.")])
            .agg({"Answer.taskAnswers": lambda lists: [x for list_ in lists for x in list_],
                  "WorkerId": lambda lists: list(lists)})
            .reset_index()
            .to_dict("records"))

    for hit in hits:
        hit["answers"] = {worker_id: {**{f"question{i + 1}": q_answers
                                         for i, q_answers in
                                         enumerate(order_worker_answers_by_question(worker_answers))},
                                      **{"comments": worker_answers.get("comments")}}
                          for worker_id, worker_answers in zip(hit["WorkerId"], hit["Answer.taskAnswers"])}
        del hit["WorkerId"]
        del hit["Answer.taskAnswers"]

        hit["question_count"] = max(int(k[len("Input.question"):]) for k in hit if k.startswith("Input.question"))

    return {hit["HITId"]: hit for hit in hits}


def hits_to_instances(hits: Mapping[str, Mapping[str, Any]]) -> Mapping[str, MutableMapping[str, Any]]:
    return {
        f"{hit_id}-{i}": {
            "question": hit[f"Input.question{i}"],
            "video_id": hit[f"Input.video{i}_id"],
            "start_time": hit[f"Input.video{i}_start_time"],
            "end_time": hit[f"Input.video{i}_end_time"],
            # This YouTube URL format (embed) supports specifying an end time.
            "video_url": f"https://www.youtube.com/embed/{hit[f'Input.video{i}_id']}?start="
                         f"{hit[f'Input.video{i}_start_time']}&end={hit[f'Input.video{i}_end_time']}",
            "label": hit[f"Input.label{i}"],
            "answers": {worker_id: answers[f"question{i}"] for worker_id, answers in hit["answers"].items()},
        }
        for hit_id, hit in hits.items()
        for i in range(1, hit["question_count"] + 1)
    }
