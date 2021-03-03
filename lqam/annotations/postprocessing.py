import html
import json
import operator
import re
import sys
from collections import defaultdict
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd
from pandas._typing import FilePathOrBuffer  # noqa
from tqdm.auto import tqdm

from lqam.annotations.metrics import compute_answer_level_annotation_metrics
from lqam.core.metrics import RE_MULTIPLE_SPACES

RE_ANSWER_KEY = re.compile(r"^answer-(?P<question_index>\d+)-(?P<answer_index>\d+)$")
RE_ANSWER_INPUT_KEY = re.compile(r"^answer-input-(?P<question_index>\d+)$")


def format_answer(answer: str) -> str:
    """Useful when wanting to print the raw worker answers, but disregarding casing and spacing.

    Like `lqam.core.metrics.normalize_answer` but changing no token.
    """
    return RE_MULTIPLE_SPACES.sub("", answer.lower()).strip()


def order_worker_answers_by_question(worker_answers: Mapping[str, str]) -> Sequence[Sequence[str]]:
    """Orders a worker's answers by question."""
    answers_by_question = defaultdict(list)
    for k in worker_answers:
        # noinspection PyUnusedLocal
        if match := RE_ANSWER_KEY.match(k) or RE_ANSWER_INPUT_KEY.match(k):
            question_i = int(match.group("question_index"))
            i = int(match.groupdict().get("answer_index", sys.maxsize))
            answers_by_question[question_i].append((i, worker_answers[k]))

    return [[sorted_i_and_question_answers[1]
             for sorted_i_and_question_answers in sorted(i_and_question_answers, key=operator.itemgetter(0))]
            for _, i_and_question_answers in sorted(answers_by_question.items(), key=operator.itemgetter(0))]


def parse_hits(filepath_or_buffer: FilePathOrBuffer, include_accepted: bool = True,
               include_rejected: bool = False) -> Mapping[str, Mapping[str, Any]]:
    df = pd.read_csv(filepath_or_buffer, converters={"Answer.taskAnswers": json.loads})

    if not include_accepted:
        df = df[df["ApprovalTime"].isna()]

    if not include_rejected:
        df = df[df["RejectionTime"].isna()]

    hits = (df
            .groupby(["HITId"] + [c for c in df.columns if c.startswith("Input.")])
            .agg({"Answer.taskAnswers": lambda lists: [x for list_ in lists for x in list_], "AssignmentId": list,
                  "WorkerId": list, "ApprovalTime": list})
            .reset_index()
            .to_dict("records"))

    for hit in hits:
        hit["answers"] = {worker_id: {**{f"question{i + 1}": q_answers
                                         for i, q_answers in
                                         enumerate(order_worker_answers_by_question(worker_answers))},
                                      **{"comments": worker_answers.get("comments")}}
                          for worker_id, worker_answers in zip(hit["WorkerId"], hit["Answer.taskAnswers"])}

        hit["assignment_ids"] = {worker_id: assignment_id
                                 for worker_id, assignment_id in zip(hit["WorkerId"], hit["AssignmentId"])}

        hit["statuses"] = {worker_id: "approved" if approval_time else "submitted"
                           for worker_id, approval_time in zip(hit["WorkerId"], hit["ApprovalTime"])}

        del hit["Answer.taskAnswers"]
        del hit["AssignmentId"]
        del hit["WorkerId"]
        del hit["ApprovalTime"]

        hit["question_count"] = max(int(k[len("Input.question"):]) for k in hit if k.startswith("Input.question"))

        for i in range(hit["question_count"]):
            df[f"Input.question{i + 1}"] = df[f"Input.question{i + 1}"].apply(html.unescape)

    return {hit["HITId"]: hit for hit in hits}


def hits_to_instances(hits: Mapping[str, Mapping[str, Any]]) -> Mapping[str, MutableMapping[str, Any]]:
    return {
        f"{hit_id}-{i}": {
            "question": hit[f"Input.question{i}"],
            "video_id": hit[f"Input.video{i}_id"],
            "video_start_time": hit[f"Input.video{i}_start_time"],
            "video_end_time": hit[f"Input.video{i}_end_time"],
            # This YouTube URL format (embed) supports specifying an end time.
            "video_url": f"https://www.youtube.com/embed/{hit[f'Input.video{i}_id']}?start="
                         f"{hit[f'Input.video{i}_start_time']}&end={hit[f'Input.video{i}_end_time']}",
            "label": hit[f"Input.label{i}"],
            "answers": {worker_id: answers.get(f"question{i}", []) for worker_id, answers in hit["answers"].items()},
            "assignment_ids": hit["assignment_ids"],
            "statuses": hit["statuses"],
        }
        for hit_id, hit in hits.items()
        for i in range(1, hit["question_count"] + 1)
    }


def compute_instances_by_worker_id(
        instances: Mapping[str, Any],
        compute_np_answers: bool = False) -> MutableMapping[str, Sequence[Mapping[str, Any]]]:
    instances_by_worker_id = defaultdict(list)

    for instance in tqdm(instances.values(), desc="Computing the instances by worker"):
        if compute_np_answers:
            formatted_answers = {worker_id: [format_answer(answer) for answer in answers]
                                 for worker_id, answers in instance["answers"].items()}

            answer_level_metrics = compute_answer_level_annotation_metrics(instance["question"], formatted_answers,
                                                                           instance["label"])
        else:
            formatted_answers = None
            answer_level_metrics = None

        for worker_id, answers in instance["answers"].items():
            worker_instance = {
                **{k: v for k, v in instance.items() if k not in {"answers", "assignment_ids", "statuses"}},
                "answers": answers,
                "assignment_id": instance["assignment_ids"][worker_id],
                "status": instance["statuses"][worker_id],
            }

            if compute_np_answers:
                worker_instance["np_answers"] = [answer
                                                 for answer in formatted_answers[worker_id]
                                                 if answer_level_metrics[worker_id][answer]["np"]]

            instances_by_worker_id[worker_id].append(worker_instance)

    return instances_by_worker_id
