#!/usr/bin/env python
import json
import sys
from collections import defaultdict

import pandas as pd

from lqam_data import QUESTIONS_PER_HIT

NEEDED_COUNT = 10
assert NEEDED_COUNT % QUESTIONS_PER_HIT == 0


def main():
    input_ = sys.argv[1] if len(sys.argv) > 1 else sys.stdin
    df = pd.read_csv(input_)
    assert len(df) >= NEEDED_COUNT

    with open("used_indices.json") as file:
        used_indices = set(json.load(file)["val"])

    selected_indices = [i for i in range(NEEDED_COUNT) if i not in used_indices]
    assert len(selected_indices) == NEEDED_COUNT

    grouped_selected_indices = [selected_indices[j * QUESTIONS_PER_HIT:(j + 1) * QUESTIONS_PER_HIT]
                                for j in range(len(selected_indices) // QUESTIONS_PER_HIT)]

    output_dict = defaultdict(list)

    for hit_indices in grouped_selected_indices:
        for i in range(QUESTIONS_PER_HIT):
            output_dict[f"video{i + 1}_id"].append(df.iloc[hit_indices[i]]["videoID"][:-14])
            output_dict[f"video{i + 1}_start_time"].append(int(df.iloc[hit_indices[i]]["videoID"][-13:-7]))
            output_dict[f"video{i + 1}_end_time"].append(int(df.iloc[hit_indices[i]]["videoID"][-6:]))
            output_dict[f"question{i + 1}"].append(df.iloc[hit_indices[i]]["masked caption"].replace("<extra_id_0>",
                                                                                                     "[MASK]"))
            output_dict[f"label{i + 1}"].append(df.iloc[hit_indices[i]]["label"])

    print(pd.DataFrame(output_dict).to_csv(index=False))


if __name__ == "__main__":
    main()
