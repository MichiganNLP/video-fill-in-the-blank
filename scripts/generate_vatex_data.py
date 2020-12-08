import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

DATA_FOLDER_PATH = Path("/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data"
                        "/multimodal_model/VATEX")
SPLIT_NAME = "val"


def main() -> None:
    df = pd.read_csv(DATA_FOLDER_PATH / f"{SPLIT_NAME}.csv")

    data = [(*row, torch.from_numpy(np.load(DATA_FOLDER_PATH / SPLIT_NAME / f"{row[0]}.npy")).squeeze(0))
            for row in tqdm(df)]  # video ID, caption, masked caption, label, video features

    with open(DATA_FOLDER_PATH / f"{SPLIT_NAME}.pkl", "wb") as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    main()
