import json
import random

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm
from transformers import BertTokenizer
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    random.seed(args.rand_seed)
    with open(args.data_dir / args.data, "r") as f:
        data = json.load(f)

    output = []
    for d in tqdm(data, desc="preprocessing data..."):
        Id = d["id"]
        question = d["question"]

        for idx, (s, e) in enumerate(d["start_end"]):
            if s != -1:
                paragraph = d["paragraph"][idx]
                relevant = d["paragraph_index"][idx]

                output.append(
                    {
                        "id": Id,
                        "question": question,
                        "paragraph": paragraph,
                        "relevant": relevant,
                        "start_end": [s, e],
                    }
                )

    with open(args.output_dir / args.data, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./cache/match/",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to the question.",
        default="train.json",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=13)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/QA",
    )
    parser.add_argument(
        "--training", help="preprocess training data or not", action="store_true"
    )
    parser.add_argument("--max_len", type=int, help="token max length.", default=512)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
