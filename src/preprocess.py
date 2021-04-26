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


def train_val_split(data, test_size=0.2, shuffle=True):
    if shuffle:
        random.shuffle(data)

    train = data[int(0.2 * len(data)) :]
    val = data[: int(0.2 * len(data))]

    return train, val


def main(args):
    random.seed(args.rand_seed)
    with open(args.data_dir / args.context, "r") as f:
        context = json.load(f)

    with open(args.data_dir / args.data, "r") as f:
        data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    max_len = args.max_len - 2

    output = []
    for d in tqdm(data, desc="preprocessing data..."):
        Id = d["id"]
        question = tokenizer.tokenize(d["question"])
        paragraphs = [
            [
                tokenizer.tokenize(
                    context[idx][
                        i
                        * (max_len - len(question)) : (i + 1)
                        * (max_len - len(question))
                    ]
                )
                for i in range(
                    np.ceil(len(context[idx]) / (max_len - len(question))).astype(
                        np.int64
                    )
                )
            ]
            for idx in d["paragraphs"]
        ]

        relevant = -1
        correct, incorrect = -1, []
        if "relevant" in d:
            relevant = d["relevant"]
            correct = d["paragraphs"].index(relevant)
            incorrect = [
                idx
                for idx, context_id in enumerate(d["paragraphs"])
                if context_id != relevant
            ]

        answers = []
        if "answers" in d:
            answers = d["answers"]
            for i in range(len(answers)):
                answers[i]["text"] = tokenizer.tokenize(answers[i]["text"])

        output.append(
            {
                "id": Id,
                "question": question,
                "paragraphs": paragraphs,
                "relevant": relevant,
                "answers": answers,
                "correct": correct,
                "incorrect": incorrect,
            }
        )

    if args.training:
        train, val = train_val_split(output, test_size=0.1)

        with open(args.output_dir / "train.json", "w") as f:
            json.dump(train, f)
        with open(args.output_dir / "val.json", "w") as f:
            json.dump(val, f)
    else:
        with open(args.output_dir / args.data, "w") as f:
            json.dump(output, f)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="Path to the context.",
        default="context.json",
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
        default="./cache/",
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
