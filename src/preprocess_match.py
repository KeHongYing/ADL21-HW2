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

    train = data[int(test_size * len(data)) :]
    val = data[: int(test_size * len(data))]

    return train, val


def main(args):
    random.seed(args.rand_seed)
    with open(args.data_dir / args.context, "r") as f:
        context = json.load(f)

    with open(args.data_dir / args.data, "r") as f:
        data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    max_len = args.max_len - 2

    output = []
    for d in tqdm(data, desc="preprocessing data..."):
        Id = d["id"]
        question = tokenizer.encode(d["question"])

        paragraph = [context[context_idx] for context_idx in d["paragraphs"]]

        start_list = []
        len_list = []
        if "answers" in d:
            for ans in d["answers"]:
                ans_token = tokenizer.tokenize(ans["text"])
                start_token_pos = len(
                    tokenizer.tokenize(context[d["relevant"]][: ans["start"]])
                )

                start_list.append(start_token_pos)
                len_list.append(len(ans_token))

        context_len = max_len - len(question)
        paragraph_index = []
        tokens = []
        paragraph_start_end = []
        relevant_index = []
        irrelevant_index = []
        raw_paragraph = []
        for idx, p in enumerate(paragraph):
            head = 0
            prev_token_len = 0
            while head < len(p):
                token = tokenizer.tokenize(p[head : head + context_len])
                start, end = -1, -1
                if idx == d["paragraphs"].index(d["relevant"]):
                    for s, l in zip(start_list, len_list):
                        if prev_token_len <= s < prev_token_len + len(token):
                            start = s - prev_token_len
                            end = min(start + l, len(token)) - 1

                if start != -1:
                    relevant_index.append(len(paragraph_index))
                else:
                    irrelevant_index.append(len(paragraph_index))

                paragraph_index.append(d["paragraphs"][idx])
                tokens.append(tokenizer.convert_tokens_to_ids(token))
                paragraph_start_end.append([start, end])
                raw_paragraph.append(p[head : head + context_len])

                prev_token_len += len(token)
                head += context_len

        output.append(
            {
                "id": Id,
                "question": question,
                "token": tokens,
                "start_end": paragraph_start_end,
                "paragraph_index": paragraph_index,
                "relevant_index": relevant_index,
                "irrelevant_index": irrelevant_index,
                "raw_paragraph": raw_paragraph,
            }
        )

    if args.training:
        train, val = train_val_split(output, test_size=0.1)

        with open(args.output_dir / "train.json", "w") as f:
            json.dump(train, f, ensure_ascii=False, indent=4)
        with open(args.output_dir / "val.json", "w") as f:
            json.dump(val, f, ensure_ascii=False, indent=4)
    else:
        with open(args.output_dir / args.data, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


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
        default="./cache/match",
    )
    parser.add_argument(
        "--training", help="preprocess training data or not", action="store_true"
    )
    parser.add_argument("--max_len", type=int, help="token max length.", default=512)
    # model
    parser.add_argument(
        "--backbone", help="bert backbone", type=str, default="bert-base-chinese"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
