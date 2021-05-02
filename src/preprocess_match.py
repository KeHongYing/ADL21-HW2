import json
import random
import logging
from typing import List, Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_val_split(
    data: List[Dict], test_size: float = 0.2, shuffle: bool = True
) -> (List[Dict], List[Dict]):
    if shuffle:
        random.shuffle(data)

    train = data[int(test_size * len(data)) :]
    val = data[: int(test_size * len(data))]

    return train, val


def construct_mark_table(paragraph: str) -> List[int]:
    pos = -1
    ret = []
    mark = set(["。", "！", "？", "；"])

    for idx, p in enumerate(paragraph):
        if p in mark:
            pos = idx

        ret.append(pos)
    return ret


def main(args):
    random.seed(args.rand_seed)
    with open(args.data_dir / args.context, "r") as f:
        context = json.load(f)

    with open(args.data_dir / args.data, "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    max_len = args.max_len - 3

    output = []
    mark = set(["。", "！", "？", "；"])
    test_file = []
    for d in tqdm(data, desc="preprocessing data..."):
        Id = d["id"]
        question = [tokenizer.bos_token_id] + tokenizer.encode(d["question"])

        paragraph = [
            context[context_idx] + ("。" if context[context_idx][-1] not in mark else "")
            for context_idx in d["paragraphs"]
        ]

        context_len = max_len - len(question)
        paragraph_index = []
        tokens = []
        paragraph_start_end = []
        relevant_index = []
        irrelevant_index = []
        raw_paragraph = []
        for idx, p in enumerate(paragraph):
            mark_list = construct_mark_table(p)
            head = 0
            while head < len(p):
                tail = mark_list[min(head + context_len, len(p)) - 1] + 1
                tail = tail if tail != head else head + context_len
                token = tokenizer(p[head:tail], return_offsets_mapping=True)
                start, end = -1, -1
                if idx == d["paragraphs"].index(d["relevant"]) and "answers" in d:
                    for i, (s, e) in enumerate(token["offset_mapping"]):
                        for ans in d["answers"]:
                            if head + s <= ans["start"] < head + e:
                                start = i
                                break
                        if start != -1:
                            expected_ans = ans["text"]
                            break

                    for i, (s, e) in enumerate(token["offset_mapping"]):
                        for ans in d["answers"]:
                            if (
                                head + s
                                <= ans["start"] + len(ans["text"]) - 1
                                < head + e
                            ):
                                end = i
                                break
                        if end != -1:
                            break

                if start != -1:
                    relevant_index.append(len(paragraph_index))
                    end = len(token["input_ids"]) - 2 - 1 if end == -1 else end
                else:
                    irrelevant_index.append(len(paragraph_index))

                paragraph_index.append(d["paragraphs"][idx])
                tokens.append(token["input_ids"][:-2])
                paragraph_start_end.append([start, end])
                raw_paragraph.append(p[head:tail])

                head = tail

                if start != -1:
                    test_file.append(
                        {
                            "expected_ans": expected_ans,
                            "your_ans": tokenizer.decode(
                                token["input_ids"][start : end + 1]
                            ),
                        }
                    )

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

    with open("xlnet_ans.json", "w") as f:
        json.dump(test_file, f, ensure_ascii=False, indent=4)


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
        "--backbone",
        help="xlnet backbone",
        type=str,
        default="hfl/chinese-xlnet-base",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
