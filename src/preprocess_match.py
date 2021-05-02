import json
import random
import logging
from typing import List, Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm
from transformers import BertTokenizer

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

    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    max_len = args.max_len - 2

    output = []
    mark = set(["。", "！", "？", "；"])
    for d in tqdm(data, desc="preprocessing data..."):
        Id = d["id"]
        question = tokenizer.encode(d["question"])

        paragraph = [
            context[context_idx] + ("。" if context[context_idx][-1] not in mark else "")
            for context_idx in d["paragraphs"]
        ]

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
            mark_list = construct_mark_table(p)
            head = 0
            prev_token_len = 0
            while head < len(p):
                tail = mark_list[min(head + context_len, len(p)) - 1] + 1
                tail = tail if tail != head else head + context_len
                token = tokenizer.tokenize(p[head:tail])
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
                raw_paragraph.append(p[head:tail])

                prev_token_len += len(token)
                head = tail

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
        "--backbone",
        help="bert backbone",
        type=str,
        default="voidful/albert_chinese_large",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
