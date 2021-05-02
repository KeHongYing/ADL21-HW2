import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from typing import List

from dataset import QADataset
from model import QAClassifier

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def reconstruct(paragraph: str, token: List[int], tokenizer: BertTokenizer) -> str:
    start, end = 0, len(paragraph)
    target = " ".join(map(str, token))

    while start < end:
        t = tokenizer.encode(paragraph[start:end])[1:-1]
        t_str = " ".join(map(str, t))

        is_start_with_target = t_str.startswith(target)
        is_end_with_targert = t_str.endswith(target)
        if not is_start_with_target:
            start += 1
        if not is_end_with_targert:
            end -= 1

        if is_start_with_target and is_end_with_targert:
            if t_str == target:
                break
            end -= 1

    return paragraph[start:end].strip()


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    data = json.loads(args.test_file.read_text())
    dataset = QADataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn
    )

    model = QAClassifier(args.backbone).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    result = {}

    with torch.set_grad_enabled(False):
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description("[predict]")

            for data in tepoch:
                tokens = data["token"].to(args.device)
                pred = model(tokens)["start_end"]

                for idx, p in enumerate(pred):
                    head, tail = data["index"][idx]

                    start = p[..., 0][head:tail].argmax()
                    end = p[..., 1][start + head : tail].argmax() + start
                    paragraph = data["paragraph"][idx]

                    Id = data["id"][idx]
                    token = tokens[idx][head:tail]
                    result[Id] = reconstruct(
                        paragraph[
                            start : -(len(token) - end - 2)
                            if (len(token) - end - 2) > 0
                            else len(paragraph)
                        ],
                        token[start : end + 1].tolist(),
                        tokenizer,
                    )

    with open(args.pred_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file", type=Path, help="Path to the test file.", required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--ckpt_path", type=Path, help="Path to model checkpoint.", required=True
    )
    parser.add_argument("--pred_file", type=Path, default="output.json")

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
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
    main(args)
