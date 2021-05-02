import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List

from dataset import QADataset
from model import QAClassifier

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def reconstruct(paragraph: str, start: int, end: int, tokenizer: AutoTokenizer) -> str:
    token = tokenizer(paragraph, return_offsets_mapping=True)
    head = token["offset_mapping"][start][0]
    tail = token["offset_mapping"][end][1]
    return "".join(paragraph[head:tail]).strip()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
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
        default="hfl/chinese-xlnet-base",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
