import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from dataset import MatchDataset
from model import MatchClassifier

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def main(args):
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    data = json.loads(args.test_file.read_text())
    dataset = MatchDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.predict_collate_fn
    )

    model = MatchClassifier("hfl/chinese-roberta-wwm-ext").to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    result = []

    with torch.set_grad_enabled(False):
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description("[predict]")

            for data in tepoch:
                token = data["token"].to(args.device)

                pred = model(token)["label"]
                idx = pred.argmax()

                Id = data["id"][idx]
                relevant = data["context_index"][idx]

                paragraph = data["paragraph"][idx]

                result.append(
                    {
                        "id": Id,
                        "paragraph": paragraph,
                        "start_end": [None, None],
                        "relevant": relevant,
                    }
                )

    with open(args.cache_dir / args.pred_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file", type=Path, help="Path to the test file.", required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/QA",
    )
    parser.add_argument(
        "--ckpt_path", type=Path, help="Path to model checkpoint.", required=True
    )
    parser.add_argument("--pred_file", type=Path, default="match.json")

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
