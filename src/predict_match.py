import json
import pickle
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
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    if args.QA_tokenizer is None:
        QA_tokenizer = tokenizer
    else:
        QA_tokenizer = pickle.load(open(args.QA_tokenizer, "rb"))

    data = json.loads(args.test_file.read_text())
    dataset = MatchDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.predict_collate_fn
    )

    model = MatchClassifier(config=args.config).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path, map_location=args.device)
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
                question = data["question"][0]
                token = (
                    QA_tokenizer.encode(question) + QA_tokenizer.encode(paragraph)[1:-1]
                )

                result.append(
                    {
                        "id": Id,
                        "paragraph": paragraph,
                        "start_end": [-1000, -1000],
                        "token": token,
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
    parser.add_argument(
        "--config",
        help="bert config",
        type=Path,
        required=True,
    )
    parser.add_argument("--tokenizer", help="tokenizer path", type=Path, required=True)
    parser.add_argument(
        "--QA_tokenizer", help="QA tokenizer path", type=Path, default=None
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    main(args)
