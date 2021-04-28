import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    with open(args.data, "r") as f:
        data = json.load(f)
    with open(args.ground_truth, "r") as f:
        ground_truth = json.load(f)

    cnt = 0
    for d, g in tqdm(zip(data, ground_truth), desc="computing..."):
        assert d["id"] == g["id"]

        cnt += d["relevant"] == g["relevant"]

    logging.info(f"matching rate: {cnt / len(data)}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "data",
        type=Path,
        help="Path to the target data.",
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to the ground truth.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)