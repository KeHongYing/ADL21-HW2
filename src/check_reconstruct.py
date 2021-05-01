import json
import logging
from argparse import ArgumentParser, Namespace

from transformers import BertTokenizer
from tqdm.auto import tqdm

from predict_QA import reconstruct

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    with open(args.data, "r") as f:
        data = json.load(f)
    with open(args.raw_data, "r") as f:
        raw_data = json.load(f)

    cnt = 0
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    raw_index = 0
    for d in tqdm(data, desc="reconstring..."):
        while d["id"] != raw_data[raw_index]["id"]:
            raw_index += 1

        head = d["token"].index(tokenizer.sep_token_id) + 1

        start, end = d["start_end"][0], d["start_end"][1]
        token = d["token"][head:]
        answer = reconstruct(
            d["paragraph"][
                start : -(len(token) - end - 2)
                if (len(token) - end - 2) > 0
                else len(d["paragraph"])
            ],
            token[start : end + 1],
            tokenizer,
        )

        flag = False
        for raw_answer in raw_data[raw_index]["answers"]:
            raw_answer = raw_answer["text"]
            # print(raw_answer)

            if answer == raw_answer:
                flag = True
        if flag is False:
            print("paragraph:", d["paragraph"][start : -(len(token) - end - 1)])
            print("token:", "".join(tokenizer.decode(token)))
            print("answer:", answer)
            print("raw answer:", raw_answer)

        cnt += flag

    print(f"reconver rate: {cnt}/{len(data)}({cnt / len(data)})")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("data", help="data for testing")
    parser.add_argument("raw_data", help="non preprocessing data")

    parser.add_argument(
        "--backbone", help="Bert backbone", default="voidful/albert_chinese_large"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
