import random
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class MatchDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def train_collate_fn(self, samples: List[Dict]) -> Dict:
        tokens = []
        labels = []

        for s in samples:
            for text in s["paragraphs"][random.choice(s["incorrect"])]:
                token = ["[CLS]"] + s["question"] + ["SEP"] + text
                token += ["[PAD]"] * (self.max_len - len(token))
                tokens.append(self.tokenizer.convert_tokens_to_ids(token))
                labels.append(0)

            for idx, text in enumerate(s["paragraphs"][s["correct"]]):
                token = ["[CLS]"] + s["question"] + ["SEP"] + text
                token += ["[PAD]"] * (self.max_len - len(token))
                tokens.append(self.tokenizer.convert_tokens_to_ids(token))

                label = 0
                for ans in s["answers"]:
                    if idx * len(text) <= ans["start"] < (idx + 1) * len(text):
                        label = 1

                labels.append(label)

        index = [i for i in range(len(tokens))]
        random.shuffle(index)

        tokens = [tokens[i] for i in index]
        labels = [labels[i] for i in index]

        return {
            "token": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(labels, dtype=torch.float),
        }

    def val_collate_fn(self, samples: List[Dict]) -> Dict:
        tokens = []
        labels = []

        for s in samples:
            for idx in s["incorrect"]:
                for text in s["paragraphs"][idx]:
                    token = ["[CLS]"] + s["question"] + ["SEP"] + text
                    token += ["[PAD]"] * (self.max_len - len(token))
                    tokens.append(self.tokenizer.convert_tokens_to_ids(token))
                    labels.append(0)

            for idx, text in enumerate(s["paragraphs"][s["correct"]]):
                token = ["[CLS]"] + s["question"] + ["SEP"] + text
                token += ["[PAD]"] * (self.max_len - len(token))
                tokens.append(self.tokenizer.convert_tokens_to_ids(token))

                label = 0
                for ans in s["answers"]:
                    if idx * len(text) <= ans["start"] < (idx + 1) * len(text):
                        label = 1

                labels.append(label)

        index = [i for i in range(len(tokens))]
        random.shuffle(index)

        tokens = [tokens[i] for i in index]
        labels = [labels[i] for i in index]

        return {
            "token": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(labels, dtype=torch.float),
        }


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

    with open("cache/train.json", "r") as f:
        data = json.load(f)

    dataset = MatchDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=32, collate_fn=dataset.val_collate_fn
    )

    for d in dataloader:
        print(d["token"])
        print(d["label"])
        print()
