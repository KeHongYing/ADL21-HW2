import random
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class QADataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def collate_fn(self, samples: List[Dict]) -> Dict:
        tokens = []
        start = []
        end = []
        index = []
        label = []

        for s in samples:
            for idx in [
                random.choice(s["relevant_index"]),
                random.choice(s["irrelevant_index"]),
            ]:
                token = self.construct(
                    s["question"], s["paragraph"][idx], s["start_end"][idx]
                )

                index.append(token["index"])
                tokens.append(token["token"])
                start.append(token["start"])
                end.append(token["end"])
                label.append(int(start != -1 or end != -1))

        return {
            "token": torch.tensor(tokens, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "end": torch.tensor(end, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "label": label,
        }

    def construct(
        self, question: List[str], paragraph: List[str], start_end: List[List]
    ) -> Dict:
        index = [len(question) + 2, len(question) + 2 + len(paragraph)]

        token = ["[CLS]"] + question + ["SEP"] + paragraph
        padding_len = self.max_len - len(token)
        token += ["[PAD]"] * padding_len
        token = self.tokenizer.convert_tokens_to_ids(token)

        start = (
            [-100 for _ in range(len(question) + 2)]
            + [1 if start_end[0] == i else 0 for i in range(len(paragraph))]
            + [-100 for _ in range(padding_len)]
        )
        end = (
            [-100 for _ in range(len(question) + 2)]
            + [1 if start_end[1] == i else 0 for i in range(len(paragraph))]
            + [-100 for _ in range(padding_len)]
        )

        return {"index": index, "token": token, "start": start, "end": end}


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

    with open("cache/train.json", "r") as f:
        data = json.load(f)

    dataset = QADataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=32, collate_fn=dataset.collate_fn
    )

    for d in dataloader:
        print(d["token"])
        print(d["start"])
        print(d["end"])
        print(d["index"])
        print()
