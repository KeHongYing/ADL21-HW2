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
        relevant = []
        paragraph = []
        Id = []

        for s in samples:
            token = self.construct(
                s["token"],
                s["start_end"],
            )

            index.append(token["index"])
            tokens.append(token["token"])
            start.append(token["start"])
            end.append(token["end"])
            relevant.append(s["relevant"])
            paragraph.append(s["paragraph"])
            Id.append(s["id"])

        return {
            "id": Id,
            "token": torch.tensor(tokens, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "end": torch.tensor(end, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "relevant": relevant,
            "paragraph": paragraph,
        }

    def construct(self, token: List[int], start_end: List[int]) -> Dict:
        question_len = token.index(self.tokenizer.sep_token_id) + 1
        index = [question_len, len(token)]

        padding_len = self.max_len - len(token)
        token += [self.tokenizer.pad_token_id for _ in range(padding_len)]

        start = start_end[0] + question_len
        end = start_end[1] + question_len

        return {"index": index, "token": token, "start": start, "end": end}


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

    def collate_fn(self, samples: List[Dict]) -> Dict:
        tokens = []
        label = []

        for s in samples:
            for idx, paragraph_idx in enumerate(
                [
                    random.choice(s["irrelevant_index"]),
                    random.choice(s["relevant_index"]),
                ]
            ):
                token = s["question"] + s["token"][paragraph_idx]
                padding_len = self.max_len - len(token)
                token += [self.tokenizer.pad_token_id] * padding_len

                tokens.append(token)
                label.append(idx & 1)

        shuffle_index = [i for i in range(len(tokens))]
        random.shuffle(shuffle_index)
        tokens = [tokens[idx] for idx in shuffle_index]
        label = [label[idx] for idx in shuffle_index]

        return {
            "token": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }

    def predict_collate_fn(self, samples):
        tokens = []
        label = []
        context_index = []
        Id = []
        paragraph = []

        for s in samples:
            for idx, paragraph_idx in enumerate(
                s["irrelevant_index"] + s["relevant_index"]
            ):
                token = s["question"] + s["token"][paragraph_idx]
                padding_len = self.max_len - len(token)
                token += [self.tokenizer.pad_token_id] * padding_len

                Id.append(s["id"])
                tokens.append(token)
                label.append(idx >= len(s["irrelevant_index"]))
                context_index.append(s["paragraph_index"][paragraph_idx])
                paragraph.append(s["raw_paragraph"][paragraph_idx])

        return {
            "id": Id,
            "token": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "context_index": context_index,
            "paragraph": paragraph,
        }


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("voidful/albert_chinese_large")

    with open("cache/train.json", "r") as f:
        data = json.load(f)

    dataset = MatchDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=32, collate_fn=dataset.predict_collate_fn
    )

    for d in dataloader:
        print(d["token"])
        print(d["context_index"])
        print()
