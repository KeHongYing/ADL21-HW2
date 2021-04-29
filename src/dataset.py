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

        for s in samples:
            token = self.construct(
                s["question"],
                s["paragraph"],
                s["start_end"],
            )

            index.append(token["index"])
            tokens.append(token["token"])
            start.append(token["start"])
            end.append(token["end"])
            relevant.append(s["relevant"])

        return {
            "token": torch.tensor(tokens, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "end": torch.tensor(end, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "relevant": relevant,
        }

    def construct(
        self, question: List[str], paragraph: List[str], start_end: List[int]
    ) -> Dict:
        index = [len(question) + 2, len(question) + 2 + len(paragraph)]

        token = ["[CLS]"] + question + ["[SEP]"] + paragraph
        padding_len = self.max_len - len(token)
        token += ["[PAD]" for _ in range(padding_len)]
        token = self.tokenizer.convert_tokens_to_ids(token)

        start = (
            [-100 for _ in range(len(question) + 2)]
            + [1 if (start_end[0] == i) else 0 for i in range(len(paragraph))]
            + [-100 for _ in range(padding_len)]
        )
        end = (
            [-100 for _ in range(len(question) + 2)]
            + [1 if (start_end[1] == i) else 0 for i in range(len(paragraph))]
            + [-100 for _ in range(padding_len)]
        )
        assert start_end[0] == torch.tensor(start[index[0] : index[1]]).argmax().item()
        assert start_end[1] == torch.tensor(end[index[0] : index[1]]).argmax().item()

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
                token = (
                    ["[CLS]"]
                    + s["question"]
                    + ["[SEP]"]
                    + s["paragraph"][paragraph_idx]
                )
                padding_len = self.max_len - len(token)
                token += ["[PAD]"] * padding_len
                token = self.tokenizer.convert_tokens_to_ids(token)

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
        question = []
        paragraph = []

        for s in samples:
            for idx, paragraph_idx in enumerate(
                s["irrelevant_index"] + s["relevant_index"]
            ):
                token = (
                    ["[CLS]"]
                    + s["question"]
                    + ["[SEP]"]
                    + s["paragraph"][paragraph_idx]
                )
                padding_len = self.max_len - len(token)
                token += ["[PAD]"] * padding_len
                token = self.tokenizer.convert_tokens_to_ids(token)

                Id.append(s["id"])
                tokens.append(token)
                label.append(idx >= len(s["irrelevant_index"]))
                context_index.append(s["paragraph_index"][paragraph_idx])
                question.append(s["question"])
                paragraph.append(s["paragraph"][paragraph_idx])

        return {
            "id": Id,
            "token": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "context_index": context_index,
            "question": question,
            "paragraph": paragraph,
        }


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

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
