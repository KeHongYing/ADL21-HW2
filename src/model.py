import pickle
from typing import Dict

import torch

from transformers import AutoModel, BertConfig


class QAClassifier(torch.nn.Module):
    def __init__(
        self, model_name: str = None, config: str = None, no_pretrained: bool = False
    ) -> None:
        super(QAClassifier, self).__init__()
        if model_name is not None:
            self.backbone = AutoModel.from_pretrained(model_name)
            if no_pretrained:
                print("no_pretrianed")
                self.backbone = AutoModel.from_config(self.backbone.config)
        else:
            self.backbone = AutoModel.from_config(
                pickle.load(open(config, "rb")) if config is not None else BertConfig()
            )
        self.fc = torch.nn.Linear(self.backbone.config.hidden_size, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.backbone(input_ids=batch)["last_hidden_state"]

        x = self.fc(x)
        return {"start_end": x}


class MatchClassifier(torch.nn.Module):
    def __init__(
        self, model_name: str = None, config: str = None, no_pretrained: bool = False
    ) -> None:
        super(MatchClassifier, self).__init__()
        if model_name is not None:
            self.backbone = AutoModel.from_pretrained(model_name)
            if no_pretrained:
                print("no_pretrianed")
                self.backbone = AutoModel.from_config(self.backbone.config)
        else:
            self.backbone = AutoModel.from_config(
                pickle.load(open(config, "rb")) if config is not None else BertConfig()
            )

        self.fc = torch.nn.Linear(self.backbone.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.backbone(input_ids=batch)["last_hidden_state"][:, 0]
        x = self.fc(x)
        x = self.sigmoid(x)

        return {"label": x.view(-1)}
