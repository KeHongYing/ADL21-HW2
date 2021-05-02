from typing import Dict

import torch

from transformers import AutoModel


class QAClassifier(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super(QAClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.fc = torch.nn.Linear(self.backbone.config.hidden_size, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.backbone(input_ids=batch)["last_hidden_state"]

        x = self.fc(x)
        return {"start_end": x}


class MatchClassifier(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super(MatchClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.fc = torch.nn.Linear(self.backbone.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.backbone(input_ids=batch)["last_hidden_state"][:, 0]
        x = self.fc(x)
        x = self.sigmoid(x)

        return {"label": x.view(-1)}
