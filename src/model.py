from typing import Dict

import torch

from transformers import BertModel


class MatchClassifier(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super(MatchClassifier, self).__init__()
        self.backbone = BertModel.from_pretrained(model_name)
        self.fc = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.backbone(input_ids=batch)["last_hidden_state"][:, 0]
        x = self.fc(x)
        x = self.sigmoid(x)

        return {"label": x.view(-1)}
