from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

class ArcFaceLoss(nn.Module):
    """
    ArcFace loss from `paper <https://arxiv.org/abs/1801.07698>`_.
    It contains projection size of ``num_features x num_classes`` inside itself. Please make sure that class labels
    started with 0 and ended as ``num_classes`` - 1.
    """

    criterion_name = "arcface"  # for better logging

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        m: float = 0.5,
        s: float = 64,
        label2category: Optional[Dict[Any, Any]] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            in_features: Input feature size
            num_classes: Number of classes in train set
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
            label2category: Optional, mapping from label to its category.
            reduction: CrossEntropyLoss reduction
        """
        super(ArcFaceLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.num_classes = num_classes
        if label2category is not None:
            mapper = {l: i for i, l in enumerate(sorted(list(set(label2category.values()))))}
            label2category = {k: mapper[v] for k, v in label2category.items()}
            self.label2category = torch.arange(num_classes).apply_(label2category.get)
        else:
            self.label2category = None
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.rescale = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = -self.cos_m
        self.mm = self.sin_m * m
        self.last_logs: Dict[str, float] = {}


    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.all(y < self.num_classes), "You should provide labels between 0 and num_classes - 1."

        cos = self.fc(x)

        self._log_accuracy_on_batch(cos, y)

        sin = torch.sqrt(1.0 - torch.pow(cos, 2))

        cos_w_margin = cos * self.cos_m - sin * self.sin_m
        cos_w_margin = torch.where(cos > self.th, cos_w_margin, cos - self.mm)

        ohe = F.one_hot(y, self.num_classes)
        logit = torch.where(ohe.bool(), cos_w_margin, cos) * self.rescale

        return self.criterion(logit, y), cos * self.rescale

    @torch.no_grad()
    def _log_accuracy_on_batch(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.last_logs["accuracy"] = torch.mean((y == torch.argmax(logits, 1)).to(torch.float32))

