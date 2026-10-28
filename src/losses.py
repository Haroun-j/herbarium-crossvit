"""Cross-entropy plus an IoU penalty between the attention rollout and the plant mask (O5)."""
import torch.nn as nn

from .interpretability import batch_attention_iou


class IoUWeightedLoss(nn.Module):
    """L = CE + lambda * (1 - mean IoU(rollout, plant mask))."""

    def __init__(self, lambda_iou=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_iou = lambda_iou

    def forward(self, logits, labels, rollout=None, masks=None):
        loss_ce = self.ce(logits, labels)
        loss_iou = 0.0
        if rollout is not None and masks is not None:
            ious = batch_attention_iou(rollout, masks)
            loss_iou = 1.0 - sum(ious) / len(ious)
        return loss_ce + self.lambda_iou * loss_iou, loss_ce, loss_iou
