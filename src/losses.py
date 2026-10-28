"""Cross-entropy plus an IoU penalty between the attention rollout and the plant mask (O5)."""
import torch
import torch.nn as nn

from .interpretability import rollout_to_map


class IoUWeightedLoss(nn.Module):
    """L = CE + lambda * (1 - soft IoU(rollout map, plant mask)).

    The notebook version thresholded the rollout map and called .item() on the
    IoU, which gave a plain float: the penalty shifted the loss value but had
    no gradient, so it could not change what the model attends to. Here the
    IoU is the soft version computed on the continuous map (min-max normalised
    per image, interpolated to the mask size), so it is differentiable. The
    thresholded IoU of objective O4 is kept for evaluation in
    interpretability.attention_iou.
    """

    def __init__(self, lambda_iou=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_iou = lambda_iou

    def forward(self, logits, labels, rollout=None, masks=None):
        loss_ce = self.ce(logits, labels)
        if rollout is None or masks is None:
            zero = torch.zeros((), device=logits.device)
            return loss_ce, loss_ce, zero

        masks = masks.to(logits.device)
        if masks.max() > 1:
            masks = masks / 255.0
        plant = (masks > 0.5).float().flatten(1)                    # [B, H*W]

        att = rollout_to_map(rollout, masks.shape[-2:]).flatten(1)  # [B, H*W]
        lo = att.min(dim=1, keepdim=True).values
        hi = att.max(dim=1, keepdim=True).values
        att = (att - lo) / (hi - lo + 1e-8)

        inter = (att * plant).sum(dim=1)
        union = (att + plant - att * plant).sum(dim=1)
        soft_iou = inter / (union + 1e-8)
        loss_iou = 1.0 - soft_iou.mean()

        return loss_ce + self.lambda_iou * loss_iou, loss_ce, loss_iou
