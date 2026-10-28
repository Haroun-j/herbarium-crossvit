"""Attention rollout and IoU between the attention and the plant mask (objective O4)."""
import torch
import torch.nn.functional as F


def attention_rollout(attentions):
    """Attention rollout (Abnar & Zuidema, 2020).

    attentions: one [B, heads, N, N] matrix per layer, as returned by the model.
    For each layer the heads are averaged, the identity is added for the
    residual path, rows are renormalised and the matrices are multiplied
    through the layers. Returns [B, N - 1]: the influence of each patch token
    on the [CLS] token.
    """
    if not attentions:
        raise ValueError("no attention matrices")
    batch, n = attentions[0].shape[0], attentions[0].shape[-1]
    eye = torch.eye(n, device=attentions[0].device)
    result = eye.unsqueeze(0).expand(batch, -1, -1)
    for attn in attentions:
        a = attn.mean(dim=1) + eye
        a = a / a.sum(dim=-1, keepdim=True)
        result = torch.matmul(a, result)
    return result[:, 0, 1:]


def rollout_to_map(rollout, size):
    """[B, n_patches] -> [B, H, W]: the patch grid interpolated to the image size."""
    b, n = rollout.shape
    g = int(round(n ** 0.5))
    grid = rollout.reshape(b, 1, g, g)
    return F.interpolate(grid, size=size, mode="bilinear", align_corners=False).squeeze(1)


def attention_iou(att_map, plant_mask, quantile=0.8):
    """IoU between the top (1 - quantile) of the attention map and the plant mask.

    att_map: [H, W] continuous; plant_mask: [H, W] in [0, 1] (or 0-255).
    The map is binarised at its 0.8 quantile, as asked in the project brief.
    """
    plant_mask = plant_mask.to(att_map.device)
    if plant_mask.max() > 1:
        plant_mask = plant_mask / 255.0
    mask = plant_mask > 0.5
    thr = torch.quantile(att_map.flatten(), quantile)
    att = att_map >= thr
    inter = (att & mask).sum().float()
    union = (att | mask).sum().float()
    return (inter / (union + 1e-8)).item()


def batch_attention_iou(rollout, masks, quantile=0.8):
    """One IoU value per image of the batch. rollout: [B, n_patches], masks: [B, H, W]."""
    maps = rollout_to_map(rollout, masks.shape[-2:])
    return [attention_iou(maps[i], masks[i], quantile) for i in range(maps.shape[0])]


def overlay(image_tensor, att_map, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Undo the normalisation and return (image, heatmap) as arrays for plt.imshow."""
    img = image_tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    heat = att_map.detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return img, heat
