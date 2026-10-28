"""Patch weights from plant density (objective O3).

For each patch p of the segmented image, r_p is the fraction of plant pixels
in the patch and the weight is w_p = f(r_p). The full sheet and the segmented
image share the same patch grid (same patch size, same alignment), so the
weight of a patch is also the weight of the co-located patch of the full
sheet: nothing else has to be propagated.

These functions were written for O3 but the weights did not make it into the
training loss before the deadline (see the README).
"""
import torch
import torch.nn.functional as F


def plant_ratio_per_patch(mask, patch_size):
    """mask: [H, W] or [B, H, W] with values in [0, 1].
    Returns r_p as [n_patches] or [B, n_patches], in row-major patch order."""
    squeeze = mask.dim() == 2
    m = mask.unsqueeze(0) if squeeze else mask
    m = (m > 0.5).float().unsqueeze(1)                                # [B, 1, H, W]
    r = F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size)    # [B, 1, h, w]
    r = r.flatten(1)
    return r.squeeze(0) if squeeze else r


def patch_weights(mask, patch_size=16, method="power", gamma=0.5, eps=0.1):
    """w_p = f(r_p).

    method: 'linear'      eps + r
            'power'       (eps + r) ** gamma, gamma in [0.5, 2]
            'normalized'  eps + r, divided by its mean so the weights of an image average to 1
    """
    r = plant_ratio_per_patch(mask, patch_size)
    if method == "linear":
        return eps + r
    if method == "power":
        return (eps + r) ** gamma
    if method == "normalized":
        w = eps + r
        return w / w.mean(dim=-1, keepdim=True)
    raise ValueError(f"unknown method {method!r}")
