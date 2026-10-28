"""CrossViT for pairs of herbarium images (full sheet + segmented plant).

Two ViT encoders, a "small" one with 12x12 patches and a "large" one with
16x16 patches as in CrossViT (Chen et al., 2021), and a cross-attention
fusion between their token sequences. `mode` selects which image feeds which
branch:

    A   full sheet only, one large branch                      (O1)
    B   segmented image only, one large branch                 (O1)
    C1  full sheet -> small branch, segmented -> large branch  (O1)
    C2  full sheet -> large branch, segmented -> small branch  (O1)
    O2  both images through two large branches of the same resolution

The [CLS] token of the query branch of the fusion goes to the classifier.
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


class Attention(nn.Module):
    """Multi-head self-attention that also returns the attention matrix (for the rollout)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x)), attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        y, attn = self.attn(self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, attn


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=8,
                 num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.pos_drop(torch.cat((cls, x), dim=1) + self.pos_embed)
        attentions = []
        for blk in self.blocks:
            x, attn = blk(x)
            attentions.append(attn)
        return self.norm(x), attentions


class CrossAttentionFusion(nn.Module):
    """The query tokens attend to the other branch's tokens, projected to the query dim."""

    def __init__(self, kv_dim, q_dim, num_heads=4):
        super().__init__()
        self.proj1 = nn.Linear(kv_dim, q_dim) if kv_dim != q_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(q_dim, num_heads, batch_first=True)

    def forward(self, kv_tokens, q_tokens):
        kv = self.proj1(kv_tokens)
        return self.cross_attn(q_tokens, kv, kv)


class DualImageCrossViT(nn.Module):
    MODES = ("A", "B", "C1", "C2", "O2")

    def __init__(self, num_classes=2, img_size=224, patch_size_small=12, patch_size_large=16,
                 embed_dim_small=192, embed_dim_large=384, depth=8, num_heads=8, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.2, attn_drop_rate=0.1, in_chans=3, mode="O2"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

        def branch(patch_size, dim):
            return VisionTransformer(img_size, patch_size, in_chans, dim, depth, num_heads,
                                     mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)

        small = (patch_size_small, embed_dim_small)
        large = (patch_size_large, embed_dim_large)
        if mode == "A":
            self.branch_nonseg = branch(*large)
        elif mode == "B":
            self.branch_seg = branch(*large)
        elif mode == "C1":
            self.branch_nonseg = branch(*small)
            self.branch_seg = branch(*large)
            self.fusion = CrossAttentionFusion(embed_dim_small, embed_dim_large, num_heads)
        elif mode == "C2":
            self.branch_nonseg = branch(*large)
            self.branch_seg = branch(*small)
            self.fusion = CrossAttentionFusion(embed_dim_small, embed_dim_large, num_heads)
        else:  # O2: same patch size, same width, separate weights
            self.branch_nonseg = branch(*large)
            self.branch_seg = branch(*large)
            self.fusion = CrossAttentionFusion(embed_dim_large, embed_dim_large, num_heads)
        self.head = nn.Linear(embed_dim_large, num_classes)

    def forward(self, img_nonseg, img_seg=None, return_attention=False):
        """Returns the logits, plus the per-layer attention matrices of the
        segmented branch (or of the only branch) when return_attention is set."""
        if self.mode == "A":
            tokens, attentions = self.branch_nonseg(img_nonseg)
        elif self.mode == "B":
            tokens, attentions = self.branch_seg(img_seg)
        else:
            tokens_nonseg, _ = self.branch_nonseg(img_nonseg)
            tokens_seg, attentions = self.branch_seg(img_seg)
            if self.mode == "C2":
                # the large branch (full sheet) is the query, the small one is projected onto it
                tokens, _ = self.fusion(tokens_seg, tokens_nonseg)
            else:
                # C1 and O2: the segmented branch is the query
                tokens, _ = self.fusion(tokens_nonseg, tokens_seg)
        logits = self.head(tokens[:, 0])
        return (logits, attentions) if return_attention else logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
