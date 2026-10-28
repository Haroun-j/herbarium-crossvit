"""Hyper-parameters. The defaults are the values used for the runs in results/."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size_small: int = 12
    patch_size_large: int = 16
    embed_dim_small: int = 192
    embed_dim_large: int = 384
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.2
    attention_dropout: float = 0.1
    num_classes: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    lambda_iou: float = 0.1       # weight of the IoU penalty (objective O5)
    seed: int = 42
    num_workers: int = 0          # workers > 0 crashed the kernel with the in-memory cache
    results_dir: str = "results"
    cache_dir: str = "img_cache"
