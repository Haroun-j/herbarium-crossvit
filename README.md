# Herbarium specimens and CrossViT

Deep Learning module project, CY Tech 2025-2026, supervised by Dr. Youcef Sklab.
Team: Amine Filali and Haroun Joudi. I wrote most of the implementation (data
pipeline, model, training and evaluation); the report and the slides were joint work.

The task is a binary classification of digitised herbarium sheets: does the specimen
have spines or not. A sheet carries a lot of visual noise (labels, rulers, colour
charts, barcodes), and for every specimen we had two aligned images: the full sheet
and a segmented version where only the plant is kept on a black background. The
question of the project was how much that segmentation helps a Vision Transformer,
and whether the segmentation mask can be used to steer the attention of the model.

The model is a CrossViT (Chen et al., 2021): two ViT encoders, one with 12x12 patches
(small branch, width 192) and one with 16x16 patches (large branch, width 384), 8 blocks
and 8 heads each, joined by a cross-attention fusion. The brief defined five objectives:

- O1: compare four input configurations with the same hyper-parameters. A: full sheet
  only. B: segmented image only. C1: full sheet in the small branch and segmented image
  in the large one. C2: the other way round.
- O2: a variant where both branches have the same patch size and width (16x16, 384),
  separate weights, one branch per image.
- O3: a weight per patch from the plant density r_p of the segmented image,
  w_p = f(r_p), shared with the co-located patch of the full sheet.
- O4: interpretability. Attention rollout to the [CLS] token and the IoU between the
  thresholded rollout map and the plant mask.
- O5: an IoU term in the loss, so that the model is penalised when it looks at the
  background instead of the plant.

## Repository

```
src/
  api_client.py        access to the private image server (credentials from .env)
  data.py              metadata, stratified 70/15/15 split, two-image dataset with plant mask
  model.py             PatchEmbed, Attention, Block, VisionTransformer, CrossAttentionFusion, DualImageCrossViT
  patch_weights.py     O3: plant ratio per patch and w_p = f(r_p)
  interpretability.py  O4: attention rollout, rollout map, IoU with the plant mask
  losses.py            O5: cross-entropy + IoU penalty
  train.py             python -m src.train --modes A B C1 C2 O2 [--iou-loss]
  evaluate.py          python -m src.evaluate  (test split, confusion matrices)
  plot_histories.py    python -m src.plot_histories  (learning curves from results/)
notebooks/herbarium_crossvit.ipynb   the notebook that ran on the server, outputs of the training cells kept
results/history_*.json               per-epoch train/val metrics of the six runs
results/figures/                     learning curves, test confusion matrices, test F1 bar chart
docs/presentation_herbiers_crossvit.pdf   final slides (French)
```

The notebook is the one that ran on the training server. I kept the outputs of the
training and evaluation cells, since they are the only record of the runs, and cleared
the rest. `src/` is the same code split into modules, with two corrections described
in the last section. The images are course data and cannot be redistributed, so
nothing here lets you re-download them: the loader expects a private HTTP API whose
address and credentials go in `.env` (see `.env.example`).

## Results

898 specimens (407 without spines, 491 with), 224x224 inputs, batch 64, AdamW with
lr 1e-5 and cosine annealing, 5 epochs, no augmentation, one NVIDIA H100 80 GB,
PyTorch 2.9 / CUDA 12.8. The six runs share the same hyper-parameters. The O2 model has
29.7M parameters, A and B 14.6M. Validation metrics are the best of the five epochs on
the 180-image validation set; test metrics come from that same checkpoint on a
137-image test split (see the caveat below). F1 is weighted by class support.

| config | inputs | val acc | val F1 | test acc | test F1 |
|---|---|---|---|---|---|
| A | full sheet, one large branch | 70.6 % | 0.695 | 65.7 % | 0.649 |
| B | segmented image, one large branch | 77.2 % | 0.769 | 73.0 % | 0.724 |
| C1 | full -> small branch, segmented -> large branch | 68.9 % | 0.676 | 64.2 % | 0.630 |
| C2 | full -> large branch, segmented -> small branch | 71.1 % | 0.696 | 65.0 % | 0.634 |
| O2 | both images, two large branches | 80.0 % | 0.798 | 70.8 % | 0.705 |
| O2 + IoU loss (O5) | same as O2 | 80.6 % | 0.804 | 72.3 % | 0.721 |

Validation columns: `results/history_*.json`. Test columns: last cell of the notebook
(classification reports and confusion matrices on the 137 test images).

What I take from it: the segmented image alone (B) beats the full sheet alone (A) by
about 7 points, so the labels and rulers really do hurt. The mixed-resolution
configurations (C1, C2) do no better than A. The symmetric two-branch model (O2) is the
best on validation and the one I would keep. The run with the IoU loss looks a little
better than O2, but the gap is within run-to-run noise and, for the reason given below,
it is not an IoU-regularised run at all.

![learning curves](results/figures/learning_curves.png)

## Running it

```
pip install -r requirements.txt
cp .env.example .env                        # server address and credentials
python -m src.train --modes A B C1 C2 O2 --epochs 5
python -m src.train --modes O2 --iou-loss
python -m src.evaluate                      # test split: reports, confusion matrices, test_metrics.json
python -m src.plot_histories                # results/figures/learning_curves.png
```

Training needs a GPU (a few minutes per configuration on an H100) and access to the
data, so `plot_histories` is the only command that runs out of the box; it is the one
that produced the figure above. The notebook does the same steps in one place.

## Limitations and what I would do differently

- Too little data for a 30M-parameter transformer trained from scratch: 718 training
  images, 5 epochs, training accuracy around 70 %. Pre-trained ViT weights, more epochs
  and paired augmentation (the same flip or crop applied to the sheet, the segmented
  image and the mask) are the obvious next steps.
- Test split. The notebook first makes a stratified 70/15/15 split, but the training
  cell then pools everything again and draws a random 80/20 train/validation split, as
  the brief asked, so the 137 "test" images overlap with the training images. The test
  columns above are therefore not from a clean held-out set, which is why I present the
  validation columns as the main result. `src/train.py` trains on the 70 % split only
  and `src/evaluate.py` uses the 15 % test split that training never sees.
- IoU term. In the notebook the IoU between the rollout and the mask was always 0.0,
  including for the O5 run. Reading the code again while splitting it into modules, I
  found two reasons: the masks stayed on the CPU while the rollout was on the GPU, and
  the resulting exception was swallowed by a bare `except`; and the IoU was computed on
  a thresholded map with `.item()`, so it had no gradient anyway. `src/losses.py` now
  uses a soft IoU on the continuous rollout map, with the masks on the right device. I
  have not re-run the trainings with it, so the last row of the table is in practice a
  second O2 run.
- O3 is half done: `src/patch_weights.py` computes w_p = (eps + r_p)^gamma per patch,
  and the propagation to the full sheet is free because both images share the patch
  grid, but the weights are not used in the loss or in the token pooling.
- The `.pth` checkpoints were lost when the session on the server died, so the
  evaluation cell of the notebook cannot be re-run as is; the history files in
  `results/` were rebuilt from the training logs printed in the notebook.
- Reproducibility: the seed is fixed (42) but cuDNN benchmark mode is on, so two runs
  are not bit-identical.
