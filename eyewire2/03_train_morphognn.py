# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: MorphoGNN
#     language: python
#     name: morphognn
# ---

# %% [markdown]
# # Train MorphoGNN on eyewire2 retina point clouds
#
# Needs `torch` and, realistically, a GPU -- run this on the cluster. Run
# `01_preprocess_data.py` there first (it picks the cluster paths
# automatically) to populate `./data`.
#
# The model itself is the unmodified `MorphoGNN.MorphoGNN`, only with
# `num_classes` taken from `data/label_map.json`. What differs from the
# repo-root training loop (`MorphoGNN.py`'s `__main__`) is:
#
# * **the sampler** -- P classes x K samples per batch, so the batch-hard
#   triplet loss has real positives and negatives. Under random sampling at
#   batch size 8 over ~28 classes, nearly every anchor is alone in its class and
#   ends up as its own hardest positive (`00_dataset_spec.md` §2.3);
# * **the split** -- selection on `Val`, with `Test` untouched until `04`. The
#   root loop saves on `test_acc >= best`, i.e. it reports the set it selected
#   on;
# * **balanced accuracy** as the selection metric, because the classes are
#   imbalanced roughly 20:235.
#
# Two things to watch on a first run:
#
# * **Memory.** `n_points` is the knob, not parameter count: the fourth EdgeConv
#   layer materializes a `(B, N, 64, 960)` tensor. At `n_points=2048` that is
#   ~500 MB per sample. If this OOMs, halve `n_points` before shrinking
#   `classes_per_batch` -- shrinking P is what re-breaks the triplet loss.
# * **Stale data.** If `02_visualize_data.py` warned that every soma sits at the
#   same z, the depth signal was destroyed in preprocessing and nothing trained
#   here will recover it.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.optim.lr_scheduler import StepLR

try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()
sys.path.insert(0, str(THIS_DIR))

from MorphoGNN import DEVICE, MorphoGNN, TripletLoss  # noqa: E402
from retina_dataset import build_dataloaders  # noqa: E402

# %% [markdown]
# #### Config

# %%
with open(THIS_DIR / 'config.json') as f:
    config = json.load(f)

TRAIN_CFG = config['training']
DATA_DIR = THIS_DIR / 'data'
CKPT_DIR = THIS_DIR / TRAIN_CFG['ckpt_dir'].lstrip('./')
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / 'MorphoGNN_eyewire2.t7'

with open(DATA_DIR / 'label_map.json') as f:
    label_map = json.load(f)
CLASSES = label_map['classes']

print(f'device: {DEVICE}')
print(f'{len(CLASSES)} classes, n_points={config["data"]["n_points"]}')

# %% [markdown]
# #### Data

# %%
loaders = build_dataloaders(config, DATA_DIR)
train_loader, val_loader, test_loader = loaders['Train'], loaders['Val'], loaders['Test']

batch_size = (TRAIN_CFG['classes_per_batch'] * TRAIN_CFG['samples_per_class']
              if TRAIN_CFG.get('sampler', 'pk') == 'pk' else TRAIN_CFG['batch_size'])
print(f'sampler={TRAIN_CFG.get("sampler", "pk")}, batch_size={batch_size}, '
      f'{len(train_loader)} iterations per epoch')
print({split: len(loader.dataset) for split, loader in loaders.items()})

# %% [markdown]
# #### Model, optimizer, losses
#
# Same recipe as the root loop: Adam, `StepLR`, a hard floor on the learning
# rate, and cross-entropy plus a batch-hard triplet term on the L2-normalized
# 2048-d embedding, summed with weight `triplet_weight`.

# %%
model = MorphoGNN(num_classes=len(CLASSES)).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=TRAIN_CFG['lr'],
                 weight_decay=TRAIN_CFG['weight_decay'])
scheduler = StepLR(opt, step_size=TRAIN_CFG['step_size'], gamma=TRAIN_CFG['gamma'])
criterion_ce = nn.CrossEntropyLoss()
criterion_triplet = TripletLoss(margin=TRAIN_CFG['triplet_margin'])


# %%
def run_epoch(loader, train, desc):
    """ One pass. Returns mean losses and the concatenated true/predicted labels. """
    model.train() if train else model.eval()
    totals = dict(loss=0.0, triplet=0.0, ce=0.0)
    count = 0
    true, pred = [], []

    batches = tqdm.tqdm(loader, desc=desc)
    with torch.set_grad_enabled(train):
        for points, label in batches:
            points = points.to(DEVICE).permute(0, 2, 1)
            label = label.to(DEVICE)
            n = label.size(0)

            if train:
                opt.zero_grad()
            features, logits = model(points)
            triplet_loss, _ = criterion_triplet(features, label)
            ce_loss = criterion_ce(logits, label)
            loss = TRAIN_CFG['triplet_weight'] * triplet_loss + ce_loss
            if train:
                loss.backward()
                opt.step()

            count += n
            totals['loss'] += loss.item() * n
            totals['triplet'] += triplet_loss.item() * n
            totals['ce'] += ce_loss.item() * n
            true.append(label.cpu().numpy())
            pred.append(logits.max(dim=1)[1].detach().cpu().numpy())
    batches.close()

    return ({k: v / max(count, 1) for k, v in totals.items()},
            np.concatenate(true), np.concatenate(pred))


def report(tag, epoch, losses, true, pred):
    acc = metrics.accuracy_score(true, pred)
    avg_acc = metrics.balanced_accuracy_score(true, pred)
    print(f'{tag} {epoch}, loss: {losses["loss"]:.6f}, triplet: {losses["triplet"]:.6f}, '
          f'CE: {losses["ce"]:.6f}, acc: {acc:.4f}, balanced acc: {avg_acc:.4f}')
    return acc, avg_acc


# %% [markdown]
# #### Train
#
# The checkpoint is written whenever balanced accuracy on **val** improves.
# `Test` is not touched here at all.

# %%
best_val_avg_acc = 0.0
history = []

for epoch in range(TRAIN_CFG['epochs']):
    losses, true, pred = run_epoch(train_loader, train=True,
                                   desc=f'Epoch-{epoch} training')
    train_acc, train_avg_acc = report('Train', epoch, losses, true, pred)

    # Step the schedule, then hold the LR at the floor (as in the root loop).
    if opt.param_groups[0]['lr'] > TRAIN_CFG['min_lr']:
        scheduler.step()
    if opt.param_groups[0]['lr'] < TRAIN_CFG['min_lr']:
        for param_group in opt.param_groups:
            param_group['lr'] = TRAIN_CFG['min_lr']

    losses, true, pred = run_epoch(val_loader, train=False,
                                   desc=f'Epoch-{epoch} validation')
    val_acc, val_avg_acc = report('Val  ', epoch, losses, true, pred)

    history.append(dict(epoch=epoch, train_acc=train_acc, train_avg_acc=train_avg_acc,
                        val_acc=val_acc, val_avg_acc=val_avg_acc,
                        lr=opt.param_groups[0]['lr']))

    if val_avg_acc >= best_val_avg_acc:
        best_val_avg_acc = val_avg_acc
        torch.save(model.state_dict(), CKPT_PATH)
        print(f'  saved {CKPT_PATH.name} (best val balanced acc {best_val_avg_acc:.4f})')

print(f'best val balanced acc: {best_val_avg_acc:.4f}')

# %% [markdown]
# #### Learning curve
#
# The gap between train and val accuracy is the thing to watch: ~2800 cells
# over ~28 classes is small for a net with a 2048-d embedding, and the
# augmentations of §4 are the only thing holding it back.

# %%
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

df_history = pd.DataFrame(history).set_index('epoch')
df_history.to_csv(CKPT_DIR / 'history.csv')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
df_history[['train_acc', 'val_acc']].plot(ax=axes[0], title='accuracy')
df_history[['train_avg_acc', 'val_avg_acc']].plot(ax=axes[1], title='balanced accuracy')
for ax in axes:
    ax.set_xlabel('epoch')
plt.tight_layout()
plt.show()

# %%

# %%
