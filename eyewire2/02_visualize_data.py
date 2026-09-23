# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: morphognn
#     language: python
#     name: morphognn
# ---

# %% [markdown]
# # Check the preprocessed eyewire2 point clouds
#
# Reads the h5 files written by `01_preprocess_data.py`. No torch and no GPU,
# so it runs locally.
#
# Three things are worth actually checking, and each corresponds to a claim in
# `00_dataset_spec.md` that fails silently if it is wrong:
#
# * **Did the depth frame survive?** (§2.1) Soma z must *vary* across cells and
#   the per-class depth profiles must differ. If every soma sits at z = 0 the
#   coordinates were centered per cell and the primary celltype signal is gone.
# * **Did size survive?** (§2.1) Dendritic field diameter must still differ
#   several-fold between cells. The stock normalization is run alongside to show
#   what it costs.
# * **Do the augmentations behave?** (§4) Rigid in xy, untouched in z.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()
sys.path.insert(0, str(THIS_DIR))

from retina_dataset import augment_points, read_h5  # noqa: E402

# %% [markdown]
# #### Load everything `01` wrote

# %%
with open(THIS_DIR / 'config.json') as f:
    config = json.load(f)

DATA_CFG = config['data']
DATA_DIR = THIS_DIR / 'data'
NORMALIZE = DATA_CFG['normalize']

with open(DATA_DIR / 'label_map.json') as f:
    label_map = json.load(f)
CLASSES = label_map['classes']
LABEL_COLUMN = label_map['label_column']

df_cells = pd.read_csv(DATA_DIR / 'cell_meta.csv', index_col='cell_id')

splits = {}
for split in ('Train', 'Val', 'Test'):
    path = DATA_DIR / f'{split}Datasets.h5'
    if path.exists():
        splits[split] = read_h5(path)

points = np.concatenate([p for p, _, _ in splits.values()])
labels = np.concatenate([lab for _, lab, _ in splits.values()])
cell_ids = np.concatenate([c for _, _, c in splits.values()])

print(f'{len(points)} cells, {points.shape[1]} points each, {len(CLASSES)} classes')
print({split: len(lab) for split, (_, lab, _) in splits.items()})

# %% [markdown]
# #### Skeleton size vs. `n_points`
#
# Cells below `n_points` are sampled with replacement, so their cloud contains
# duplicates. A few is harmless; a large fraction means `n_points` is too high
# for this skeleton set and the model is being fed padded-out cells.

# %%
n_nodes = df_cells['n_nodes'].to_numpy()

plt.hist(n_nodes, bins=40)
plt.axvline(DATA_CFG['n_points'], color='k', ls='--', label='n_points')
plt.axvline(DATA_CFG['min_nodes'], color='r', ls=':', label='min_nodes (QC floor)')
plt.xlabel('# dendritic nodes')
plt.ylabel('# cells')
plt.legend()
plt.title('Skeleton size after axon removal')
plt.show()

print(f'nodes: min={n_nodes.min()}, median={int(np.median(n_nodes))}, max={n_nodes.max()}')
print(f"{int(df_cells['oversampled'].sum())} / {len(df_cells)} cells below n_points "
      f"({DATA_CFG['n_points']}) -- these contain duplicated points")

# %% [markdown]
# #### Did dendritic field size survive normalization?
#
# Left: the xy extent of each cell as stored. It must span several-fold, because
# alpha and midget types differ mostly by size and that is a feature the model
# is meant to use. Right: the same cells under the stock per-axis
# bounding-box normalization, which rescales every cell to its own extent --
# every bar lands on 2.0 and the size signal is gone. That is §2.1 of the spec,
# measured rather than asserted.

# %%
from SWC2H5PY import Normalization  # noqa: E402

xy_extent = np.ptp(points[:, :, :2], axis=1).max(axis=1)

# Undo the retina scaling first: the stock normalization has to be measured on
# microns, the units it would have seen, or it looks better than it is (it only
# rescales an axis whose max exceeds 1).
microns = points * [NORMALIZE['xy_scale'], NORMALIZE['xy_scale'], NORMALIZE['z_scale']]
microns[:, :, 2] += NORMALIZE['z_offset']
subset = microns[np.random.default_rng(0).choice(len(points), size=min(200, len(points)),
                                                 replace=False)]
xy_extent_stock = np.ptp(Normalization(subset)[:, :, :2], axis=1).max(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
axes[0].hist(xy_extent * NORMALIZE['xy_scale'], bins=40)
axes[0].set(xlabel='xy extent (µm)', ylabel='# cells', title='retina normalization')
axes[1].hist(xy_extent_stock, bins=40)
axes[1].set(xlabel='xy extent (normalized units)', title='stock normalization')
plt.tight_layout()
plt.show()

print(f'xy extent (µm): p10={np.percentile(xy_extent, 10) * NORMALIZE["xy_scale"]:.0f}, '
      f'median={np.median(xy_extent) * NORMALIZE["xy_scale"]:.0f}, '
      f'p90={np.percentile(xy_extent, 90) * NORMALIZE["xy_scale"]:.0f} '
      f'-- a {np.percentile(xy_extent, 90) / np.percentile(xy_extent, 10):.1f}x spread')
print(f'under stock normalization: {xy_extent_stock.min():.2f} .. {xy_extent_stock.max():.2f} '
      f'(every cell rescaled to its own extent)')

stock_soma_z = Normalization(subset)[:, 0, 2]
print(f'and soma z under stock normalization spans '
      f'{stock_soma_z.min():.2f} .. {stock_soma_z.max():.2f} in normalized units -- '
      f'per-cell bounding-box centering, so the shared depth frame is gone too')

# %% [markdown]
# #### Did the depth frame survive?
#
# x and y are soma-centered, z is not. So soma z must **vary** across cells --
# if it is 0 everywhere, the coordinates were centered per cell and the
# stratification signal is gone -- and the pooled node-z distribution should
# show IPL band structure rather than one smooth blob.

# %%
soma_z = points[:, 0, 2]
all_z = points[:, :, 2].ravel()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(soma_z * NORMALIZE['z_scale'] + NORMALIZE['z_offset'], bins=40)
axes[0].set(xlabel='soma z (µm)', ylabel='# cells', title='Soma depth across cells')
axes[1].hist(all_z * NORMALIZE['z_scale'] + NORMALIZE['z_offset'], bins=120)
axes[1].set(xlabel='node z (µm)', ylabel='# nodes', title='Pooled depth distribution')
plt.tight_layout()
plt.show()

if np.allclose(soma_z, soma_z[0]):
    print('WARNING: every soma sits at the same z -- the depth frame was destroyed. '
          'Check data.normalize.mode in config.json (00_dataset_spec.md section 2.1).')
else:
    print(f'soma z (µm): {np.min(soma_z) * NORMALIZE["z_scale"] + NORMALIZE["z_offset"]:.1f} '
          f'.. {np.max(soma_z) * NORMALIZE["z_scale"] + NORMALIZE["z_offset"]:.1f}')

# %% [markdown]
# #### Per-class depth profiles
#
# The real test of whether the depth frame is intact and informative: the
# z-histogram of each celltype's nodes, which is the classical stratification
# profile. Different types should peak at different depths. If these curves lie
# on top of each other, no amount of training will separate the types on depth.

# %%
n_show = min(8, len(CLASSES))
biggest = pd.Series(labels).value_counts().index[:n_show]
bins = np.linspace(np.percentile(all_z, 0.5), np.percentile(all_z, 99.5), 61)
centers = 0.5 * (bins[1:] + bins[:-1]) * NORMALIZE['z_scale'] + NORMALIZE['z_offset']

plt.figure(figsize=(7, 4))
for label in biggest:
    profile, _ = np.histogram(points[labels == label][:, :, 2].ravel(), bins=bins)
    plt.plot(centers, profile / profile.sum(), label=f'{CLASSES[label]} (n={int((labels == label).sum())})')
plt.xlabel('z (µm)')
plt.ylabel('fraction of nodes')
plt.title(f'Stratification profiles, {n_show} largest classes')
plt.legend(fontsize='x-small')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Example cells
#
# Top row: xy (dendritic field, soma-centered, drawn on a shared axis so
# relative size is visible). Bottom row: xz (stratification depth, *not*
# centered -- cells sit at different depths on purpose).

# %%
n_examples = min(5, len(points))
show = np.random.default_rng(1).choice(len(points), size=n_examples, replace=False)
xy_lim = np.abs(points[show][:, :, :2]).max() * 1.05
z_lim = np.abs(points[:, :, 2]).max() * 1.05

fig, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 6.5))
axes = np.atleast_2d(axes)
for col, i in enumerate(show):
    axes[0, col].scatter(points[i, :, 0], points[i, :, 1], s=0.5, lw=0)
    axes[0, col].set(xlim=(-xy_lim, xy_lim), ylim=(-xy_lim, xy_lim), aspect='equal')
    axes[0, col].set_title(f'{CLASSES[labels[i]]}\n{cell_ids[i]}', fontsize='x-small')
    axes[1, col].scatter(points[i, :, 0], points[i, :, 2], s=0.5, lw=0)
    axes[1, col].set(xlim=(-xy_lim, xy_lim), ylim=(-z_lim, z_lim))
axes[0, 0].set_ylabel('y')
axes[1, 0].set_ylabel('z (depth)')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Augmentation preview
#
# Two independent views of one cell, using the `augment` block of
# `config.json`. In xy the cloud should rotate (and sometimes mirror) rigidly;
# in xz it must stay put in depth.

# %%
cell = points[show[0]]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
for col, title in enumerate(['original', 'view 1', 'view 2']):
    view = cell if col == 0 else augment_points(cell, normalize=NORMALIZE,
                                                **DATA_CFG['augment'])
    axes[0, col].scatter(view[:, 0], view[:, 1], s=0.5, lw=0)
    axes[0, col].set(xlim=(-xy_lim, xy_lim), ylim=(-xy_lim, xy_lim), aspect='equal')
    axes[0, col].set_title(title)
    axes[1, col].scatter(view[:, 0], view[:, 2], s=0.5, lw=0)
    axes[1, col].set(xlim=(-xy_lim, xy_lim), ylim=(-z_lim, z_lim))
axes[0, 0].set_ylabel('y')
axes[1, 0].set_ylabel('z (depth)')
plt.tight_layout()
plt.show()

# %% [markdown]
# The same thing numerically: with scaling and noise switched off, the rotation
# must preserve every pairwise xy distance and leave z untouched. (The stock
# `ssl_neuron.utils.rotate_graph(axis='z')` fails this -- it zeroes the z row
# and column of a random 3D rotation, leaving a non-orthogonal xy block that
# randomly squashes the arbor.)

# %%
rigid = augment_points(cell, rotate_xy=True, mirror_xy=True, scale_xy=0.0,
                       jitter=(0, 0, 0), translate=(0, 0, 0), normalize=NORMALIZE)

sample = np.random.default_rng(0).choice(len(cell), size=min(500, len(cell)), replace=False)
before = np.linalg.norm(cell[sample, None, :2] - cell[None, sample, :2], axis=-1)
after = np.linalg.norm(rigid[sample, None, :2] - rigid[None, sample, :2], axis=-1)

print(f'max relative change in pairwise xy distance: '
      f'{np.abs(after - before).max() / max(before.max(), 1e-9):.2e}')
print(f'max change in z: {np.abs(rigid[:, 2] - cell[:, 2]).max():.2e}')

# %% [markdown]
# #### Class balance
#
# 20:235 is a wide spread, so `03` reports balanced accuracy alongside plain
# accuracy and `retina_dataset.PKSampler` draws classes uniformly.

# %%
counts = pd.Series(labels).value_counts().sort_index()
plt.figure(figsize=(8, 3.5))
plt.bar(range(len(counts)), counts.to_numpy())
plt.xticks(range(len(counts)), [CLASSES[i] for i in counts.index],
           rotation=90, fontsize='x-small')
plt.ylabel('# cells')
plt.title(f'{LABEL_COLUMN} distribution')
plt.tight_layout()
plt.show()

print(f'smallest class: {counts.min()}, largest: {counts.max()}')

# %%
