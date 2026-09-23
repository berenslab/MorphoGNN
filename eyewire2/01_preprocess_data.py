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
# # Preprocess eyewire2 retina skeletons for MorphoGNN
#
# Converts the `.swc` skeletons produced by `skeliner` into the `data`/`label`
# h5 pair MorphoGNN trains on -- but along the retina coordinate contract, not
# the stock one. See `00_dataset_spec.md` in this folder for why the stock
# recipe (`SWC2H5PY.GenerateNeuronDataset`) is wrong for this dataset in three
# independent, silent ways.
#
# Per cell (`preprocessing.preprocess_cell`):
#
# 1. parse the SWC, soma first,
# 2. drop axon nodes -- unreliable in these reconstructions, and their extent
#    reflects how far the tracing got rather than the cell's biology,
# 3. QC on dendritic node count,
# 4. sample **exactly** `n_points` nodes (the stock path pads short cells with
#    zeros and silently splits long ones into two half-cells),
# 5. center x and y on the soma and rescale by *global* constants; z is offset
#    and scaled by global constants and never centered per cell, so
#    stratification depth and dendritic field size both survive.
#
# Then, across cells: build the label map (dropping the long tail of celltypes
# too small to split), make a stratified train/val/test split, and write one h5
# per split plus a `cell_meta.csv` sidecar.
#
# Needs numpy/pandas/pyarrow/h5py/matplotlib -- no torch, no GPU -- so it runs
# locally. Re-run it on the cluster (it picks the cluster paths automatically)
# before training there.

# %%
import json
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()
sys.path.insert(0, str(THIS_DIR))

from preprocessing import SkeletonQCError, preprocess_cell  # noqa: E402

# %% [markdown]
# #### Config
#
# `config.json` lists the cluster path first and the local one second for both
# inputs; the first that exists wins, so the same file works in both places.

# %%
with open(THIS_DIR / 'config.json') as f:
    config = json.load(f)

DATA_CFG = config['data']
OUT_DIR = THIS_DIR / 'data'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def first_existing(candidates, what):
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(f'None of the configured {what} paths exist: {candidates}')


SWC_DIR = first_existing(config['paths']['swc_dir'], 'swc_dir')
DF_PATH = first_existing(config['paths']['dataframe'], 'dataframe')
print(f'skeletons:  {SWC_DIR}')
print(f'dataframe:  {DF_PATH}')
print(f'n_points={DATA_CFG["n_points"]}, normalize={DATA_CFG["normalize"]}')

# %% [markdown]
# #### Select cells
#
# Unlike the self-supervised GraphDINO pipeline, MorphoGNN is a classifier: an
# unlabeled cell is of no use to it. So on top of the RGC selection
# (`cellclass_final == 'RGC'`, masked by `valid_cellclass_final` and
# `valid_status`), a cell must carry a valid `celltype_final`.

# %%
META_COLUMNS = [
    'cellclass_final', 'valid_cellclass_final', 'valid_status',
    'celltype_final', 'valid_celltype_final', 'celltype_final_decision',
    'soma_x_um', 'soma_y_um', 'soma_z_um',
]


def load_metadata(path, columns=META_COLUMNS):
    """ Load cell metadata (the same dataframe `plot_RGCs.ipynb` uses). Falls
    back to a full read if this dataframe version is missing some column, so a
    renamed column shows up as a loud message instead of a crash. """
    try:
        return pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as err:
        print(f'Column-limited read failed ({err}); falling back to a full read.')
        df = pd.read_parquet(path)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f'Missing columns in this dataframe version: {missing}')
        return df[[c for c in columns if c in df.columns]]


def select_labeled_cells(df, label_column, cellclass='RGC'):
    """ Confidently-labeled cells of one cellclass. `cellclass=None` keeps every
    class, which is what makes the coarse AC/BC/RGC task -- and a local smoke
    test on the handful of skeletons that live outside the cluster -- possible.
    """
    selected = pd.Series(True, index=df.index)
    if cellclass is not None:
        selected &= df['cellclass_final'] == cellclass
    for flag in ('valid_cellclass_final', 'valid_status', f'valid_{label_column}'):
        if flag in df.columns:
            selected &= df[flag].fillna(False).astype(bool)
    return df[selected & df[label_column].notna()]


LABEL_COLUMN = DATA_CFG['label_column']
CELLCLASS = DATA_CFG.get('cellclass', 'RGC')
df_meta = select_labeled_cells(load_metadata(DF_PATH), LABEL_COLUMN, CELLCLASS)
cell_id_filter = {str(i) for i in df_meta.index}
print(f'{len(cell_id_filter)} labeled {CELLCLASS or "cells"} selected from {DF_PATH.name}')
print(f'{df_meta[LABEL_COLUMN].nunique()} distinct {LABEL_COLUMN} values')

# %% [markdown]
# #### Find skeletons
#
# Prefer the `_cal` variant of each skeleton when present (better radius
# estimates, otherwise identical). Radius is not a model feature here -- the
# point cloud is xyz only -- but the `_cal` files are the canonical ones.

# %%
def find_swc_files(swc_dir):
    files = {}
    for path in sorted(swc_dir.rglob('*.swc')):
        if path.stem.endswith('_cal'):
            continue
        cal_path = path.with_name(f'{path.stem}_cal.swc')
        files[path.stem] = cal_path if cal_path.exists() else path
    return files


swc_files = {cell_id: path for cell_id, path in find_swc_files(SWC_DIR).items()
             if cell_id in cell_id_filter}
print(f'Found {len(swc_files)} / {len(cell_id_filter)} labeled skeletons under {SWC_DIR}')
assert swc_files, 'None of the selected cells have a skeleton under SWC_DIR'

# %% [markdown]
# #### Preprocess each cell
#
# Cells that fail QC are skipped and tallied by reason rather than silently
# dropped. `oversampled` counts cells with fewer nodes than `n_points`, whose
# point cloud therefore contains duplicates -- a handful is fine, a large
# fraction means `n_points` is set too high for this skeleton set.
#
# The one thing QC does *not* catch is an arbor clipped by the volume boundary:
# it keeps all its nodes, so it passes the floor, and since dendritic field size
# is a preserved feature it then presents as a genuinely smaller cell. There is
# no truncation flag in the dataframe to filter on -- see `00_dataset_spec.md`
# section 7.5 for the `hull_points` handle that would.

# %%
points_by_cell = {}
rows = []
qc_failures = Counter()

for cell_id, swc_path in tqdm(sorted(swc_files.items())[:500]):
    try:
        points, soma_xyz, info = preprocess_cell(
            swc_path,
            n_points=DATA_CFG['n_points'],
            min_nodes=DATA_CFG['min_nodes'],
            normalize=DATA_CFG['normalize'],
            seed=int(cell_id) % (2 ** 32),  # deterministic per cell
        )
    except SkeletonQCError as err:
        qc_failures[re.sub(r'\d+', 'N', str(err))] += 1
        continue

    points_by_cell[cell_id] = points
    rows.append({
        'cell_id': cell_id,
        'soma_x': soma_xyz[0], 'soma_y': soma_xyz[1], 'soma_z': soma_xyz[2],
        **info,
    })

print(f'Preprocessed {len(rows)} / {len(swc_files)} skeletons')
for reason, count in qc_failures.most_common():
    print(f'  dropped {count:5d}: {reason}')

df_cells = pd.DataFrame(rows).set_index('cell_id')
df_cells = df_cells.join(df_meta.rename(index=str), how='left')
print(df_cells[['n_nodes_raw', 'n_nodes', 'n_axon_nodes']].describe().round(1))
print(f"{int(df_cells['oversampled'].sum())} cells have fewer than "
      f"{DATA_CFG['n_points']} nodes and contain duplicated points")

# %% [markdown]
# #### Label map
#
# The eyewire2 analog of `SWC2H5PY.LABEL7`, but derived from the data rather
# than hardcoded: celltypes with fewer than `min_cells_per_class` cells cannot
# be split three ways, let alone learned, so they are dropped here and the
# remaining types are numbered alphabetically.

# %%
counts = df_cells[LABEL_COLUMN].value_counts()
keep = sorted(counts[counts >= DATA_CFG['min_cells_per_class']].index)
dropped = counts[counts < DATA_CFG['min_cells_per_class']]

label_map = {celltype: i for i, celltype in enumerate(keep)}
df_cells = df_cells[df_cells[LABEL_COLUMN].isin(label_map)].copy()
df_cells['label'] = df_cells[LABEL_COLUMN].map(label_map).astype(int)

print(f'{len(label_map)} classes kept ({len(df_cells)} cells); '
      f'{len(dropped)} dropped below {DATA_CFG["min_cells_per_class"]} cells '
      f'({int(dropped.sum())} cells)')
if len(dropped):
    print('  dropped: ' + ', '.join(f'{t} ({n})' for t, n in dropped.items()))
print(counts[keep].to_string())

with open(OUT_DIR / 'label_map.json', 'w') as f:
    json.dump({'label_column': LABEL_COLUMN, 'classes': keep}, f, indent=2)

# %% [markdown]
# #### Stratified train / val / test split
#
# 70 / 15 / 15, per class, seeded. Selection happens on **val** and the number
# reported in `04_evaluate_results.py` is **test** -- unlike the upstream loop,
# which saves on `test_acc >= best` and so reports the set it selected on.

# %%
rng = np.random.default_rng(DATA_CFG['seed'])
val_fraction, test_fraction = DATA_CFG['val_fraction'], DATA_CFG['test_fraction']

split = pd.Series('Train', index=df_cells.index, name='split')
for label, group in df_cells.groupby('label'):
    ids = group.index.to_numpy()
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    n_test = max(1, int(round(len(ids) * test_fraction)))
    split.loc[ids[:n_val]] = 'Val'
    split.loc[ids[n_val:n_val + n_test]] = 'Test'

df_cells['split'] = split
print(df_cells['split'].value_counts().to_string())
print(pd.crosstab(df_cells[LABEL_COLUMN], df_cells['split']).to_string())

# %% [markdown]
# #### Write the h5 files
#
# The upstream layout (`data` float32 (M, n_points, 3), `label` uint8 (M, 1))
# plus a `cell_id` column, so the root `dataset.DataSet` can read these too --
# with `norm=False`, which is **not** optional: `norm=True` re-applies the
# stock per-axis normalization on top of the retina one and undoes everything
# this script did.

# %%
for split_name, group in df_cells.groupby('split'):
    ids = group.index.to_numpy()
    data = np.stack([points_by_cell[c] for c in ids]).astype(np.float32)
    label = group['label'].to_numpy().reshape(-1, 1).astype(np.uint8)

    path = OUT_DIR / f'{split_name}Datasets.h5'
    with h5py.File(path, 'w') as f:
        f['data'] = data
        f['label'] = label
        f['cell_id'] = np.array(ids, dtype='S32')
    print(f'{path.name}: data {data.shape}, label {label.shape}')

df_cells.to_csv(OUT_DIR / 'cell_meta.csv')
print(f'{len(df_cells)} cells written to {OUT_DIR / "cell_meta.csv"}')

# %% [markdown]
# #### Sanity-check one cell
#
# The soma should sit at x = y = 0 while z should **not** be centered: the z
# range below is the cell's stratification depth in the shared IPL frame,
# divided by `z_scale`. If every cell's soma z came out at 0, the retina
# normalization was bypassed and the depth signal is gone.

# %%
sample_id = df_cells.index[0]
sample = points_by_cell[sample_id]
print(f'{sample_id} ({df_cells.loc[sample_id, LABEL_COLUMN]}): '
      f'soma at {sample[0].round(3)}, '
      f'z range {sample[:, 2].min():.3f} .. {sample[:, 2].max():.3f}')

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].scatter(sample[:, 0], sample[:, 1], s=1, lw=0)
axes[0].set(xlabel='x', ylabel='y', title='dendritic field', aspect='equal')
axes[1].scatter(sample[:, 0], sample[:, 2], s=1, lw=0)
axes[1].set(xlabel='x', ylabel='z (depth)', title='stratification')
plt.tight_layout()
plt.show()

# %%
