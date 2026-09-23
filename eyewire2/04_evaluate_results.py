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
# # Evaluate MorphoGNN on eyewire2 retina point clouds
#
# Loads the checkpoint written by `03_train_morphognn.py` and reports, on the
# **test** split -- untouched during training and model selection:
#
# * classifier accuracy, balanced accuracy and a confusion matrix,
# * k-NN accuracy in the 2048-d embedding, which is the retrieval-style number
#   and the one comparable to the GraphDINO evaluation in `ssl_neuron`,
# * precision@k for retrieval, plus example queries,
# * a t-SNE of the embedding colored by celltype.
#
# Needs `torch`; a GPU helps but is not required. The embeddings are also
# written to `data/database.npy` as a `cell_id -> vector` dict, the eyewire2
# analog of what `retrieval.py --task=ExtractFeature` writes.
#
# > Retrieval here works on index arrays, not on `retrieval.ReturnKey` -- that
# > function recovers a filename by linearly scanning the database *comparing
# > vectors by value*, so two identical embeddings return the wrong cell.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
from sklearn import manifold

try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()
sys.path.insert(0, str(THIS_DIR))

from MorphoGNN import DEVICE, MorphoGNN  # noqa: E402
from retina_dataset import RetinaDataset  # noqa: E402

# %% [markdown]
# #### Config and checkpoint

# %%
with open(THIS_DIR / 'config.json') as f:
    config = json.load(f)

EVAL_CFG = config['evaluation']
DATA_DIR = THIS_DIR / 'data'
CKPT_PATH = THIS_DIR / config['training']['ckpt_dir'].lstrip('./') / 'MorphoGNN_eyewire2.t7'

with open(DATA_DIR / 'label_map.json') as f:
    label_map = json.load(f)
CLASSES = label_map['classes']

if not CKPT_PATH.exists():
    raise FileNotFoundError(f'No checkpoint at {CKPT_PATH}. Run 03_train_morphognn.py first.')

model = MorphoGNN(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print(f'Loaded {CKPT_PATH} ({len(CLASSES)} classes) onto {DEVICE}')

# %% [markdown]
# #### Embed every split
#
# No augmentation anywhere here, and file order is preserved so rows line up
# with `cell_meta.csv`. `forward` returns `(embedding, logits)`; the embedding
# is the 2048-d concatenation of max- and average-pooled features.

# %%
@torch.no_grad()
def embed(dataset, batch_size=8):
    features, logits = [], []
    for start in range(0, len(dataset), batch_size):
        batch = np.stack([dataset.points[i] for i in range(start, min(start + batch_size,
                                                                     len(dataset)))])
        points = torch.from_numpy(batch).float().to(DEVICE).permute(0, 2, 1)
        f, l = model(points)
        features.append(f.cpu().numpy())
        logits.append(l.cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


splits = {}
for split in ('Train', 'Val', 'Test'):
    dataset = RetinaDataset(DATA_DIR / f'{split}Datasets.h5')
    features, logits = embed(dataset)
    splits[split] = dict(features=features, logits=logits,
                         label=dataset.label, cell_id=dataset.cell_id)
    print(f'{split}: {features.shape}')

database = {cell_id: vec
            for split in splits.values()
            for cell_id, vec in zip(split['cell_id'], split['features'])}
np.save(DATA_DIR / 'database.npy', database)
print(f'{len(database)} embeddings -> {DATA_DIR / "database.npy"}')

# %% [markdown]
# #### Classifier performance on the test split

# %%
test = splits['Test']
pred = test['logits'].argmax(axis=1)

acc = metrics.accuracy_score(test['label'], pred)
avg_acc = metrics.balanced_accuracy_score(test['label'], pred)
print(f'test accuracy:          {acc:.4f}')
print(f'test balanced accuracy: {avg_acc:.4f}   (chance = {1 / len(CLASSES):.4f})')
print()
print(metrics.classification_report(test['label'], pred,
                                    labels=range(len(CLASSES)), target_names=CLASSES,
                                    zero_division=0))

# %%
confusion = metrics.confusion_matrix(test['label'], pred, labels=range(len(CLASSES)),
                                     normalize='true')

fig, ax = plt.subplots(figsize=(0.35 * len(CLASSES) + 4, 0.35 * len(CLASSES) + 3))
im = ax.imshow(confusion, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize='x-small')
ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize='x-small')
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_title('Test confusion matrix (row-normalized)')
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### k-NN accuracy in the embedding
#
# Train cells are the database, test cells the queries, cosine similarity as in
# `retrieval.py`. This measures the embedding rather than the classifier head,
# so it is the number to compare against a GraphDINO embedding evaluated the
# same way.

# %%
def cosine_similarity(query, database):
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    d = database / np.linalg.norm(database, axis=1, keepdims=True)
    return q @ d.T


train = splits['Train']
similarity = cosine_similarity(test['features'], train['features'])
ranking = np.argsort(-similarity, axis=1)

k = EVAL_CFG['knn_k']
neighbor_labels = train['label'][ranking[:, :k]]
knn_pred = np.array([np.bincount(row, minlength=len(CLASSES)).argmax()
                     for row in neighbor_labels])

print(f'{k}-NN accuracy:          {metrics.accuracy_score(test["label"], knn_pred):.4f}')
print(f'{k}-NN balanced accuracy: '
      f'{metrics.balanced_accuracy_score(test["label"], knn_pred):.4f}')

# %% [markdown]
# #### Retrieval precision@k
#
# Of the k most similar training cells to each test cell, what fraction share
# its celltype? Chance is the class's share of the database.

# %%
retrieval_k = EVAL_CFG['retrieval_k']
hits = train['label'][ranking[:, :retrieval_k]] == test['label'][:, None]
precision = hits.mean(axis=1)

per_class = pd.DataFrame({'label': test['label'], 'precision': precision})
per_class = per_class.groupby('label')['precision'].agg(['mean', 'size'])
per_class.index = [CLASSES[i] for i in per_class.index]
chance = pd.Series(train['label']).value_counts(normalize=True).sort_index()
per_class['chance'] = [chance.get(CLASSES.index(c), 0.0) for c in per_class.index]

print(f'precision@{retrieval_k}: {precision.mean():.4f} '
      f'(macro over classes: {per_class["mean"].mean():.4f})')
print(per_class.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(range(len(per_class)), per_class['mean'], label=f'precision@{retrieval_k}')
ax.plot(range(len(per_class)), per_class['chance'], 'k.', label='chance')
ax.set_xticks(range(len(per_class)), per_class.index, rotation=90, fontsize='x-small')
ax.set_ylabel(f'precision@{retrieval_k}')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Example queries
#
# One test cell per row, then its most similar training cells. Green titles are
# same-celltype hits, red are misses. xz views, because stratification depth is
# what should be driving the similarity.

# %%
train_points = RetinaDataset(DATA_DIR / 'TrainDatasets.h5').points
test_points = RetinaDataset(DATA_DIR / 'TestDatasets.h5').points

n_queries, n_hits = 4, 5
queries = np.random.default_rng(0).choice(len(test_points), size=n_queries, replace=False)

fig, axes = plt.subplots(n_queries, n_hits + 1, figsize=(2.2 * (n_hits + 1), 2.4 * n_queries))
axes = np.atleast_2d(axes)
for row, q in enumerate(queries):
    axes[row, 0].scatter(test_points[q][:, 0], test_points[q][:, 2], s=0.3, lw=0, c='k')
    axes[row, 0].set_title(f'query\n{CLASSES[test["label"][q]]}', fontsize='xx-small')
    for col, hit in enumerate(ranking[q, :n_hits], start=1):
        same = train['label'][hit] == test['label'][q]
        axes[row, col].scatter(train_points[hit][:, 0], train_points[hit][:, 2],
                               s=0.3, lw=0, c='tab:blue')
        axes[row, col].set_title(f'{CLASSES[train["label"][hit]]}\n{similarity[q, hit]:.3f}',
                                 fontsize='xx-small',
                                 color='tab:green' if same else 'tab:red')
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# #### t-SNE of the embedding
#
# All splits together, colored by celltype. What to look for is not tight
# clusters per type -- 28 RGC types are not 28 islands -- but whether cells that
# land together share a stratification depth. If they do, the embedding is
# picking up the feature the spec says it should.

# %%
features = np.concatenate([s['features'] for s in splits.values()])
labels = np.concatenate([s['label'] for s in splits.values()])

tsne = manifold.TSNE(n_components=2, init='pca', random_state=13,
                     perplexity=min(30, max(5, len(features) // 4)))
embedded = tsne.fit_transform(features)

fig, ax = plt.subplots(figsize=(8, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(CLASSES)))
for i, celltype in enumerate(CLASSES):
    mask = labels == i
    ax.scatter(embedded[mask, 0], embedded[mask, 1], s=8, color=colors[i], label=celltype)
ax.legend(fontsize='xx-small', ncol=2, loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('MorphoGNN embedding of eyewire2 RGCs')
plt.tight_layout()
plt.show()

# %% [markdown]
# Same t-SNE, colored by mean dendritic depth instead of celltype. If the
# embedding has learned stratification, this should vary smoothly across the
# map -- a stronger sanity check than the class colors, because it needs no
# labels.

# %%
points_all = np.concatenate([RetinaDataset(DATA_DIR / f'{s}Datasets.h5').points
                             for s in ('Train', 'Val', 'Test')])
mean_depth = (points_all[:, :, 2].mean(axis=1) * config['data']['normalize']['z_scale']
              + config['data']['normalize']['z_offset'])

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(embedded[:, 0], embedded[:, 1], s=8, c=mean_depth, cmap='coolwarm')
fig.colorbar(scatter, ax=ax, label='mean dendritic z (µm)')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Same embedding, colored by stratification depth')
plt.tight_layout()
plt.show()

# %%
