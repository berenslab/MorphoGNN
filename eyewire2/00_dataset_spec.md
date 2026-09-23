# MorphoGNN on eyewire2 RGCs — dataset & training spec

Status: written 2026-09-23. Everything marked **Assumed** is a recommendation
adopted without explicit sign-off — cheap to revisit, but write down the reason
if you change it.

This folder is the eyewire2 counterpart of the upstream `neuron7` pipeline at
the repo root. It exists because MorphoGNN's stock recipe — directory-per-class
labels, 6000 zero-padded points, per-axis bounding-box normalization — is wrong
for retinal ganglion cells in three separate ways, each of which fails
*silently*.

The biological reasoning about this dataset (why only dendrites are
trustworthy, why depth is the load-bearing axis, why xy rotation invariance is
imposed on purpose) is not repeated here. It lives in
`ssl_neuron/ssl_neuron/eyewire2/00_dataset_spec.md`, sections 1–4, and this
pipeline follows it. What follows is only what changes when the model is a
*supervised point-cloud classifier* rather than a self-supervised graph
transformer.

---

## 1. What MorphoGNN needs that GraphDINO did not

| | GraphDINO (`ssl_neuron`) | MorphoGNN (here) |
| --- | --- | --- |
| Supervision | none — all RGCs usable | **labels required** — only typed cells |
| Input | graph: `features.npy` + `neighbors.pkl` | point cloud: N×3, no edges |
| Cells per class | irrelevant | must be enough to train *and* split |
| Per-cell node count | variable, subsampled per view | **fixed N, identical for every cell** |

Consequences:

- **The training set is the labeled subset**, not all RGCs. Selection is
  `cellclass_final == 'RGC'` masked by `valid_cellclass_final` and
  `valid_status` (as in the GraphDINO pipeline), *plus* `celltype_final`
  present and `valid_celltype_final` true. On the 2026-09-16 dataframe that is
  2915 cells over 39 celltypes.
- Both halves of that are config, not code: `data.cellclass` (`null` keeps every
  class) and `data.label_column`. Setting them to `null` / `cellclass_final`
  turns the pipeline into the coarse AC/BC/RGC task, which is also the only
  version that can be exercised on a laptop — the ~110 skeletons mirrored
  outside the cluster hold at most one cell per RGC celltype, but three
  cellclasses with 10–62 cells each.
- **The long tail has to go.** 39 types, but the smallest have a single cell —
  unsplittable and unlearnable. `min_cells_per_class` (default 20, **Assumed**)
  keeps 28 classes / 2816 cells. Everything dropped is tallied, never silent.
- **Edges are thrown away.** MorphoGNN rebuilds a kNN graph in feature space at
  every layer, so `neighbors.pkl` has no use here and connectivity repair is
  pointless: a disconnected fragment is just more points. The
  `reconnect_components` machinery from the GraphDINO pipeline is deliberately
  **not** duplicated into this folder.
- Axon removal still applies, and is simpler: drop every node typed 2. With no
  graph to keep intact, the hole this leaves in a mislabeled dendrite costs
  nothing. (`ssl_neuron.data.data_utils.remove_axon` is also a plain type mask,
  so the two pipelines drop exactly the same nodes.)

---

## 2. The three silent failures of the stock recipe

### 2.1 `SWC2H5PY.Normalization` destroys the depth frame

The stock normalization centers each cloud on its **bounding-box midpoint** and
then divides **each axis by its own max**. On a retinal ganglion cell that does
two fatal things at once:

1. Centering z per cell discards *where in the IPL the arbor sits*, which is the
   primary celltype signal (`ssl_neuron` spec §1.2). This is the same bug as
   that spec's §6.2, in a different disguise.
2. Scaling each axis to its own extent discards *dendritic field diameter* and
   *stratification thickness* — alpha types differ from midget types mostly by
   size (`ssl_neuron` spec §1.4). Two cells 5× apart in field diameter come out
   the same size.

`preprocessing.normalize_retina` replaces it:

| Axis | Center | Scale |
| --- | --- | --- |
| x, y | per cell, on the **soma** | global constant `xy_scale` (300 µm) |
| z | global constant `z_offset` (25 µm) | global constant `z_scale` |

Every scale is a *dataset-level constant*, never a per-cell statistic — that is
the whole point. Relative size survives, so a 900 µm alpha arbor still comes out
3× wider than a 300 µm midget one, and two cells stratifying at different
depths still land at different z.

**`z_scale` defaults to `xy_scale`, i.e. the transform is isotropic.** That
preserves the true geometry, which matters because MorphoGNN's first EdgeConv
layer builds its kNN graph on these coordinates: an anisotropic squash would
change which points are neighbours, artificially separating strata. Setting
`z_scale` to ~20 to stretch depth to the same numeric range as xy is the obvious
first ablation, but it is an ablation, not the default. **Assumed** — §7.3.

The stock behaviour is still reachable — `"mode": "stock"` in the `normalize`
block calls `SWC2H5PY.Normalization` — because "what does the naive port score?"
is the baseline this folder exists to beat.

### 2.2 `GenerateH5py` corrupts any neuron with more than 6000 points

`ReadSWC` pads short neurons to 6000 points but **never clips long ones**;
`GenerateH5py` then concatenates everything and does `.reshape(-1, 6000, 3)`.
A neuron with 12000 points silently becomes *two* neurons, each half a cell,
both carrying the full label. A neuron with 6001 makes the reshape raise.

eyewire2 skeletons run from ~360 to ~31000 nodes, median ~6900 — so on this
dataset the stock path is wrong for roughly half the cells. `ReadSWC(CLIP=True)`
exists but no caller uses it, and clipping to the *first* N rows would take
whatever subtree skeliner happened to write first, not a sample of the arbor.

`preprocessing.sample_points` takes a uniform random sample of exactly
`n_points` nodes instead, soma always included, drawn with a per-cell
deterministic seed so a re-run reproduces the dataset. Node density is a
faithful stand-in for dendritic length density on these skeletons (`ssl_neuron`
spec §7.3 measured this: profile correlation ≥ 0.9935 over 184 cells), so a
uniform node sample is a uniform arbor sample.

Cells with fewer than `n_points` nodes are sampled **with replacement**, not
zero-padded. Stock zero-padding puts every pad point at the origin — which,
after soma-centering, is exactly the soma — inventing a dense blob whose size
encodes only how short the reconstruction was. Duplicating real points invents
no geometry. The QC floor (`min_nodes`, 1000, same as the GraphDINO pipeline)
keeps this to mild oversampling rather than a handful of points smeared out.

### 2.3 Batch-hard triplets degenerate at `batch_size=4`

`MorphoGNN.TripletLoss` picks, per anchor, the farthest same-class and nearest
different-class sample *within the batch*, and the similarity matrix includes
the diagonal. So an anchor alone in its class gets **itself** as hardest
positive (distance ~1e-6, the `euclidean_dist` floor). At batch size 4 over 7
classes that is already common; over **28** classes it is the overwhelmingly
likely case, and the triplet term would be almost pure noise.

`retina_dataset.PKSampler` fixes it the standard way: each batch is `P` classes
× `K` samples, so every anchor has `K-1` genuine positives and `(P-1)*K` genuine
negatives. Defaults `P=4, K=2` → batch size 8. `"sampler": "random"` restores
the stock behaviour for comparison.

---

## 3. Point count and batch size

`n_points` is the real memory knob, not parameter count. MorphoGNN's fourth
EdgeConv layer materializes a `(B, N, k, 2·480)` tensor at `k=64`:

| `n_points` | per-sample activation (layer 4) | usable batch |
| --- | --- | --- |
| 6000 (stock) | ~1.5 GB | 4 |
| 2048 (default here) | ~500 MB | 8 |
| 1024 | ~250 MB | 16 |

`knn`'s N×N distance matrix adds `4·N²` bytes per sample on top.

Default: **`n_points = 2048`, `batch_size = 8` (P=4, K=2)**. **Assumed** — and the
table above is arithmetic rather than measurement, §7.6. 2048
points over a ~35 µm depth range still gives a far finer depth profile than the
512 nodes the GraphDINO pipeline settled on, and buys a batch big enough for the
triplet loss to mean something. If this OOMs, halve `n_points` before touching
`P` — shrinking `P` is what re-breaks §2.3.

---

## 4. Augmentation

2816 cells over 28 classes is small for a supervised net with a 2048-d
embedding, so the position augmentations from the GraphDINO policy
(`ssl_neuron` spec §4) are applied at training time. Same policy, same
reasoning, applied to a point cloud instead of a graph:

| Augmentation | Setting | Note |
| --- | --- | --- |
| Rotation about z | uniform angle, proper SO(2) | Not the stock `rotate_graph` — that one is not a rotation (`ssl_neuron` spec §6.1). |
| Mirror in xy | random x-flip, p = 0.5 | |
| Isotropic xy scaling | ±5% | Noise absorption only; real size differences must survive. |
| Jitter | σ = (1.0, 1.0, 0.4) µm | Before the global scale is applied? No — see below. |
| Translation | σ = (10.0, 10.0, **0.0**) µm | z component must be exactly zero. |

Because the h5 stores *already-normalized* coordinates, the µm-valued
magnitudes above are divided by `xy_scale` / `z_scale` at load time
(`retina_dataset.augment_points` takes the same `normalize` block). A
z-translation is refused outright, not clamped, exactly as
`ssl_neuron.eyewire2.augment` refuses it.

Order: rotate → mirror → scale → jitter → translate.

Explicitly **not** done: point resampling per epoch. It would be a natural
extra view, but it needs a larger point pool in the h5 and breaks the drop-in
compatibility of §6. Worth revisiting if the model overfits.

---

## 5. Split and evaluation

Stratified **train / val / test**, 70 / 15 / 15, seeded. The upstream loop saves
on `test_acc >= best`, i.e. it selects the model on the set it reports — so its
headline number is optimistic. Here selection is on **val** and the number
reported in `04` is **test**, untouched during training. Classes with too few
cells to give at least one cell per split are dropped by the
`min_cells_per_class` filter of §1.

Reported in `04`:

- balanced accuracy and confusion matrix on the test split (the classes are
  imbalanced 20:235, so plain accuracy flatters the big types),
- k-NN accuracy in the 2048-d embedding — the retrieval metric, and the one
  comparable to the GraphDINO evaluation,
- precision@k for retrieval, and a t-SNE colored by celltype.

One caveat on all three: 406 of the 2816 cells carry a classifier-assigned
celltype rather than a human consensus one, and they are currently spread across
all three splits — so the test number partly measures agreement with that
classifier. See §7.4, which also explains why this can be sliced apart without
re-running `01`.

---

## 6. Artifacts and compatibility

`01` writes, into `./data/`:

| File | Contents |
| --- | --- |
| `TrainDatasets.h5`, `ValDatasets.h5`, `TestDatasets.h5` | `data` (M, `n_points`, 3) float32, `label` (M, 1) uint8, `cell_id` (M,) — the upstream h5 layout plus `cell_id`. |
| `label_map.json` | celltype string ↔ integer, the eyewire2 analog of `SWC2H5PY.LABEL7`. |
| `cell_meta.csv` | one row per kept cell: split, label, absolute soma position, node counts. |

The h5s are deliberately layout-compatible with the root `dataset.DataSet`, so
`DataSet(train_dir='eyewire2/data/TrainDatasets.h5', norm=False)` works as a
sanity check. **`norm=False` is not optional** — `norm=True` would re-apply the
stock normalization of §2.1 on top of the retina one.

---

## 7. Open items

Numbered so they can be cited from code and commits. Items shared with the
GraphDINO pipeline (what unit z is in, whether node spacing is uniform, how
accurate the warp is across cells) live in
`ssl_neuron/ssl_neuron/eyewire2/00_dataset_spec.md` §7 and are not duplicated
here — except where this pipeline depends on them differently, which is 7.1 and
7.5 below.

### 7.1 Is z really a *shared* depth frame in `skel_final`?

This is the load-bearing assumption of the whole folder. §2.1 never centers z
per cell, on the grounds that z lives in a depth frame shared across cells
(pywarper's conformal flattening, per the GraphDINO spec §1.2). **This pipeline
assumes that and does not verify it.** If the `skel_final` skeletons carried raw
EM z instead, the argument inverts: cells from different parts of the retina
would sit at different z because of curvature rather than celltype, a global
`z_offset` would be meaningless, and per-cell centering would be the *correct*
choice.

What is established so far only says z is in microns, not that the frame is
shared: the dataframe's own `height` over the 2816 selected cells has median
42.5 µm (range 11–67), and the locally mirrored skeletons have a node-z extent
of median ~35 µm. Both read as an IPL thickness.

Settle it with `02_visualize_data.py` on the full cluster set, before the first
real training run. Two of its plots answer it: the pooled depth histogram should
show IPL band structure rather than one smooth blob, and the per-class
stratification profiles should peak at *different* depths. Types cannot separate
by depth unless the frame is shared, so if that second plot shows separation,
the assumption holds.

### 7.2 The normalization constants were set from the wrong sample

`xy_scale = 300` and `z_offset = 25` are global constants, so getting them wrong
rescales or offsets every cell identically — it cannot destroy a signal, only
the numeric conditioning and the µm axis labels `02` prints. Still, both were
picked before the cluster numbers were in hand:

- `xy_scale = 300` µm. `hull_diameter` over the 2816 selected cells runs
  66–384 µm (median 164), so arbors land at roughly 0.2–1.3 units. Fine.
- `z_offset = 25` µm was the median soma z over the ~110 locally mirrored
  skeletons — which are mostly amacrine cells, not the RGC set this trains on.
  Recompute it on the cluster set.

### 7.3 Isotropic z, or depth-stretched?

§2.1 defaults to `z_scale = xy_scale`, preserving true geometry. The competing
option — `z_scale ≈ 20`, stretching the ~35 µm depth range to the same numeric
span as xy — would make depth differences dominate the kNN graph that every
EdgeConv layer rebuilds. The argument against is that this separates strata
that are genuinely adjacent; the argument for is that depth is the primary
signal and isotropic scaling buries it in a pancake. One config change, one
training run, settles it. Run it as the first ablation.

### 7.4 406 of 2816 labels come from a classifier, not from consensus

`celltype_final_decision` over the selected cells: 1611 `both_strong`, 799
`both_weak`, **406 `classifier`**. That last group's celltype was assigned by a
model rather than by human consensus.

Training on them teaches MorphoGNN another model's decision boundary. Having
them in **val/test** is the worse half of the problem, because the headline
number in `04` then partly measures agreement with that classifier rather than
with the biology.

Nothing is excluded today. `celltype_final_decision` is carried into
`cell_meta.csv`, so this can be sliced after the fact without re-running `01`.
The cheapest experiment is to keep them in train and drop them from val/test;
the strictest is to drop all 406 (−14% of the dataset) and see whether test
accuracy moves.

### 7.5 A truncated arbor now corrupts a *feature*, not just the cell count

Cells whose dendrites leave the imaged volume have no flag in
`df_all_neurons` (GraphDINO spec §7.2). Here that hurts more than it does there:
§2.1 preserves dendritic field size *on purpose*, so a clipped arbor does not
merely lose points — it presents as a genuinely smaller cell, which is exactly
how a different celltype presents.

The only guard in place is the `min_nodes = 1000` floor, which catches
small-and-clipped and misses large-and-clipped. The handle suggested by the
GraphDINO spec — `hull_points` running along the volume border — is not wired in
here either.

### 7.6 The memory table in §3 is arithmetic, not measurement

Those per-sample figures come from tensor shapes and ignore what autograd keeps
alive for the backward pass, so the real ceiling is higher than stated.
Measure `torch.cuda.max_memory_allocated()` on the first cluster run and correct
the table rather than tuning `n_points` by trial and error.

### 7.7 Uniform class sampling is already a rebalancing

`PKSampler` draws classes uniformly, so the smallest class (20 cells) is seen
~12× more often per epoch than the largest (235). That rebalancing hits the
cross-entropy term as well as the triplet term it was introduced for (§2.3).
Class weights are therefore deliberately *not* used — they would rebalance
twice. Whether uniform class sampling or the natural prior gives better
balanced accuracy is untested; `"sampler": "random"` is the comparison, though
it re-breaks the triplet loss, so the two effects are confounded.

### 7.8 The model cannot see which point is the soma

The cloud is xyz only, with no channel marking node 0, so nothing tells
MorphoGNN which of the `n_points` points is the soma — and the xy translation
augmentation (σ = 10 µm) moves the origin off it anyway. "Soma-centered" is thus
only a frame convention that makes cells comparable, not information the model
receives; it can infer soma location only from point density.

If soma position within the arbor turns out to matter (it plausibly does for the
asymmetric types), the fix is a fourth input channel, which means `conv1` goes
from 6 to 8 input dims — `get_graph_feature` concatenates `(x_j - x_i, x_i)`, so
input width is twice the feature dim.

### 7.9 Nothing is compared against anything yet

"Test MorphoGNN on eyewire2" is only meaningful against a baseline. Three exist,
in increasing order of work:

1. `"mode": "stock"` in the `normalize` block — what a naive port of the
   `neuron7` recipe scores. Reachable today by editing `config.json`.
2. The root `morphometrics.py` MLP on NeuroM features, which is the comparison
   upstream cares about. Needs its own eyewire2 adapter — it reads
   directory-per-class SWCs and hardcodes 16 input features.
3. The GraphDINO embedding from `ssl_neuron/ssl_neuron/eyewire2`, evaluated with
   the same k-NN protocol `04` uses. This is the interesting one: supervised
   point cloud vs. self-supervised graph, same cells, same metric.

### 7.10 Resolved: `_cal` vs. plain skeletons

`01` prefers `<id>_cal.swc` where it exists (98 of the 113 local skeletons),
following the GraphDINO pipeline. Checked over 25 pairs: node counts and xyz
coordinates are **identical**, only radius differs (by up to ~1 µm). Since this
pipeline reads xyz only, the preference is a no-op. Kept for consistency with
`ssl_neuron`, and safe to drop if it ever becomes inconvenient.

---

## 8. Where this lives

| File | Role |
| --- | --- |
| `00_dataset_spec.md` | this document |
| `config.json` | every number above, in one place |
| `preprocessing.py` | SWC parsing, axon removal, QC, point sampling, the §2.1 normalization. Pure numpy — no torch. |
| `retina_dataset.py` | `RetinaDataset`, `augment_points` (§4), `PKSampler` (§2.3). |
| `01_preprocess_data.py` | cell selection, the preprocessing loop with a QC breakdown, label map, stratified split. |
| `02_visualize_data.py` | sanity checks: does the depth frame survive, does size survive, do the augmentations behave. |
| `03_train_morphognn.py` | training. A third copy of the root training loop — see below. |
| `04_evaluate_results.py` | embeddings, confusion matrix, k-NN accuracy, retrieval, t-SNE. |

Nothing outside this folder is modified beyond `.gitignore` (which now ignores
`eyewire2/data/` and `eyewire2/ckpts/`) and an `eyewire2` extra in
`pyproject.toml` for `pandas` + `pyarrow`, which only this pipeline needs
because only it reads a parquet dataframe. `SWC2H5PY.Normalization` and
`MorphoGNN.MorphoGNN` / `DEVICE` / `TripletLoss` are imported from the repo
root as-is.

```bash
uv sync --extra eyewire2
uv run python eyewire2/01_preprocess_data.py     # -> eyewire2/data/
uv run python eyewire2/02_visualize_data.py      # sanity checks, no GPU
uv run python eyewire2/03_train_morphognn.py     # -> eyewire2/ckpts/
uv run python eyewire2/04_evaluate_results.py    # test metrics + t-SNE
```

Unlike the repo-root scripts, these resolve every path relative to **this
folder**, not the working directory, so they can be run from anywhere.

Two sharp edges worth knowing:

- **The dataset module is called `retina_dataset.py`, not `dataset.py`, on
  purpose.** Running a script from this folder puts it first on `sys.path`; a
  `dataset.py` here would shadow the root `dataset.py` that `MorphoGNN.py`
  imports at module level, and the resulting circular import fails at
  `import MorphoGNN`.
- `MorphoGNN.py` and `morphometrics.py` already carry near-duplicate training
  loops (root `CLAUDE.md`: "changes to one usually belong in both").
  `03_train_morphognn.py` is a **third**, and it is deliberately not a shared
  one: it differs in the sampler, the three-way split and the val-based
  selection of §5, and factoring those back into the root loop would change the
  `neuron7` results. If you fix a bug in the shared part — the LR floor, the
  save condition, the loss weighting — fix it in all three.
