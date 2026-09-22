# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MorphoGNN: a DGCNN-style (EdgeConv) point-cloud graph network that classifies and retrieves single neuron morphologies read from `.swc` files. Vendored third-party research code (upstream author `fun0515`, Apache-2.0) — the git history is upstream's, not this workspace's. A companion repo for reconstruction-quality classification is linked at the bottom of the README.

Two parallel tracks live here, sharing the same data layout and label maps:

- **The GNN track** — `SWC2H5PY.py` (data) → `MorphoGNN.py` (train) → `retrieval.py` (embed + query).
- **The morphometrics baseline** — `morphometrics.py`, which classifies the same neurons from 16 classical NeuroM measurements with a small MLP, to compare against the learned embedding.

## Environment & packaging

- `uv sync`, then either prefix commands with `uv run` or activate `.venv`.
- Packaged with hatchling. The modules stay **flat at the repo root on purpose** — every entry point is `python <script>.py` run from the repo root, so there is no package directory. `[tool.hatch.build.targets.wheel] only-include` lists the five modules explicitly so the wheel ships exactly those.
- **`neurom<3` is load-bearing, not conservatism.** `morphometrics.py` uses the pre-3.0 NeuroM API: `nm.load_neuron` (removed in 4.x) and the feature names `n_sections`, `n_leaves`, `neurite_lengths`, `n_bifurcation_points`, `n_segments` (all removed in 3.x, renamed to `number_of_*` / `total_length_per_neurite`). On 2.3.1 all 16 features resolve; on 3.x five of them raise `NeuroMError`; on 4.x the loader is gone too. Bumping the pin requires porting `ReturnFeatures` and re-checking that each renamed feature means the same thing.
- No test suite, linter config, or CI exists. `uvx ruff check --select F,E9 .` is a useful ad-hoc check for undefined names and unused imports (currently clean).
- `torch` resolves to the CPU build from PyPI here; nothing pins a CUDA index.

## Running things

```bash
uv run python SWC2H5PY.py    --swc_dir=./neuron7    # -> TrainDatasets_6000.h5 / TestDatasets_6000.h5
uv run python MorphoGNN.py                          # 50 epochs -> MorphoGNN.t7
uv run python retrieval.py   --task=ExtractFeature --model_path=./MorphoGNN.t7 --swc_dir=./neuron7
uv run python retrieval.py   --task=QueryTest --swc_dir=./neuron7
uv run python retrieval.py   --task=Visualize       # t-SNE of database.npy
uv run python morphometrics.py --swc_dir=./neuron7  # builds its own .h5 pair, then trains the MLP
```

Every script reads and writes relative to the **current working directory**, not the repo root, and the paths (`./TrainDatasets_6000.h5`, `./database.npy`, `./MorphoGNN.t7`) are hardcoded rather than flags. Run from wherever the artifacts should land. All generated `.h5`/`.npy`/`.t7` are gitignored.

`MorphoGNN.py` and `morphometrics.py` each have a training loop as a near-duplicate `__main__`/`train()` — same optimizer, `StepLR`, LR floor of 1e-5, and "save on `test_acc >= best`" logic. Changes to one usually belong in both.

## Data layout contract

Each script takes a directory holding **one sub-directory per class**, and derives the integer label from the directory name via `LABEL7`:

```text
neuron7/{amacrine,aspiny,basket,bipolar,pyramidal,spiny,stellate}/*.swc
```

The label is `dir_list.split('/')[-1]`, and the parent path is always joined with `/` in code, so the parent's style (`.\neuron7`, a trailing slash, an absolute Windows path) does not matter — only the leaf directory name does. Two sharp edges do bite:

- A sub-directory name outside `LABEL7` is a `KeyError`, not a skip.
- The class loop does `os.listdir` on **every** entry of the parent directory, so a stray *file* there (a `.DS_Store`, a README, a leftover `.h5`) raises `NotADirectoryError`. Keep the parent directory free of anything but class directories.

`LABEL7` and `LABEL10` are defined once in [SWC2H5PY.py](SWC2H5PY.py) and imported by `retrieval.py` and `morphometrics.py`. `LABEL10` is a different dataset's map and is currently unused by any code path.

## Architecture

**Point-cloud representation** (`SWC2H5PY.py`): `ReadSWC` keeps only the xyz columns of each `.swc` row, discarding radius, compartment type and parent — so the graph structure is thrown away and rebuilt later as a kNN graph in feature space. Every neuron is forced to exactly 6000 points, zero-padded when short. **Nothing clips when long**: `GenerateH5py` concatenates every neuron's points into one array and then `.reshape(-1, 6000, 3)`. So an over-long neuron either makes the reshape raise (total not a multiple of 6000) or, worse, passes silently and gets split across two "neurons" that are each half a cell. `ReadSWC(CLIP=True)` exists for this but no caller uses it — every call site passes `CLIP=False`. `Normalization` centers each cloud on its bounding-box midpoint and divides each axis by its max only when that max exceeds 1 — so axes are scaled independently and not always to a common range.

**Model** (`MorphoGNN.py`): EdgeConv over a kNN graph recomputed in *feature* space at every layer, with **k growing 8 → 16 → 32 → 64** as receptive fields widen, and dense concatenation of all previous layers' outputs feeding each next layer (6 → 32 → 64 → 128 → 256, concatenated to 480, then conv5 → 1024). Max- and average-pooling are concatenated into a **2048-d embedding**. `forward` returns `(feature, logits)` and *also* stashes the embedding on the module as `self.feature`; that attribute is redundant now that every call site unpacks the tuple, but it makes the module stateful and keeps the last batch's activations alive. Input is `(batch, 3, n_points)`; the datasets yield `(n_points, 3)`, so callers must `permute(0, 2, 1)`.

`knn` materializes an N×N distance matrix per sample. At N=6000 that, not parameter count, is what forces `batch_size=4`.

**Loss**: cross-entropy on the logits plus a batch-hard `TripletLoss` on the L2-normalized embedding (margin 0.5), summed 1:1. `_batch_hard` picks, per anchor, the *farthest* same-class and *nearest* different-class sample, masking the other side with ±100000. The similarity matrix includes the diagonal, so an anchor that is the only member of its class in the batch gets **itself** as its hardest positive (distance ~0), and a batch that is entirely one class gets no valid negative at all (the masked 100000 survives). With `batch_size=4` over 7 classes both degenerate cases are common, so the triplet term is noisy by construction — worth remembering before tuning its weight.

`euclidean_dist` floors squared distances at `1e-12` before `sqrt` (to keep the gradient finite), so self-distances come out as 1e-6 rather than 0. That is intended; do not "fix" it by comparing against `torch.cdist`.

**Device**: `MorphoGNN.DEVICE` is the single definition (`cuda` when available, else `cpu`), imported by `retrieval.py` and `morphometrics.py`. `get_graph_feature` takes its device from the input tensor. Keep new code on that constant rather than reintroducing literal `'cuda'`.

**Retrieval** (`retrieval.py`): `ExtractFeature` writes `database.npy`, a dict of `"<class>/<file>.swc"` → 2048-d vector. `QueryTest` ranks by cosine similarity, then recovers each hit's filename with `ReturnKey`, a **linear search of the dict comparing vectors by value** — two identical embeddings would return the wrong key. Note also that it builds `dict(zip(dist, value))` keyed by float similarity, so exact-tie distances silently collapse.

Known CLI/behavior gaps, left as upstream had them: `QueryTest` queries one hardcoded `.swc` path, so `--query_times` has no effect. The commented-out `random.sample` line above it cannot simply be restored — `random.sample` rejects a `dict_keys` with `TypeError: Population must be a sequence`; it needs `list(database)`. `QueryTests` loops the whole database but is not wired to argparse.

**Morphometrics baseline** (`morphometrics.py`): `ReturnFeatures` returns exactly **16** scalars, and `Mlp.linear1` is `nn.Linear(16, 512)`. That width is hardcoded — adding or removing a morphometric means editing both (upstream's recent commits were exactly this, 5 → 16). `GenerateMorphometricsAddMorphoGNNDatabase` concatenates the 16 morphometrics with the 2048-d learned features into a fused dataset, which would need a third input width again.

## Safety notes for this code

- `np.load(..., allow_pickle=True)` is used for every `.npy` database in `retrieval.py` and `morphometrics.py`. Loading a `database.npy` from an untrusted source executes arbitrary code.
- `morphometrics.Replace()` rewrites an `.swc` in place via `os.remove` + `os.rename` on the paths given to it. Nothing calls it; it is a manual repair utility.
- Otherwise the modules only do array math and read/write `.swc`/`.h5`/`.npy`/`.t7` — no network, subprocess, or `eval`.
