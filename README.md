# MorphoGNN

Morphological embedding of single neurons with graph neural networks.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

That creates a `.venv` with all dependencies (PyTorch, NeuroM, h5py, scikit-learn, ...).
Either prefix the commands below with `uv run`, or activate the environment first.

Training, feature extraction and retrieval use CUDA when it is available and fall back
to CPU otherwise.

## Expected data layout

All scripts read a directory holding one sub-directory per class, each containing the
`.swc` files of that class. Directory names are the class labels and must be the seven
keys of `LABEL7` in [SWC2H5PY.py](SWC2H5PY.py):

```
neuron7/
├── amacrine/    *.swc
├── aspiny/      *.swc
├── basket/      *.swc
├── bipolar/     *.swc
├── pyramidal/   *.swc
├── spiny/       *.swc
└── stellate/    *.swc
```

## Data preprocessing

```bash
python SWC2H5PY.py --swc_dir=./neuron7
```

`--swc_dir` is the path where the morphology data is stored. Each neuron is read as a
point cloud of 6000 points (shorter ones are zero-padded), and the neurons are shuffled
and split 70/30 into `TrainDatasets_6000.h5` and `TestDatasets_6000.h5` in the current
directory.

## Train MorphoGNN

Once the trainable data is generated, run:

```bash
python MorphoGNN.py
```

to train the MorphoGNN model for 50 epochs. The checkpoint with the best test accuracy
is written to `MorphoGNN.t7` in the current directory.

## Retrieval

[retrieval.py](retrieval.py) retrieves nerve fibers with a trained MorphoGNN model.
First build the feature library, saved as `database.npy`:

```bash
python retrieval.py --task=ExtractFeature --model_path=./MorphoGNN.t7 --swc_dir=./neuron7
```

Then query it for the ten most similar neurons (by cosine similarity) and plot them next
to the query:

```bash
python retrieval.py --task=QueryTest --swc_dir=./neuron7
```

Note that `QueryTest` currently queries one hard-coded neuron, so the `--query_times`
flag has no effect. `QueryTests()` loops over the whole database, but is not wired to
the command line.

To visualize the feature distribution with t-SNE:

```bash
python retrieval.py --task=Visualize
```

## Morphometrics

[morphometrics.py](morphometrics.py) is an example of classifying neurons with sixteen
traditional morphometrics, captured through [NeuroM](https://github.com/BlueBrain/NeuroM).
Run:

```bash
python morphometrics.py --swc_dir=./neuron7
```

to generate datasets of traditional morphometrics and train a simple multilayer
perceptron to classify them.

This uses NeuroM's pre-3.0 API, so `pyproject.toml` pins `neurom<3`; NeuroM 3 renamed the
loader and dropped several of the feature names used here.

## Reconstruction Quality Classification

https://github.com/sfwmusi/MorphoGNN_reconcls
