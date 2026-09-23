""" Dataset, augmentations and batch sampler for MorphoGNN on eyewire2 RGCs.

Deliberately **not** called `dataset.py`: running a script from this folder
puts the folder first on `sys.path`, and a `dataset.py` here would shadow the
repo-root `dataset.py` that `MorphoGNN.py` imports at module level -- the
resulting circular import fails at `import MorphoGNN`.

Three things live here, each answering a specific defect of the stock setup
(`00_dataset_spec.md` sections 2.3 and 4):

* `RetinaDataset` -- reads the h5 written by `01_preprocess_data.py`. The
  coordinates in it are **already normalized**, so unlike the root `DataSet`
  this one never re-normalizes.
* `augment_points` -- the position augmentation policy of the GraphDINO
  pipeline, applied to a point cloud: proper SO(2) rotation about z, xy mirror,
  bounded xy scaling, per-axis jitter, xy-only translation. A z-translation is
  refused outright.
* `PKSampler` -- P classes x K samples per batch, so the batch-hard triplet
  loss has real positives and negatives to pick from. Without it, at batch
  size 8 over 28 classes, almost every anchor is alone in its class and gets
  *itself* as its hardest positive.
"""
import logging

import h5py
import numpy as np
import torch
import torch.utils.data as data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def read_h5(path):
    """ The h5 layout written by `01_preprocess_data.py`: the upstream
    `data`/`label` pair plus a `cell_id` column. Read without normalization --
    the file already holds normalized coordinates. """
    with h5py.File(path, 'r') as f:
        points = f['data'][:]
        label = f['label'][:]
        cell_id = f['cell_id'][:] if 'cell_id' in f else np.arange(len(label))
    if cell_id.dtype.kind == 'S':
        cell_id = cell_id.astype(str)
    return points, label.squeeze(-1) if label.ndim > 1 else label, cell_id


def _axis_scales(normalize):
    """ The µm-per-unit factors the stored coordinates were divided by, used to
    convert the µm-valued augmentation magnitudes of `config.json` into the
    normalized frame. `stock` normalization has no µm scale at all, so the
    magnitudes are left as-is and jitter/translation should be switched off. """
    if normalize is None or normalize.get('mode', 'retina') != 'retina':
        return np.ones(3)
    xy = float(normalize.get('xy_scale', 300.0))
    return np.array([xy, xy, float(normalize.get('z_scale', xy))])


def augment_points(points, rotate_xy=True, mirror_xy=True, scale_xy=0.05,
                   jitter=(1.0, 1.0, 0.4), translate=(10.0, 10.0, 0.0),
                   normalize=None, rng=None):
    """ One augmented view of a point cloud, in the order rotate -> mirror ->
    scale -> jitter -> translate.

    `jitter` and `translate` are given in **microns** (as in `config.json`) and
    converted here, because the stored coordinates are normalized.

    The z axis is close to sacred: it carries stratification depth in a frame
    shared across cells, so nothing here translates, flips or scales it. Only a
    bounded jitter touches z, at roughly 1% of IPL thickness. A non-zero
    z-translation is a bug in the config, not something to clamp -- it moves a
    cell to a different celltype's depth -- so it raises.
    """
    rng = np.random.default_rng() if rng is None else rng
    scales = _axis_scales(normalize)
    out = np.asarray(points, dtype=np.float64).copy()

    if rotate_xy:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cos, sin = np.cos(theta), np.sin(theta)
        out[:, :2] = out[:, :2] @ np.array([[cos, -sin], [sin, cos]]).T

    if mirror_xy and rng.random() < 0.5:
        out[:, 0] *= -1.0

    if scale_xy:
        out[:, :2] *= 1.0 + rng.uniform(-scale_xy, scale_xy)

    if jitter is not None and np.any(np.asarray(jitter) > 0):
        out += rng.normal(0.0, np.asarray(jitter, dtype=float) / scales, size=out.shape)

    if translate is not None and np.any(np.asarray(translate) != 0):
        translate = np.asarray(translate, dtype=float)
        if translate[2] != 0:
            raise ValueError(
                'translate[2] must be 0: shifting a cell in z moves it to another '
                'stratification depth, i.e. to another celltype '
                '(00_dataset_spec.md section 4).')
        out += rng.normal(0.0, translate / scales, size=(1, 3))

    return out


class RetinaDataset(data.Dataset):
    """ Fixed-size point clouds plus integer celltype labels.

    Args:
        h5_path: one of the files written by `01_preprocess_data.py`.
        augment: the `data.augment` block of `config.json`, or None for no
            augmentation (validation, test, and any feature extraction).
        normalize: the `data.normalize` block, needed only to convert the
            augmentation magnitudes out of microns.
    """

    def __init__(self, h5_path, augment=None, normalize=None):
        self.points, self.label, self.cell_id = read_h5(h5_path)
        self.augment = augment
        self.normalize = normalize

        logging.info('{}: data {}, label {}, {} classes'.format(
            h5_path, self.points.shape, self.label.shape, len(np.unique(self.label))))

    def __getitem__(self, item):
        points = self.points[item]
        if self.augment:
            points = augment_points(points, normalize=self.normalize, **self.augment)
        return points.astype(np.float32), np.int64(self.label[item])

    def __len__(self):
        return len(self.label)

    def class_counts(self):
        labels, counts = np.unique(self.label, return_counts=True)
        return dict(zip(labels.tolist(), counts.tolist()))


class PKSampler(data.Sampler):
    """ Batches of P classes x K samples, the standard companion to a
    batch-hard triplet loss.

    `MorphoGNN.TripletLoss` takes, per anchor, the farthest same-class and the
    nearest different-class sample *inside the batch*, and its similarity
    matrix includes the diagonal. Under random sampling at batch size 8 over 28
    classes, an anchor is almost always the only member of its class present,
    so its "hardest positive" is itself at distance ~1e-6 and the term is noise
    (`00_dataset_spec.md` section 2.3). Here every anchor has K-1 real
    positives and (P-1)*K real negatives.

    Classes with fewer than K samples still get drawn -- with replacement
    within the batch -- rather than being silently excluded from training.
    """

    def __init__(self, labels, classes_per_batch=4, samples_per_class=2,
                 batches_per_epoch=None, seed=None):
        self.labels = np.asarray(labels)
        self.P = int(classes_per_batch)
        self.K = int(samples_per_class)
        self.classes = np.unique(self.labels)
        if len(self.classes) < self.P:
            raise ValueError(f'{len(self.classes)} classes but classes_per_batch={self.P}')
        self.by_class = {c: np.flatnonzero(self.labels == c) for c in self.classes}
        self.batches_per_epoch = (batches_per_epoch if batches_per_epoch is not None
                                  else max(1, len(self.labels) // (self.P * self.K)))
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            for c in self.rng.choice(self.classes, size=self.P, replace=False):
                pool = self.by_class[c]
                idx = self.rng.choice(pool, size=self.K, replace=len(pool) < self.K)
                yield from idx.tolist()

    def __len__(self):
        return self.batches_per_epoch * self.P * self.K


def build_dataloaders(config, data_dir, splits=('Train', 'Val', 'Test')):
    """ One loader per split. Train gets the augmentations and (by default) the
    PK sampler; val and test get neither, and keep their file order so
    predictions line up with `cell_meta.csv`. """
    from pathlib import Path

    data_cfg, train_cfg = config['data'], config['training']
    loaders = {}
    for split in splits:
        path = Path(data_dir) / f'{split}Datasets.h5'
        is_train = split == 'Train'
        dataset = RetinaDataset(
            path,
            augment=data_cfg['augment'] if is_train else None,
            normalize=data_cfg['normalize'],
        )
        if is_train and train_cfg.get('sampler', 'pk') == 'pk':
            sampler = PKSampler(dataset.label,
                                classes_per_batch=train_cfg['classes_per_batch'],
                                samples_per_class=train_cfg['samples_per_class'],
                                seed=data_cfg['seed'])
            batch_size = train_cfg['classes_per_batch'] * train_cfg['samples_per_class']
            loaders[split] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=train_cfg['num_workers'], drop_last=True)
        else:
            loaders[split] = torch.utils.data.DataLoader(
                dataset, batch_size=train_cfg['batch_size'], shuffle=is_train,
                num_workers=train_cfg['num_workers'], drop_last=is_train)
    return loaders
