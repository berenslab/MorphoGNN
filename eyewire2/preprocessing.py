""" Turning skeliner `.swc` skeletons into the fixed-size point clouds MorphoGNN
expects.

The pure functions behind `01_preprocess_data.py`; the notebook itself is the
driver (paths, cell selection, label map, plots, train/val/test split). Only
needs numpy, so it runs locally on Windows without a GPU.

What this replaces, and why, is `00_dataset_spec.md` section 2:

* `SWC2H5PY.ReadSWC` keeps every row in file order, pads short neurons with
  zeros and never clips long ones -- so `GenerateH5py`'s `.reshape(-1, N, 3)`
  splits any neuron above `n_points` into two half-cells. Replaced by
  `sample_points`.
* `SWC2H5PY.Normalization` centers each cloud on its own bounding box and
  scales each axis by its own extent, which discards both stratification depth
  and dendritic field size -- the two signals that separate RGC types.
  Replaced by `normalize_retina`.

The SWC parsing duplicates `ssl_neuron.eyewire2.preprocessing.load_swc` rather
than importing it: the two repos are installed independently and neither
depends on the other (workspace `CLAUDE.md`). The graph machinery from that
module -- neighbor dicts, connected components, stitching -- is deliberately not
duplicated, because MorphoGNN throws the edges away and rebuilds a kNN graph in
feature space at every layer.
"""
import numpy as np

# Skeletons below this many nodes (after axon removal) are broken or partial
# reconstructions rather than small cells. Same floor as the GraphDINO
# pipeline, for the same reason: a truncated arbor is worse than no cell,
# now that arbor size is a feature the model is meant to use.
MIN_NODES = 1000

SOMA_TYPES = (1, -1)
AXON_TYPE = 2


class SkeletonQCError(ValueError):
    """ A skeleton that must not enter the dataset. The message is the reason,
    and `01_preprocess_data.py` tallies these into a drop breakdown. """


def load_swc(path):
    """ Standard 7-column SWC (`id type x y z radius parent`). Node type follows
    the usual convention (1 or -1 = soma, 2 = axon, 3 = dendrite, >=4 = apical
    dendrite); most of these retina skeletons only have soma + dendrite nodes,
    but a few also carry a (partial, unreliable) axon.

    Returns:
        xyz: node positions (N x 3).
        radii: node radii (N,), unused by the model but kept for filtering.
        types: SWC compartment type (N,).
        parent_idx: parent *row index* per node, -1 for the root.
    """
    ids, types, xyz, radii, parents = [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            n_id, n_type, x, y, z, r, parent = line.split()
            ids.append(int(n_id))
            types.append(int(n_type))
            xyz.append((float(x), float(y), float(z)))
            radii.append(float(r))
            parents.append(int(parent))

    id2idx = {node_id: i for i, node_id in enumerate(ids)}
    parent_idx = np.array([id2idx[p] if p != -1 else -1 for p in parents])
    return np.array(xyz), np.array(radii), np.array(types), parent_idx


def find_soma(types, parent_idx):
    """ Index of the soma node: the unique root, which must also be typed soma. """
    roots = np.where(parent_idx == -1)[0]
    if len(roots) != 1:
        raise SkeletonQCError(f'{len(roots)} root nodes, expected exactly 1')
    soma_id = int(roots[0])
    if types[soma_id] not in SOMA_TYPES:
        raise SkeletonQCError(f'root node is typed {types[soma_id]}, expected soma')
    return soma_id


def drop_axon(xyz, types, soma_id):
    """ Drop every node typed as axon, keeping the soma first.

    Axons in these skeletons are incomplete and unreliable: their extent
    reflects how far the reconstruction got, not the cell's biology (see
    `ssl_neuron/ssl_neuron/eyewire2/00_dataset_spec.md` section 1.1). This is a
    plain type mask, the same nodes `ssl_neuron.data.data_utils.remove_axon`
    removes -- but with no graph to keep connected, the hole it leaves in a
    mislabeled dendrite costs nothing here, so there is no repair step.

    Returns:
        xyz with the soma as row 0 and axon nodes removed, and the axon count.
    """
    keep = types != AXON_TYPE
    keep[soma_id] = True  # never drop the soma, whatever it is typed
    n_axon = int((~keep).sum())

    kept = np.flatnonzero(keep)
    soma_row = int(np.flatnonzero(kept == soma_id)[0])
    order = np.concatenate([[soma_row], np.delete(np.arange(len(kept)), soma_row)])
    return xyz[kept][order], n_axon


def sample_points(xyz, n_points, rng):
    """ Exactly `n_points` nodes, the soma (row 0) always among them.

    Uniform over nodes, which is uniform over dendritic length on these
    skeletons: node spacing does not vary with depth, so a node histogram
    tracks a length-weighted one to within a few percent (measured in
    `ssl_neuron/ssl_neuron/eyewire2/00_dataset_spec.md` section 7.3).

    Short cells are sampled **with replacement** rather than zero-padded.
    Padding with [0, 0, 0] would pile every pad point onto the soma -- which
    sits at the origin after centering -- inventing a dense blob whose size
    encodes nothing but how short the reconstruction was.
    """
    n_rest = n_points - 1
    replace = len(xyz) - 1 < n_rest
    idx = rng.choice(np.arange(1, len(xyz)), size=n_rest, replace=replace)
    return xyz[np.concatenate([[0], idx])]


def normalize_retina(points, soma_xyz, xy_scale=300.0, z_offset=25.0, z_scale=300.0):
    """ The retina coordinate frame of `00_dataset_spec.md` section 2.1.

    x and y are centered on the cell's own soma and divided by a **global**
    constant; z is offset and divided by **global** constants and is never
    centered per cell. Every scale here is a dataset-level number, not a
    per-cell statistic -- that is the entire point. Relative dendritic field
    size survives across cells, and so does absolute stratification depth.

    This encodes the assumption that z is a depth frame *shared across cells*
    (pywarper's flattening). That assumption is not verified anywhere in this
    pipeline -- if these skeletons carried raw EM z, per-cell centering would be
    the correct choice instead. See `00_dataset_spec.md` section 7.1, and settle
    it with `02_visualize_data.py` before the first real training run.

    `z_scale` defaults to `xy_scale`, i.e. the map is isotropic and the true
    geometry is preserved. MorphoGNN's first EdgeConv layer builds its kNN
    graph on these coordinates, so an anisotropic squash would change which
    points are neighbours. Stretching depth (`z_scale` ~ 20) is an ablation,
    not the default.
    """
    out = np.empty_like(points, dtype=np.float64)
    out[:, :2] = (points[:, :2] - soma_xyz[:2]) / xy_scale
    out[:, 2] = (points[:, 2] - z_offset) / z_scale
    return out


def normalize_points(points, soma_xyz, mode='retina', **kwargs):
    """ Dispatch between the retina frame and upstream's per-axis bounding-box
    normalization. `mode='stock'` is the naive-port baseline this folder exists
    to beat -- it is wrong for this dataset (section 2.1), on purpose. """
    if mode == 'retina':
        return normalize_retina(points, soma_xyz, **kwargs)
    if mode == 'stock':
        from SWC2H5PY import Normalization
        return Normalization(points[None])[0]
    raise ValueError(f"unknown normalization mode {mode!r}, expected 'retina' or 'stock'")


def preprocess_cell(swc_path, n_points, min_nodes=MIN_NODES, normalize=None, seed=0):
    """ Full per-cell preprocessing: parse, drop axons, QC, sample, normalize.

    Args:
        swc_path: path to the skeleton.
        n_points: how many points the model gets, exactly.
        min_nodes: QC floor on dendritic node count.
        normalize: the `data.normalize` block of `config.json`, or None to
            leave coordinates in raw microns.
        seed: per-cell seed, so a re-run reproduces the same sample.

    Returns:
        points: (n_points, 3) float32 in the normalized frame.
        soma_xyz: absolute soma position before centering, unrecoverable from
            `points` and needed by any later mosaic analysis.
        info: per-cell counts for the preprocessing report.

    Raises:
        SkeletonQCError: if the skeleton must not enter the dataset.
    """
    xyz, _, types, parent_idx = load_swc(swc_path)
    soma_id = find_soma(types, parent_idx)

    n_raw = len(xyz)
    xyz, n_axon = drop_axon(xyz, types, soma_id)
    if len(xyz) < min_nodes:
        raise SkeletonQCError(f'{len(xyz)} nodes after axon removal, below {min_nodes}')

    soma_xyz = xyz[0].copy()
    points = sample_points(xyz, n_points, np.random.default_rng(seed))
    if normalize is not None:
        points = normalize_points(points, soma_xyz, **normalize)

    info = {
        'n_nodes_raw': n_raw,
        'n_nodes': len(xyz),
        'n_axon_nodes': n_axon,
        'oversampled': bool(len(xyz) < n_points),
    }
    return points.astype(np.float32), soma_xyz, info
