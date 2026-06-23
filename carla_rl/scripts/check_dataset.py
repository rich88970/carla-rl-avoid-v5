"""Sanity-check the EasyCarla offline dataset (HDF5).

Usage:
    python -m carla_rl.scripts.check_dataset [path/to/easycarla_offline_dataset.hdf5]
"""

import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'data' / 'easycarla_offline_dataset.hdf5'


def walk_datasets(node, prefix=''):
    for key in node.keys():
        item = node[key]
        path = f'{prefix}/{key}' if prefix else key
        if isinstance(item, h5py.Group):
            yield from walk_datasets(item, path)
        else:
            yield path, item


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    print(f'Opening {path} ({path.stat().st_size / 1e9:.2f} GB)')

    with h5py.File(path, 'r') as f:
        datasets = list(walk_datasets(f))

        print('\nKeys, shapes, stats (first 100k rows):')
        for name, ds in datasets:
            sample = np.asarray(ds[: min(len(ds), 100_000)], dtype=np.float64)
            print(
                f'  {name}: shape={ds.shape} dtype={ds.dtype} '
                f'min={sample.min():.3f} max={sample.max():.3f} mean={sample.mean():.3f}'
            )

        by_name = dict(datasets)
        obs = next((ds for name, ds in datasets if 'observation' in name.lower() or name == 'obs'), None)
        if obs is not None and len(obs.shape) == 2:
            status = 'OK' if obs.shape[1] == 307 else 'UNEXPECTED (wrapper assumes 307)'
            print(f'\nObservation dim: {obs.shape[1]} -> {status}')

        done_key = next((k for k in ('done', 'dones', 'terminals') if k in by_name), None)
        if done_key:
            dones = np.asarray(by_name[done_key][:], dtype=bool)
            print(f'Episodes (by {done_key}): {dones.sum()} over {len(dones)} steps')


if __name__ == '__main__':
    main()
