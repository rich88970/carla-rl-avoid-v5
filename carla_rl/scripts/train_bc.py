"""Behavior cloning on the EasyCarla offline dataset.

Trains the GaussianActor's mean head with MSE on expert-filtered transitions.
Saves checkpoint + normalizer to carla_rl/checkpoints/bc_actor.pth and a
training log CSV next to it.

Usage:
    python -m carla_rl.scripts.train_bc [--epochs 10] [--expert-percentile 20]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from carla_rl.data.offline_dataset import load_bc_data
from carla_rl.models import GaussianActor, ObsNormalizer

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--expert-percentile', type=float, default=20.0)
    parser.add_argument('--hidden', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--steer-weight', type=float, default=5.0,
                        help='loss weight on the steer dim (vs 1.0 throttle/brake)')
    parser.add_argument('--steer-sample-boost', type=float, default=4.0,
                        help='per-sample weight = 1 + boost * |steer|, emphasizes curves')
    parser.add_argument('--out', default=str(CHECKPOINT_DIR / 'bc_actor.pth'))
    parser.add_argument('--data-npz', default=None,
                        help='train on a fresh (obs, act) npz (V10 imitation) instead of the '
                             'HDF5 dataset; obs may be any width (e.g. 317)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('loading dataset...')
    if args.data_npz:
        d = np.load(args.data_npz)
        obs, act = d['obs'].astype(np.float32), d['act'].astype(np.float32)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(obs))
        nv = int(len(obs) * 0.05)
        val_obs, val_act = obs[perm[:nv]], act[perm[:nv]]
        train_obs, train_act = obs[perm[nv:]], act[perm[nv:]]
        stats = {'source': str(args.data_npz), 'transitions': int(len(obs)),
                 'train_size': int(len(train_obs)), 'val_size': int(nv)}
    else:
        train_obs, train_act, val_obs, val_act, stats = load_bc_data(
            expert_percentile=args.expert_percentile
        )
    print(json.dumps(stats, indent=2))

    normalizer = ObsNormalizer.fit(train_obs)
    train_obs = normalizer(train_obs)
    val_obs = normalizer(val_obs)

    obs_dim = train_obs.shape[1]
    actor = GaussianActor(obs_dim, hidden=tuple(args.hidden)).to(device)
    optim = torch.optim.Adam(actor.parameters(), lr=args.lr)

    t_obs = torch.as_tensor(train_obs, device=device)
    t_act = torch.as_tensor(train_act, device=device)
    v_obs = torch.as_tensor(val_obs, device=device)
    v_act = torch.as_tensor(val_act, device=device)

    # MSE regression-to-mean attenuates steering (pred slope 0.36 on curve
    # frames with plain MSE -> instant off-road). Weight the steer dim and
    # boost curve samples; Huber blunts random-action outliers in the data.
    dim_weights = torch.tensor([1.0, args.steer_weight, 1.0], device=device)
    t_sample_w = 1.0 + args.steer_sample_boost * t_act[:, 1].abs()

    def weighted_loss(pred, target, sample_w):
        per_dim = torch.nn.functional.smooth_l1_loss(
            pred, target, beta=0.1, reduction='none'
        )
        return (per_dim * dim_weights * sample_w.unsqueeze(-1)).mean()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix('.train_log.csv')
    log_rows = []

    n = len(t_obs)
    best_val = float('inf')
    for epoch in range(args.epochs):
        actor.train()
        perm = torch.randperm(n, device=device)
        losses = []
        t0 = time.time()
        for i in range(0, n, args.batch_size):
            batch = perm[i: i + args.batch_size]
            pred = actor.mean_action(t_obs[batch])
            loss = weighted_loss(pred, t_act[batch], t_sample_w[batch])
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        actor.eval()
        with torch.no_grad():
            v_pred = actor.mean_action(v_obs)
            val_loss = torch.nn.functional.mse_loss(v_pred, v_act).item()
            # steering quality is what decides lane keeping — track it directly
            big = v_act[:, 1].abs() > 0.2
            steer_slope = (
                (v_pred[big, 1] * v_act[big, 1]).mean() / (v_act[big, 1] ** 2).mean()
            ).item() if big.any() else float('nan')
        train_loss = float(np.mean(losses))
        print(f'epoch {epoch}: train_loss={train_loss:.5f} val_mse={val_loss:.5f} '
              f'steer_slope={steer_slope:.3f} ({time.time() - t0:.0f}s)')
        log_rows.append({'epoch': epoch, 'train_mse': train_loss, 'val_mse': val_loss,
                         'steer_slope': steer_slope})

        # select on steer-weighted val loss: plain MSE under-values steering,
        # which is exactly what decides whether the car stays in its lane
        with torch.no_grad():
            v_w = 1.0 + args.steer_sample_boost * v_act[:, 1].abs()
            val_select = weighted_loss(v_pred, v_act, v_w).item()
        if val_select < best_val:
            best_val = val_select
            torch.save({
                'actor': actor.state_dict(),
                'obs_dim': obs_dim,
                'hidden': list(args.hidden),
                'normalizer': normalizer.state_dict(),
                'dataset_stats': stats,
                'val_mse': val_loss,
                'epoch': epoch,
            }, out_path)

    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['epoch', 'train_mse', 'val_mse', 'steer_slope']
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print(f'best weighted val loss={best_val:.5f}; checkpoint -> {out_path}')


if __name__ == '__main__':
    main()
