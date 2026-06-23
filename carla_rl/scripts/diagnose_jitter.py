"""Diagnose the source of action jitter — OFFLINE, no CARLA server.

Replays a contiguous dataset trajectory (autopilot-driven, so the obs are
on-distribution and smooth-by-construction) through:
  - the BC actor (307-dim), and
  - the v2.2 SAC actor (315-dim, with its EMA+rate-cap smoothing applied)
and measures action smoothness against the expert's own actions. Then probes
action SENSITIVITY: perturb one observation with small noise, measure the
spread of the policy output, isolating which obs block drives the jitter.

Decisive questions:
  Q1  Did RL make the policy intrinsically jittery? (BC smooth + SAC jittery
      on the SAME obs => yes, training is the cause.)
  Q2  Is the policy amplifying observation noise? (small obs noise -> large
      action spread => yes; and which block: lidar / nearby / waypoints.)
"""

import sys

import h5py
import numpy as np
import torch

from carla_rl.models import GaussianActor, ObsNormalizer
from carla_rl.wrappers.traffic_light import NEUTRAL_FEATURES
from carla_rl.data.offline_dataset import DATASET_PATH

BC_PATH = r'carla_rl\checkpoints\bc_actor.pth'
# any SAC checkpoint: arg 1, default v2.2
SAC_PATH = sys.argv[1] if len(sys.argv) > 1 else r'carla_rl\checkpoints\sac_v22_latest.pth'


def load_actor(path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    a = GaussianActor(c['obs_dim'], hidden=tuple(c['hidden']))
    a.load_state_dict(c['actor'])
    a.eval()
    meta = c.get('meta', {})
    return a, ObsNormalizer.from_state_dict(c['normalizer']), c['obs_dim'], meta


def smoothness(actions):
    d_steer = np.abs(np.diff(actions[:, 1]))
    d_pedal = np.abs(np.diff(actions[:, 0] - actions[:, 2]))
    return d_steer.mean(), d_pedal.mean()


def main():
    bc, bc_norm, _, _ = load_actor(BC_PATH)
    sac, sac_norm, sac_dim, meta = load_actor(SAC_PATH)
    has_prev = sac_dim >= 315                       # smooth wrapper appended prev action
    ema = float(meta.get('steer_ema', 0.0))
    cap = float(meta.get('smooth_steer', 0.0))
    print(f'SAC checkpoint: {SAC_PATH}')
    print(f'  obs_dim={sac_dim} prev_action={has_prev} steer_ema={ema} cap={cap} '
          f'bc_anchor={meta.get("bc_anchor", "n/a")} steps={meta.get("env_steps")}')

    # a contiguous 400-step slice well inside one episode
    with h5py.File(DATASET_PATH, 'r') as f:
        done = f['done'][:5000]
        start = int(np.flatnonzero(done)[0]) + 1
        obs = f['observations'][start:start + 400]
        expert_act = f['actions'][start:start + 400]

    # --- expert reference ---
    print(f'\nexpert (dataset)        : |dsteer|={smoothness(expert_act)[0]:.4f}  '
          f'|dpedal|={smoothness(expert_act)[1]:.4f}')

    # --- BC on the same obs ---
    with torch.no_grad():
        bc_act = bc.mean_action(torch.as_tensor(bc_norm(obs[:, :307]))).numpy()
    print(f'BC actor (307)          : |dsteer|={smoothness(bc_act)[0]:.4f}  '
          f'|dpedal|={smoothness(bc_act)[1]:.4f}')

    def assemble(o307, prev_act):
        parts = [o307, NEUTRAL_FEATURES] + ([prev_act] if has_prev else [])
        return np.concatenate(parts).astype(np.float32)

    # --- SAC on the same obs, autoregressive prev-action + (EMA/cap if any) ---
    prev = np.zeros(3, dtype=np.float32)
    sac_raw, sac_applied = [], []
    for o in obs:
        full = assemble(o[:307], prev)
        with torch.no_grad():
            a = sac.mean_action(torch.as_tensor(sac_norm(full)).unsqueeze(0)).squeeze(0).numpy()
        sac_raw.append(a.copy())
        applied = a.copy()
        if ema > 0.0:
            applied[1] = ema * prev[1] + (1.0 - ema) * a[1]
        if cap > 0.0:
            applied[1] = np.clip(applied[1], prev[1] - cap, prev[1] + cap)
        sac_applied.append(applied.copy())
        prev = applied
    sac_raw = np.array(sac_raw)
    sac_applied = np.array(sac_applied)
    print(f'SAC RAW network output  : |dsteer|={smoothness(sac_raw)[0]:.4f}  '
          f'|dpedal|={smoothness(sac_raw)[1]:.4f}   <- the real test (vs BC 0.018, expert 0.010)')
    if ema > 0.0 or cap > 0.0:
        print(f'SAC after EMA/cap       : |dsteer|={smoothness(sac_applied)[0]:.4f}  '
              f'|dpedal|={smoothness(sac_applied)[1]:.4f}')

    # --- Q2: sensitivity. perturb one obs block with small noise, 200 draws ---
    print('\nsensitivity (action std under small obs noise on one frame):')
    base = obs[200, :307]
    with h5py.File(DATASET_PATH, 'r') as f:
        std307 = f['observations'][start:start + 2000].std(axis=0)
    blocks = {'ego(0:9)': (0, 9), 'lidar(11:251)': (11, 251),
              'nearby(251:271)': (251, 271), 'waypoints(271:307)': (271, 307),
              'ALL(0:307)': (0, 307)}
    rng = np.random.default_rng(0)
    for label, (lo, hi) in blocks.items():
        outs = []
        for _ in range(200):
            o = base.copy()
            o[lo:hi] = o[lo:hi] + rng.normal(0, 0.1 * std307[lo:hi], hi - lo)
            full = assemble(o, np.zeros(3, np.float32))
            with torch.no_grad():
                outs.append(sac.mean_action(torch.as_tensor(sac_norm(full)).unsqueeze(0)).squeeze(0).numpy())
        outs = np.array(outs)
        print(f'  {label:18s} steer_std={outs[:,1].std():.4f}  pedal_std={(outs[:,0]-outs[:,2]).std():.4f}')

    # --- raw obs step-to-step jumpiness (is the INPUT noisy?) ---
    print('\nraw obs step-to-step |delta| (normalized units, dataset):')
    n = sac_norm(np.concatenate([obs[:, :307],
                                 np.zeros((len(obs), sac_dim - 307), np.float32)], axis=1))
    for label, (lo, hi) in [('ego', (0, 9)), ('lidar', (11, 251)),
                            ('nearby', (251, 271)), ('waypoints', (271, 307))]:
        dd = np.abs(np.diff(n[:, lo:hi], axis=0)).mean()
        print(f'  {label:10s} mean|dnorm|={dd:.4f}')


if __name__ == '__main__':
    main()
