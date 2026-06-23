"""Aggregate the fast-autopilot demos with recovery-DAgger corrections (V11).

The DAgger set is small but high-value: corrective (obs, action) pairs in the
student's drifted (over-speed) state distribution that the base demos never
visit. We OVERSAMPLE it so it carries real weight against the ~46k base demos —
otherwise BC retraining barely moves. Oversample is a knob: too low and the
correction washes out, too high and BC overfits to braking and drives timid.

Usage: python -m carla_rl.scripts.aggregate_dagger <base.npz> <dagger.npz> <out.npz> [oversample]
"""

import sys

import numpy as np


def main():
    base_path = sys.argv[1] if len(sys.argv) > 1 else r'data\autopilot_fast.npz'
    dagger_path = sys.argv[2] if len(sys.argv) > 2 else r'data\dagger_iter1.npz'
    out = sys.argv[3] if len(sys.argv) > 3 else r'data\dagger_aggregated_iter1.npz'
    oversample = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    base = np.load(base_path)
    dag = np.load(dagger_path)
    b_obs, b_act = base['obs'].astype(np.float32), base['act'].astype(np.float32)
    d_obs, d_act = dag['obs'].astype(np.float32), dag['act'].astype(np.float32)
    assert b_obs.shape[1] == d_obs.shape[1], \
        f'obs width mismatch: base {b_obs.shape[1]} vs dagger {d_obs.shape[1]}'

    obs = np.concatenate([b_obs] + [d_obs] * oversample, axis=0)
    act = np.concatenate([b_act] + [d_act] * oversample, axis=0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    obs, act = obs[perm], act[perm]

    np.savez(out, obs=obs, act=act)
    dag_eff = len(d_obs) * oversample
    print(f'base {len(b_obs)} + dagger {len(d_obs)}x{oversample}={dag_eff} '
          f'-> {len(obs)} total ({100 * dag_eff / len(obs):.1f}% dagger) -> {out}')
    # quick look at what the corrections teach: mean pedal of the dagger labels
    print(f'dagger mean throttle={d_act[:, 0].mean():.3f} brake={d_act[:, 2].mean():.3f} '
          f'| base mean throttle={b_act[:, 0].mean():.3f} brake={b_act[:, 2].mean():.3f}')


if __name__ == '__main__':
    main()
