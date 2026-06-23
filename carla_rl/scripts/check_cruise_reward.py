"""Offline verification of the cruise reward pipeline (no CARLA server, no torch).

Run from the repo root before any cruise training:
    .venv\\Scripts\\python.exe -u -m carla_rl.scripts.check_cruise_reward

Checks, in order:
  1. DEFAULT config is inert: cruise_shaping() is exactly zero on real data.
  2. Prefill regression: iter_prefill_transitions(config=DEFAULT) is
     byte-identical to the legacy call — the stage-1 supervisor can relaunch
     train_sac safely on this code.
  3./4. Cruise relabeling is finite and sane (term statistics printed);
     episode-boundary pedal jerk is exactly zero.
  5. ShapedRewardWrapper reproduces the legacy formula exactly under DEFAULT
     and matches an independent recomputation under CRUISE, on a fake env.

No torch + no CARLA client connection -> bigstack not required (importing the
carla module via carla_rl.wrappers is safe without a server).
"""

import gym
import h5py
import numpy as np

from carla_rl.configs.reward_config import (
    CRUISE_REWARD_CONFIG,
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    cruise_desired_speed,
    cruise_shaping,
)
from carla_rl.data.offline_dataset import DATASET_PATH, iter_prefill_transitions
from carla_rl.wrappers.reward_shaping import ShapedRewardWrapper


def check_default_inert():
    assert DEFAULT_REWARD_CONFIG == RewardConfig(), 'defaults drifted'
    with h5py.File(DATASET_PATH, 'r') as f:
        nxt = f['next_observations'][:50_000]
        act = f['actions'][:50_000]
    delta, _ = cruise_shaping(
        nxt[:, 3], nxt[:, 7], act[:, 0] - act[:, 2],
        np.zeros(len(act)), DEFAULT_REWARD_CONFIG, 8.0,
    )
    assert np.all(np.asarray(delta) == 0.0), 'DEFAULT config must add exactly zero'
    print('1. DEFAULT config inert on 50k real transitions: OK')


def check_prefill_regression():
    legacy = list(iter_prefill_transitions(limit=5_000, steer_jerk_weight=-0.5))
    gated = list(iter_prefill_transitions(limit=5_000, steer_jerk_weight=-0.5,
                                          config=DEFAULT_REWARD_CONFIG))
    assert len(legacy) == len(gated) == 5_000
    for (o1, a1, r1, n1, d1, p1), (o2, a2, r2, n2, d2, p2) in zip(legacy, gated):
        assert r1 == r2 and d1 == d2
        assert np.array_equal(o1, o2) and np.array_equal(a1, a2)
        assert np.array_equal(n1, n2) and np.array_equal(p1, p2)
    print('2. prefill(config=DEFAULT) byte-identical to legacy prefill (5k): OK')


def check_cruise_relabel():
    cfg = CRUISE_REWARD_CONFIG
    with h5py.File(DATASET_PATH, 'r') as f:
        n = f['observations'].shape[0]
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=100_000, replace=False))
        nxt = f['next_observations'][idx]
        act = f['actions'][idx]
        rew = f['rewards'][idx]
        prev_idx = np.maximum(idx - 1, 0)
        prev_act = f['actions'][prev_idx]
        prev_done = f['done'][prev_idx].astype(bool)
    v, gap = nxt[:, 3], nxt[:, 7]
    pedal = act[:, 0] - act[:, 2]
    prev_pedal = np.where(prev_done, pedal, prev_act[:, 0] - prev_act[:, 2])
    delta, terms = cruise_shaping(v, gap, pedal, prev_pedal, cfg, 8.0)
    relabeled = rew + delta

    assert np.all(np.isfinite(delta)), 'NaN/inf in cruise shaping'
    lead = gap > 0.0
    print(f'3. cruise relabel on 100k transitions (lead present {lead.mean():.1%}):')
    p = np.percentile(terms['v_des'], [10, 50, 90])
    print(f'   v_des p10/p50/p90 = {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f} m/s')
    for key in ('tent_removed', 'speed_track', 'low_speed', 'overspeed',
                'tailgate', 'pedal_jerk'):
        t = np.asarray(terms[key], dtype=np.float64)
        print(f'   {key:13s} mean {t.mean():+8.3f}   p10 {np.percentile(t, 10):+8.3f}'
              f'   p90 {np.percentile(t, 90):+8.3f}   active {np.mean(t != 0.0):6.1%}')
    print(f'   tailgate active among lead steps: '
          f'{np.mean(np.asarray(terms["tailgate"])[lead] != 0.0):.1%}')
    print(f'   reward mean {rew.mean():+.3f} -> {relabeled.mean():+.3f}   '
          f'min {relabeled.min():+.1f}   max {relabeled.max():+.1f}')

    assert np.all(np.asarray(terms['pedal_jerk'])[prev_done] == 0.0), \
        'episode-boundary pedal jerk must be zero'
    print('4. episode-boundary pedal jerk is zero: OK')


class _FakeEnv(gym.Env):
    """Stand-in for CarlaGymEnv: scripted (obs, reward, done, info) steps."""

    def __init__(self, steps):
        self.params = {'dt': 0.1, 'desired_speed': 8}
        self._steps = steps
        self._i = 0

    def reset(self, **kwargs):
        self._i = 0
        return np.zeros(312, dtype=np.float32)

    def step(self, action):
        obs, rew, done, info = self._steps[self._i]
        self._i += 1
        return obs, float(rew), bool(done), dict(info)


def _make_obs(v, gap):
    o = np.zeros(312, dtype=np.float32)
    o[3] = v
    o[7] = gap
    return o


def check_wrapper_equivalence():
    # info supplies the path-aware lead values, as CarlaGymEnv now does
    steps = [
        (_make_obs(6.5, 0.0), 5.0, False, {'front_gap': 0.0, 'lead_rel_speed': 0.0}),
        (_make_obs(6.0, 12.0), 4.0, False, {'front_gap': 12.0, 'lead_rel_speed': 0.0}),
        (_make_obs(2.0, 6.0), 1.0, False, {'front_gap': 6.0, 'lead_rel_speed': 0.0}),
        (_make_obs(7.8, 0.0), 6.0, True, {'front_gap': 0.0, 'lead_rel_speed': 0.0}),
    ]
    actions = [
        np.array([0.6, 0.10, 0.0], dtype=np.float32),
        np.array([0.2, -0.05, 0.4], dtype=np.float32),
        np.array([0.0, 0.20, 0.8], dtype=np.float32),
        np.array([0.9, 0.00, 0.0], dtype=np.float32),
    ]

    # DEFAULT: legacy formula, exactly
    env = ShapedRewardWrapper(_FakeEnv(steps), DEFAULT_REWARD_CONFIG)
    env.reset()
    prev_steer = 0.0
    for (obs_s, rew_s, _, _), action in zip(steps, actions):
        _, shaped, _, info = env.step(action)
        expect = rew_s + DEFAULT_REWARD_CONFIG.steer_jerk_weight * abs(
            float(action[1]) - prev_steer)
        assert shaped == expect, f'DEFAULT mismatch: {shaped} != {expect}'
        assert 'v_des' not in info, 'DEFAULT must not emit cruise info keys'
        prev_steer = float(action[1])

    # CRUISE: matches independent recomputation; first step pedal jerk 0
    cfg = CRUISE_REWARD_CONFIG
    env = ShapedRewardWrapper(_FakeEnv(steps), cfg)
    env.reset()
    prev_steer = 0.0
    prev_pedal = None
    for (obs_s, rew_s, _, _), action in zip(steps, actions):
        _, shaped, _, info = env.step(action)
        pedal = float(action[0]) - float(action[2])
        pp = pedal if prev_pedal is None else prev_pedal
        delta, terms = cruise_shaping(
            float(obs_s[3]), float(obs_s[7]), pedal, pp, cfg, 8.0,
            yaw_rate=float(obs_s[4]), prev_yaw_rate=0.0)
        expect = rew_s + cfg.steer_jerk_weight * abs(float(action[1]) - prev_steer) \
            + float(delta)
        assert abs(shaped - expect) < 1e-9, f'CRUISE mismatch: {shaped} != {expect}'
        assert abs(info['v_des'] - float(cruise_desired_speed(float(obs_s[7]), cfg))) < 1e-12
        assert 'reward_terms' in info and info['reward_env'] == rew_s
        prev_steer = float(action[1])
        prev_pedal = pedal
    print('5. wrapper equivalence (DEFAULT exact, CRUISE matches recompute): OK')


def main():
    check_default_inert()
    check_prefill_regression()
    check_cruise_relabel()
    check_wrapper_equivalence()
    print('\nALL CRUISE REWARD CHECKS PASSED')


if __name__ == '__main__':
    main()
