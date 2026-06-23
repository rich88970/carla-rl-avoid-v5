"""Loader for the EasyCarla offline dataset (HDF5).

The data is a mix of expert (~80%) and random (~20%) policies with no source
label, so for BC we filter by per-episode return: keep episodes above a return
percentile. Random-policy episodes score far below expert ones (Phase 1
baselines: random mean -127 vs autopilot +1535), so a low percentile cleanly
drops them.
"""

from pathlib import Path

import h5py
import numpy as np

from carla_rl.configs.reward_config import cruise_shaping

DATASET_PATH = Path(__file__).resolve().parents[2] / 'data' / 'easycarla_offline_dataset.hdf5'


def episode_slices(done):
    """List of (start, end_exclusive) episode index ranges."""
    ends = np.flatnonzero(done)
    slices = []
    start = 0
    for end in ends:
        slices.append((start, end + 1))
        start = end + 1
    if start < len(done):
        slices.append((start, len(done)))  # trailing partial episode
    return slices


def load_bc_data(path=DATASET_PATH, expert_percentile=20.0, val_fraction=0.05,
                 drop_idle=True, seed=0):
    """Returns (train_obs, train_act, val_obs, val_act, stats dict).

    expert_percentile: drop episodes whose return is below this percentile.
    drop_idle: remove unjustified standstill frames — stopped (speed < 0.5)
      with no front vehicle within 10 m and a brake-style action. ~19% of the
      dataset is standstill with mean brake 0.7; cloning it verbatim creates a
      stationary attractor (the policy brakes at spawn and never moves).
      Justified stops (front vehicle near) and stop->go recoveries
      (throttle-style actions at standstill) are kept.
    """
    with h5py.File(path, 'r') as f:
        obs = f['observations'][:]
        act = f['actions'][:]
        rew = f['rewards'][:]
        done = f['done'][:]

    slices = episode_slices(done)
    returns = np.array([rew[s:e].sum() for s, e in slices])
    threshold = np.percentile(returns, expert_percentile)
    kept = [(s, e) for (s, e), ret in zip(slices, returns) if ret >= threshold]

    idx = np.concatenate([np.arange(s, e) for s, e in kept])
    obs, act = obs[idx], act[idx]

    n_idle_dropped = 0
    if drop_idle:
        speed = obs[:, 3]            # ego_state[3]
        front_dist = obs[:, 7]       # ego_state[7]; 0.0 = none within 20 m
        stopped = speed < 0.5
        unjustified = (front_dist == 0.0) | (front_dist > 10.0)
        idle_action = (act[:, 2] > 0.2) | (act[:, 0] < 0.2)  # braking or coasting
        drop = stopped & unjustified & idle_action
        n_idle_dropped = int(drop.sum())
        obs, act = obs[~drop], act[~drop]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(obs))
    n_val = int(len(obs) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    stats = {
        'episodes_total': len(slices),
        'episodes_kept': len(kept),
        'return_threshold': float(threshold),
        'idle_frames_dropped': n_idle_dropped,
        'transitions_kept': len(obs),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
    }
    return obs[train_idx], act[train_idx], obs[val_idx], act[val_idx], stats


def iter_prefill_transitions(path=DATASET_PATH, limit=100_000,
                             steer_jerk_weight=0.0, seed=0,
                             config=None, env_desired_speed=8.0):
    """Yield (obs307, act, reward, next_obs307, done) tuples for replay prefill,
    sampled from random episode positions. Steer-jerk shaping is recomputed
    within episodes so prefilled rewards match the live shaped reward (red-light
    and progress terms are zero under the dataset's collection settings).

    config: optional RewardConfig; when its cruise terms are active the reward
    is additionally relabeled with cruise_shaping() — computed from
    next_observations (the env derived rewards[i] from the post-step obs) and
    pre-exclusion pedal commands. env_desired_speed is the dataset's
    collection setting (8 m/s), NOT the live env's params.
    """
    with h5py.File(path, 'r') as f:
        n = f['observations'].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(limit, n), replace=False))
        obs = f['observations'][idx]
        act = f['actions'][idx]
        rew = f['rewards'][idx]
        next_obs = f['next_observations'][idx]
        done = f['done'][idx]
        # previous action's steer within the same episode (approximate: the
        # sampled index's predecessor, valid unless it crosses an episode edge)
        prev_idx = np.maximum(idx - 1, 0)
        prev_act = f['actions'][prev_idx]
        prev_steer = prev_act[:, 1]
        prev_done = f['done'][prev_idx]

    if steer_jerk_weight != 0.0:
        jerk = np.abs(act[:, 1] - np.where(prev_done, act[:, 1], prev_steer))
        rew = rew + steer_jerk_weight * jerk

    if config is not None and (config.replace_speed_term or config.speed_weight != 0.0
                               or config.pedal_jerk_weight != 0.0):
        from carla_rl.wrappers.lead_vehicle import path_aware_lead

        # the env computed rewards[i] from next_observations[i] (post-step obs)
        v = next_obs[:, 3]
        pedal = act[:, 0] - act[:, 2]
        prev_pedal = np.where(prev_done, pedal, prev_act[:, 0] - prev_act[:, 2])
        curve = np.abs(next_obs[:, 288])  # ~10 m-lookahead waypoint rel yaw
        # path-aware lead, identical function to the live wrapper
        gap, lead_speed = path_aware_lead(next_obs)
        rel_speed = np.where(gap > 0.0, v - lead_speed, 0.0)
        # yaw flip: post-step yaw rate vs pre-step yaw rate (same transition —
        # no neighbor indexing, exact at episode boundaries too)
        delta, _ = cruise_shaping(v, gap, pedal, prev_pedal, config,
                                  env_desired_speed, curve=curve,
                                  rel_speed=rel_speed,
                                  yaw_rate=next_obs[:, 4], prev_yaw_rate=obs[:, 4])
        rew = rew + delta

    # previous APPLIED action for the prev-action observation feature; zeros
    # at episode starts (matches SmoothActionWrapper.reset). next-state prev
    # action is simply this step's action.
    prev_act_feat = np.where(prev_done[:, None], 0.0, prev_act).astype(np.float32)

    for i in range(len(obs)):
        yield obs[i], act[i], float(rew[i]), next_obs[i], bool(done[i]), prev_act_feat[i]
