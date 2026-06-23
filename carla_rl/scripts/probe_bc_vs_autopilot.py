"""Decisive BC diagnostic: autopilot drives, BC predicts in parallel.

Compares BC's predicted actions against the expert's on LIVE states, and
flags live observation dims that are far outside the dataset distribution.

If BC tracks the expert here, the live failure is compounding drift (BC's
classic weakness, fixable by SAC fine-tuning). If it diverges immediately,
the live observation pipeline mismatches the dataset.
"""

import numpy as np
import torch

from carla_rl.configs.env_params import make_params
from carla_rl.scripts.run_eval import BCPolicy
from carla_rl.wrappers import CarlaGymEnv


def main():
    policy = BCPolicy()
    norm = policy.normalizer

    params = make_params(number_of_vehicles=30, max_time_episode=400)
    env = CarlaGymEnv(params)

    bc_actions, ap_actions = [], []
    z_max = np.zeros(307)
    obs = env.reset()
    env.ego.set_autopilot(True)
    for step in range(350):
        control = env.ego.get_control()
        ap = np.array([control.throttle, control.steer, control.brake], dtype=np.float32)
        bc = policy.act(obs, env)
        ap_actions.append(ap)
        bc_actions.append(bc)

        z = np.abs((obs[:307] - norm.mean) / norm.std)
        z_max = np.maximum(z_max, z)

        obs, _, done, _ = env.step(ap)
        if done:
            obs = env.reset()
            env.ego.set_autopilot(True)
    env.close()

    bc_a = np.array(bc_actions)
    ap_a = np.array(ap_actions)
    moving = ap_a[:, 0] > 0.05  # compare while the expert is actually driving

    print(f'\nsteps compared: {len(bc_a)} (moving: {moving.sum()})')
    for i, name in enumerate(['throttle', 'steer', 'brake']):
        diff = np.abs(bc_a[moving, i] - ap_a[moving, i])
        corr = np.corrcoef(bc_a[moving, i], ap_a[moving, i])[0, 1]
        print(f'{name}: mean|bc-ap|={diff.mean():.3f} corr={corr:.3f} '
              f'(bc mean={bc_a[moving, i].mean():.3f}, ap mean={ap_a[moving, i].mean():.3f})')

    steer_big = moving & (np.abs(ap_a[:, 1]) > 0.1)
    if steer_big.any():
        slope = np.polyfit(ap_a[steer_big, 1], bc_a[steer_big, 1], 1)[0]
        print(f'live steer slope on |ap steer|>0.1 ({steer_big.sum()} frames): {slope:.3f}')

    worst = np.argsort(z_max)[::-1][:8]
    print('\nlive dims furthest outside dataset distribution (max |z|):')
    blocks = [('ego_state', 0, 9), ('lane_info', 9, 11), ('lidar', 11, 251),
              ('nearby', 251, 271), ('waypoints', 271, 307)]
    for d in worst:
        block = next(b for b, s, e in blocks if s <= d < e)
        print(f'  dim {d} ({block}): max|z|={z_max[d]:.1f}')


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
