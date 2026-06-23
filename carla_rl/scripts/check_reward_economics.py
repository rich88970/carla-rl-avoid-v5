"""Reward-economics assertions: every failure mode we have hit, as math.

Run this for ANY new preset before training on it. History:
  stage-2 collapse: optimum was negative -> suicide beat living
  stage-6 exploit:  low-v_des zones paid like full speed -> slow-farming
  stage-7b collapse: (v_des/target)^2 bump made correct reduced-speed driving
                     negative -> suicide returned in dense traffic

Usage: python -m carla_rl.scripts.check_reward_economics [preset]
"""

import sys

import numpy as np

from carla_rl.configs.reward_config import (
    REWARD_PRESETS,
    cruise_desired_speed,
    cruise_shaping,
    speed_tent,
)

ENV_DESIRED = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0  # env desired_speed
LANE_COST = 0.5        # typical env lane-deviation + smoothness cost per step


def total_per_step(v, gap, cfg, curve=0.0, rel_speed=0.0):
    """env speed tent + shaping delta - typical lane cost = realistic per-step."""
    delta, _ = cruise_shaping(v, gap, 0.5, 0.5, cfg, ENV_DESIRED,
                              curve=curve, rel_speed=rel_speed)
    return float(speed_tent(v, ENV_DESIRED) + delta - LANE_COST)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'cruise5'
    cfg = REWARD_PRESETS[name]
    failures = []

    # scenario grid: (label, v_des-inducing state, expected sign/ordering)
    straight = total_per_step(cfg.target_speed, 0.0, cfg)
    apex = total_per_step(
        float(cruise_desired_speed(0.0, cfg, 0.43)), 0.0, cfg, curve=0.43)
    follow = total_per_step(
        float(cruise_desired_speed(12.0, cfg)), 12.0, cfg)
    jam = total_per_step(0.0, 4.0, cfg)
    idle_street = total_per_step(0.3, 0.0, cfg)

    print(f'preset {name}:')
    for label, val in [('straight @target', straight), ('curve apex (0.43 rad)', apex),
                       ('following @12m', follow), ('jam stop @4m', jam),
                       ('idle on empty street', idle_street)]:
        print(f'  {label:26s} {val:+8.2f} /step')

    # 1. NO-SUICIDE: every CORRECT driving state must clearly beat dying.
    #    (collision = -100 once; living must be >= 0 net wherever driving is right)
    for label, val in [('straight', straight), ('curve apex', apex), ('following', follow)]:
        if val < 0.0:
            failures.append(f'no-suicide violated: {label} = {val:+.2f}/step')
    if jam < -1.0:
        failures.append(f'jam stop too punitive ({jam:+.2f}) - forced stops happen')

    # 2. NO-SLOW-FARM: full-speed straights must pay strictly best.
    for label, val in [('curve apex', apex), ('following', follow), ('jam', jam)]:
        if val >= straight - 0.5:
            failures.append(f'slow-farm risk: {label} ({val:+.2f}) ~ straight ({straight:+.2f})')

    # 3. NO-IDLE-FARM: idling on an empty street must be clearly worse than driving.
    if idle_street > 0.0:
        failures.append(f'idle-farm risk: idle pays {idle_street:+.2f}')

    # 4. v_des monotone in gap and curve (sanity of the target itself)
    gaps = np.array([0.0, 6.0, 10.0, 16.0, 20.0])
    vd = cruise_desired_speed(gaps, cfg)
    if not np.all(np.diff(vd[1:]) >= -1e-9):
        failures.append(f'v_des not monotone in gap: {vd}')

    if failures:
        print('\nFAILED:')
        for f in failures:
            print(' -', f)
        sys.exit(1)
    print('\nALL ECONOMICS CHECKS PASSED')


if __name__ == '__main__':
    main()
