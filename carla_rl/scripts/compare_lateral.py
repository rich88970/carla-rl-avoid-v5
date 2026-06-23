"""Compare lateral controllers on turning quality (the user-reported gap),
all at constant speed with RL off, vs the autopilot reference. The new
mean_abs_yaw_jerk captures cornering roughness the straight-only metric missed.

Usage: python -m carla_rl.scripts.compare_lateral
"""

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.evaluation.evaluator import AutopilotPolicy, evaluate
from carla_rl.utils.server import ensure_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import (
    HybridSteerWrapper,
    PurePursuitController,
    StanleyController,
)

TARGET = 6.5


class ConstantSpeedPolicy:
    def on_episode_start(self, env):
        try:
            env.ego.set_autopilot(False)
        except Exception:
            pass

    def act(self, obs, env):
        v = float(obs[3])
        t = float(np.clip(0.4 * (TARGET - v), 0.0, 0.6))
        b = float(np.clip(0.4 * (v - TARGET), 0.0, 0.5)) if v > TARGET + 1 else 0.0
        return np.array([t, 0.0, b], dtype=np.float32)


def run(name, factory, policy, n=5):
    _, s = evaluate(factory, policy, n, rf'carla_rl\logs\lat_{name}', name,
                    record_first_n=0, nullrhi=True)
    return s


def main():
    params = make_params(number_of_vehicles=50, max_time_episode=1000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)
    res = {}
    res['autopilot'] = run('autopilot', lambda: CarlaGymEnv(params), AutopilotPolicy())
    res['pure_pursuit'] = run(
        'pp', lambda: HybridSteerWrapper(CarlaGymEnv(params), PurePursuitController()),
        ConstantSpeedPolicy())
    res['stanley'] = run(
        'stanley', lambda: HybridSteerWrapper(CarlaGymEnv(params), StanleyController()),
        ConstantSpeedPolicy())

    print('\n=== LATERAL CONTROLLER COMPARISON (5 eps each, 50 veh, const speed) ===')
    print(f"{'controller':13s} {'yaw_jerk':>9s} {'str_yaw':>8s} {'off_road':>9s} "
          f"{'success':>8s} {'steps':>6s}")
    for k, s in res.items():
        print(f"{k:13s} {s['mean_abs_yaw_jerk']:9.3f} "
              f"{s['straight_mean_abs_yaw_rate']:8.3f} {s['off_road_rate']:9.2f} "
              f"{s['success_rate']:8.2f} {s['mean_steps']:6.0f}")
    print('LATERAL COMPARE DONE')


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
