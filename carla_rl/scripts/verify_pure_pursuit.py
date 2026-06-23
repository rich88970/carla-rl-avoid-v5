"""Verify the geometric lateral controller in isolation (no RL).

Drives at a constant target speed with a trivial P throttle controller while
PurePursuitController does ALL the steering. If body weave
(straight_mean_abs_yaw_rate) drops near autopilot's 0.92 deg/s, the lateral
architecture is sound and RL only needs to learn longitudinal control.

Usage: python -m carla_rl.scripts.verify_pure_pursuit [gain]
"""

import sys

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.evaluation.evaluator import evaluate
from carla_rl.utils.server import ensure_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import HybridSteerWrapper, PurePursuitController

GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
LOOKAHEAD_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
LOOKAHEAD_K = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
TARGET = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0


class ConstantSpeedPolicy:
    def on_episode_start(self, env):
        env.ego.set_autopilot(False)

    def act(self, obs, env):
        v = float(obs[3])
        throttle = float(np.clip(0.4 * (TARGET - v), 0.0, 0.6))
        brake = float(np.clip(0.4 * (v - TARGET), 0.0, 0.5)) if v > TARGET + 1.0 else 0.0
        return np.array([throttle, 0.0, brake], dtype=np.float32)  # steer ignored


def main():
    params = make_params(number_of_vehicles=50, max_time_episode=1000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)
    ctrl = PurePursuitController(gain=GAIN, lookahead_min=LOOKAHEAD_MIN,
                                 lookahead_k=LOOKAHEAD_K)
    print(f'pure-pursuit verify: gain={GAIN}, lookahead_min={LOOKAHEAD_MIN}, '
          f'k={LOOKAHEAD_K}, target={TARGET} m/s, 50 vehicles')
    _, summary = evaluate(
        lambda: HybridSteerWrapper(CarlaGymEnv(params), ctrl),
        ConstantSpeedPolicy(), episodes=5,
        out_dir=r'carla_rl\logs\pp_verify', run_name='pp_verify',
        record_first_n=0, nullrhi=True,
    )
    print('\n=== pure-pursuit lateral verification ===')
    for k in ['success_rate', 'collision_rate', 'off_road_rate', 'mean_steps',
              'mean_avg_speed', 'straight_mean_abs_yaw_rate', 'yaw_flip_rate',
              'straight_abs_steer_delta']:
        print(f'  {k}: {round(summary[k], 4)}')
    print(f"  (autopilot reference straight_yaw 0.92; bar <= 1.85)")


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
