"""Record a video of pure-pursuit steering + constant-speed throttle (RL off),
so the smoothness can be seen, not just measured. Rendered server."""

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.evaluation.evaluator import evaluate
from carla_rl.utils.server import restart_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import HybridSteerWrapper, PurePursuitController

TARGET = 6.0


class ConstantSpeedPolicy:
    def on_episode_start(self, env):
        env.ego.set_autopilot(False)

    def act(self, obs, env):
        v = float(obs[3])
        throttle = float(np.clip(0.4 * (TARGET - v), 0.0, 0.6))
        brake = float(np.clip(0.4 * (v - TARGET), 0.0, 0.5)) if v > TARGET + 1.0 else 0.0
        return np.array([throttle, 0.0, brake], dtype=np.float32)


def main():
    params = make_params(number_of_vehicles=50, max_time_episode=1000)
    restart_server(port=params['port'], town=params['town'], nullrhi=False)  # rendered
    evaluate(
        lambda: HybridSteerWrapper(CarlaGymEnv(params), PurePursuitController()),
        ConstantSpeedPolicy(), episodes=3,
        out_dir=r'carla_rl\logs\pp_demo_video', run_name='pp_demo',
        record_first_n=3, nullrhi=False,
    )
    print('PP DEMO VIDEOS DONE')


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
