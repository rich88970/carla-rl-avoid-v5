"""Verify global-route following (no RL): constant-speed throttle + pure-pursuit
steering on the PLANNED route. The key signal is route_completion — if the car
follows a real destination route it should progress toward 1.0 and exit
roundabouts, instead of circling. 8 episodes to hit varied topology.

Usage: python -m carla_rl.scripts.verify_route
"""

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.utils.server import ensure_server, restart_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import HybridSteerWrapper, PurePursuitController

TARGET = 6.0


def act(obs):
    v = float(obs[3])
    t = float(np.clip(0.4 * (TARGET - v), 0.0, 0.6))
    b = float(np.clip(0.4 * (v - TARGET), 0.0, 0.5)) if v > TARGET + 1 else 0.0
    return np.array([t, 0.0, b], dtype=np.float32)


def main():
    params = make_params(number_of_vehicles=50, max_time_episode=1000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)
    env = HybridSteerWrapper(CarlaGymEnv(params, use_route=True), PurePursuitController())

    comps, offroads, colls, reached = [], [], [], []
    for ep in range(8):
        obs = env.reset()
        env.ego.set_autopilot(False)
        done, info, steps, comp = False, {}, 0, 0.0
        while not done:
            obs, r, done, info = env.step(act(obs))
            comp = info.get('route_completion', 0.0)
            steps += 1
        comps.append(comp)
        offroads.append(bool(info.get('is_off_road')))
        colls.append(bool(info.get('is_collision')))
        reached.append(bool(info.get('reached_destination')))
        print(f'ep {ep}: route_completion={comp:.2f} steps={steps} '
              f'reached={info.get("reached_destination")} '
              f'collided={info.get("is_collision")} off_road={info.get("is_off_road")}',
              flush=True)
    env.close()
    # arrival should now end the episode cleanly: reached -> not off_road
    print(f'\nmean route_completion={np.mean(comps):.2f}  '
          f'reached_destination={sum(reached)}/8  '
          f'collisions={sum(colls)}/8  off_road={sum(offroads)}/8')
    print('ROUTE VERIFY DONE')


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
