"""Live check that the lookahead traffic-light obs works: drive forward and log
obs[307:311] (present/red/yellow/green) + distance vs the real at-light state.

Confirms `upcoming_light` sees a light AHEAD (present=1, a color set, distance
shrinking on approach) BEFORE the ego is at it — the prerequisite for learning
to stop. Run (server auto-started):
    .\.venv\Scripts\python.exe -m carla_rl.scripts.probe_lookahead_light
"""

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.utils.bigstack import run_with_big_stack
from carla_rl.utils.server import ensure_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv


def main():
    params = make_params(town='Town03', number_of_vehicles=10, traffic='on',
                         max_time_episode=400)
    ensure_server(port=params['port'], town=params['town'])
    env = CarlaGymEnv(params, predictive_obs=True, risk_obs=True)
    obs = env.reset()
    seen_present = seen_red = 0
    for t in range(400):
        # 直行油門(轉向交給 env 內部 None→0),只為了讓車前進去遇到燈
        obs, reward, done, info = env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
        tl = obs[307:312]
        present, red, yel, grn, dist = tl
        if present >= 0.5:
            seen_present += 1
            if red >= 0.5:
                seen_red += 1
            if t % 5 == 0 or red >= 0.5:
                color = 'RED' if red >= 0.5 else 'YEL' if yel >= 0.5 else 'GRN'
                print(f't={t:3d} speed={obs[3]:4.1f} AHEAD={color} dist={dist*50:5.1f}m '
                      f'red_violation={info.get("red_light_violation")}')
        if done:
            print(f'  [episode done at t={t}; resetting]')
            obs = env.reset()
    print(f'\nSUMMARY: steps with a light AHEAD={seen_present}, of which RED={seen_red}')
    print('PASS: lookahead sees lights ahead at distance' if seen_present > 0
          else 'FAIL: never saw a light ahead — upcoming_light query is broken')
    env.close()


if __name__ == '__main__':
    run_with_big_stack(main)
