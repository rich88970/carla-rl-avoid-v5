"""V10 imitation: collect demonstrations from the FAST autopilot (ego TM +30%
over the limit -> ~8 m/s free, 0.033 collision). Mirrors the evaluator's
autopilot path (set_autopilot True; record obs + the TM's control each step;
step with that control to tick + advance). Saves (obs_317, action_3) to
data/autopilot_fast.npz, incrementally so a server crash never loses progress.

Usage: python -m carla_rl.scripts.collect_autopilot [episodes]
"""

import sys
from pathlib import Path

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.utils.server import ensure_server, restart_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv

OUT = Path(__file__).resolve().parents[1].parent / 'data' / 'autopilot_fast.npz'
SPEED_PCT = -30.0   # ego TM percentage-speed-difference (negative = faster than limit)


def _make_env(params):
    # use_route=False: the TM autopilot follows lanes, so greedy waypoints match
    # its path (route-conditioned obs would mismatch what the demonstrator does).
    return CarlaGymEnv(params, use_route=False, predictive_obs=True)


def _set_fast(env, port):
    import carla
    env.ego.set_autopilot(True)
    tm = carla.Client('localhost', port).get_trafficmanager(8000)
    tm.vehicle_percentage_speed_difference(env.ego, SPEED_PCT)


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    params = make_params(number_of_vehicles=50, desired_speed=10, max_time_episode=1000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)
    env = _make_env(params)

    obs_buf, act_buf = [], []
    ep = 0
    restarts = 0
    while ep < episodes:
        try:
            obs = env.reset()
            _set_fast(env, params['port'])
            done, n = False, 0
            ep_obs, ep_act = [], []
            while not done:
                ctrl = env.ego.get_control()
                a = np.array([ctrl.throttle, ctrl.steer, ctrl.brake], dtype=np.float32)
                ep_obs.append(np.asarray(obs, dtype=np.float32))
                ep_act.append(a)
                obs, _, done, info = env.step(a)
                n += 1
            # only keep episodes that actually moved (autopilot occasionally
            # spawns blocked); skip a near-stationary or instant-collision ep
            if n >= 30:
                obs_buf.extend(ep_obs)
                act_buf.extend(ep_act)
            ep += 1
            print(f'ep {ep}/{episodes}: {n} steps, total transitions {len(obs_buf)}', flush=True)
            if ep % 10 == 0 and obs_buf:
                np.savez(OUT, obs=np.asarray(obs_buf), act=np.asarray(act_buf))
        except RuntimeError as exc:
            restarts += 1
            print(f'[collect] CARLA error: {exc}; restart {restarts}', flush=True)
            try:
                env.close()
            except Exception:
                pass
            restart_server(port=params['port'], nullrhi=True)
            env = _make_env(params)

    env.close()
    np.savez(OUT, obs=np.asarray(obs_buf), act=np.asarray(act_buf))
    print(f'COLLECT DONE: {len(obs_buf)} transitions from {episodes} eps, '
          f'{restarts} restarts -> {OUT}', flush=True)


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
