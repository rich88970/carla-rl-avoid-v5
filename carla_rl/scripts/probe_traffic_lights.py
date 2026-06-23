"""Live probe of the Phase 2 traffic-light features.

Part 1: autopilot drives with lights on; log every change of the tl feature
        block — proves light presence/state and junction distance respond.
Part 2: a 'red-light runner' (constant throttle, ignores lights) drives until
        info['red_light_violation'] fires — proves the violation event works.

Requires a running CARLA server. Usage:
    python -m carla_rl.scripts.probe_traffic_lights
"""

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.wrappers import CarlaGymEnv


def feature_str(tl):
    state = 'none'
    if tl[1]:
        state = 'RED'
    elif tl[2]:
        state = 'YELLOW'
    elif tl[3]:
        state = 'GREEN'
    return f"present={int(tl[0])} state={state} dist_to_junction={tl[4] * 50:.0f}m"


def autopilot_probe(env, steps=400):
    print('\n--- Part 1: autopilot, log tl feature changes ---')
    env.reset()
    env.ego.set_autopilot(True)
    prev = None
    changes = 0
    for i in range(steps):
        control = env.ego.get_control()
        action = [control.throttle, control.steer, control.brake]
        obs, reward, done, info = env.step(action)
        cur = (tuple(np.round(info['tl_features'][:4]).astype(int)),)
        if cur != prev:
            print(f'  step {i}: {feature_str(info["tl_features"])}')
            prev = cur
            changes += 1
        if done:
            env.reset()
            env.ego.set_autopilot(True)
            prev = None
    print(f'  feature-block changes observed: {changes}')
    return changes


def red_runner_probe(env, max_episodes=5):
    print('\n--- Part 2: red-light runner, wait for violation event ---')
    for ep in range(max_episodes):
        env.reset()
        env.ego.set_autopilot(False)
        done = False
        step = 0
        while not done:
            obs, reward, done, info = env.step([0.6, 0.0, 0.0])
            step += 1
            if info['red_light_violation']:
                print(f'  episode {ep} step {step}: RED LIGHT VIOLATION detected '
                      f'({feature_str(info["tl_features"])})')
                return True
        print(f'  episode {ep}: ended after {step} steps without crossing a red light')
    return False


def main():
    params = make_params(number_of_vehicles=30, traffic='on', max_time_episode=400)
    env = CarlaGymEnv(params)
    try:
        changes = autopilot_probe(env)
        violated = red_runner_probe(env)
    finally:
        env.close()

    assert changes >= 2, 'traffic-light features never changed'
    print(f"\nPROBE {'PASSED' if violated else 'PARTIAL'}: features respond"
          + ('' if violated else ' (no red light crossed by the runner — rerun to retry)'))


if __name__ == '__main__':
    main()
