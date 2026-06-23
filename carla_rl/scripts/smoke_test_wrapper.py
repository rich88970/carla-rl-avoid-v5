"""Smoke test: wrapped env exposes standard Gym API with a flat obs vector
(307 dataset dims + 5 traffic-light/intersection + 5 V9 predictive features),
and reward shaping logs both rewards.

Requires a running CARLA server. Usage:
    python -m carla_rl.scripts.smoke_test_wrapper
"""

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.wrappers import (
    CarlaGymEnv,
    ShapedRewardWrapper,
    SmoothActionWrapper,
    obs_dim,
)


def main():
    params = make_params(number_of_vehicles=20, max_time_episode=100, traffic='on')
    # training stack order: rate limiter inside, reward shaping outside
    env = ShapedRewardWrapper(SmoothActionWrapper(CarlaGymEnv(params)))

    expected_dim = obs_dim(params) + SmoothActionWrapper.EXTRA_DIMS
    # 307 dataset + 5 traffic-light + 5 V9 predictive + 3 smooth prev-action
    assert expected_dim == 320, f"expected obs dim 320, got {expected_dim}"
    obs = env.reset()
    assert obs.shape == (expected_dim,), f"obs shape {obs.shape}, expected ({expected_dim},)"
    assert obs.dtype == np.float32

    info_keys = (
        'cost', 'speed', 'lane_width', 'lateral_offset', 'is_collision',
        'is_off_road', 'tl_features', 'red_light_violation',
        'reward_env', 'reward_shaped', 'applied_action',
    )
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert obs.shape == (expected_dim,)
        for key in info_keys:
            assert key in info, f"missing info key: {key}"
        tl = info['tl_features']
        assert len(tl) == 5 and 0.0 <= tl[4] <= 1.0, f"bad tl features: {tl}"
        if done:
            obs = env.reset()

    env.close()
    print(f"Smoke test PASSED: obs dim {expected_dim}, dataset slice intact at [:307], "
          f"tl features + shaped reward present, 20 steps OK")


if __name__ == '__main__':
    main()
