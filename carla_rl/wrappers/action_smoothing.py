"""Steer rate limiter + previous-action observation.

Stage-1 SAC oscillated badly on straights (applied |dsteer| 0.31/step vs
autopilot 0.007). Two structural causes, both fixed here:

  - nothing physically prevented per-step steering jumps -> the per-step
    steering change is now clipped to max_steer_delta (full lock still takes
    only ~1-2 s at 10 Hz, so cornering is unaffected)
  - the smoothness penalty depends on the previous action, which the policy
    could not observe (non-Markov reward) -> the previously APPLIED action
    (post-limit, pre pedal-exclusion) is appended to the observation (3 dims)

The appended dims go AFTER everything else, so obs[:307] stays
dataset-compatible and obs[:312] keeps the traffic-light layout.
info['applied_action'] reports what actually drove the car; metrics use it
for honest smoothness numbers.
"""

import gym
import numpy as np


class SmoothActionWrapper(gym.Wrapper):
    EXTRA_DIMS = 3

    def __init__(self, env, max_steer_delta=0.1, steer_ema=0.0):
        """steer_ema: low-pass gain on steering, applied BEFORE the rate cap.
        applied = (1-g)*prev + g*raw with g = 1-steer_ema. The rate cap bounds
        AMPLITUDE per step; the EMA kills FREQUENCY — v2 proved that reward
        penalties + cap alone cannot beat bang-bang lane centering (straight
        yaw 6.5 deg/s at the cap), because zigzag is near-optimal for bounded
        tracking. Filtering makes high-frequency policy output physically
        unreachable. Pedals are NOT filtered (emergency braking must be
        instant). The filter state (prev applied action) is already in the
        observation, so the MDP stays Markov."""
        super().__init__(env)
        self.max_steer_delta = float(max_steer_delta)
        self.steer_ema = float(steer_ema)
        self._prev = np.zeros(3, dtype=np.float32)
        low = np.concatenate(
            [env.observation_space.low, np.array([0.0, -1.0, 0.0], dtype=np.float32)]
        )
        high = np.concatenate(
            [env.observation_space.high, np.array([1.0, 1.0, 1.0], dtype=np.float32)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        self._prev = np.zeros(3, dtype=np.float32)
        obs = self.env.reset(**kwargs)
        return np.concatenate([obs, self._prev])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self.steer_ema > 0.0:
            action[1] = (self.steer_ema * self._prev[1]
                         + (1.0 - self.steer_ema) * action[1])
        action[1] = np.clip(
            action[1],
            self._prev[1] - self.max_steer_delta,
            self._prev[1] + self.max_steer_delta,
        )
        self._prev = action
        obs, reward, done, info = self.env.step(action)
        info['applied_action'] = action.copy()
        return np.concatenate([obs, action]), reward, done, info
