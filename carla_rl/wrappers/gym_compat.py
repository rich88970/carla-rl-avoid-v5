"""Standard-Gym wrapper around EasyCarla-RL's CarlaEnv.

CarlaEnv deviates from the Gym API in two ways:
  - observations are a dict of arrays (307 dims total)
  - step() returns a 5-tuple (obs, reward, cost, done, info)

This wrapper instantiates CarlaEnv directly (bypassing gym.make, whose passive
env checker chokes on the 5-tuple) and exposes the classic 4-tuple API with a
flat float32 observation vector. The safety cost goes into info['cost'].

With extended_obs=True (default) the vector gains 5 traffic-light/intersection
features appended AFTER the 307 dataset-compatible dims, so obs[:307] always
matches the offline dataset layout. Red-light violation events are emitted as
info['red_light_violation'].
"""

import gym
import numpy as np
from gym import spaces

from easycarla.envs import CarlaEnv

from carla_rl.wrappers.traffic_light import (
    N_FEATURES,
    RedLightMonitor,
    traffic_light_features,
)
from carla_rl.wrappers.predictive_features import (
    N_PRED_FEATURES,
    NEUTRAL_PRED_FEATURES,
    predictive_features,
)
from carla_rl.wrappers.risk_features import (
    N_RISK_FEATURES,
    NEUTRAL_RISK_FEATURES,
    risk_features,
    min_ttc,
)

# Concatenation order MUST match the offline dataset and the upstream DQL
# example (example/run_dql_in_carla.py), so models transfer between the two.
OBS_KEYS = ('ego_state', 'lane_info', 'lidar', 'nearby_vehicles', 'waypoints')

BASE_DIM_FIXED = 9 + 2 + 240  # ego_state + lane_info + lidar

# Treat the route as completed a few metres short of the final point: the last
# lookahead window collapses onto the destination there and the geometric
# controller would otherwise U-turn back toward it.
ROUTE_ARRIVE_FRAC = 0.95

# Anti-idle: end the episode (as a NON-success) if the car sits still on a CLEAR
# road (no lead within the 20 m corridor) this many steps. A car blocked by a
# lead — or, once traffic lights are added, stopped at a red — is legitimately
# waiting and is NOT counted as stuck (gap > 0 resets the counter).
STUCK_SPEED = 0.3   # m/s
STUCK_STEPS = 120   # = 12 s at dt=0.1


def flatten_obs(obs_dict):
    return np.concatenate(
        [np.asarray(obs_dict[k], dtype=np.float32).ravel() for k in OBS_KEYS]
    )


def obs_dim(params, extended=True, predictive=True, risk=True):
    base = BASE_DIM_FIXED + params['max_nearby_vehicles'] * 4 + params['max_waypoints'] * 3
    dim = base + (N_FEATURES if extended else 0)
    dim += (N_PRED_FEATURES if (extended and predictive) else 0)
    # RL-核心避撞:5 扇區風險特徵(10 維)只在 predictive 也開時附加在最後 → 327 維
    return dim + (N_RISK_FEATURES if (extended and predictive and risk) else 0)


class CarlaGymEnv(gym.Env):
    def __init__(self, params, extended_obs=True, use_route=False, predictive_obs=True,
                 risk_obs=True):
        self.params = dict(params)
        self.extended_obs = extended_obs
        self.use_route = use_route
        self.predictive_obs = predictive_obs
        self.risk_obs = risk_obs
        self.env = CarlaEnv(self.params)
        dim = obs_dim(self.params, extended=extended_obs, predictive=predictive_obs,
                      risk=risk_obs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )
        self.action_space = self.env.action_space
        self.last_obs_dict = None
        self._red_light_monitor = RedLightMonitor()
        self._world_map = self.env.world.get_map()
        self._route_planner = None
        self._route_ok = False
        self._route_completion = 0.0
        self._stuck_counter = 0

    # Passthroughs used by the evaluator / video recorder
    @property
    def ego(self):
        return self.env.ego

    @property
    def world(self):
        return self.env.world

    @property
    def time_step(self):
        return self.env.time_step

    def _build_obs(self, obs_dict, info=None):
        flat = flatten_obs(obs_dict)
        # overwrite the greedy waypoints with the planned route (goal-directed)
        if self.use_route and self._route_ok:
            tr = self.env.ego.get_transform()
            eyaw = np.deg2rad(tr.rotation.yaw)
            wps, comp = self._route_planner.waypoints(tr.location.x, tr.location.y, eyaw)
            if wps is not None:
                flat[271:307] = wps.flatten()
                self._route_completion = comp
                if info is not None:
                    info['route_completion'] = comp
                    # 0.95 (not ~1.0): the last ~12 route points collapse onto
                    # the destination, so the lookahead degenerates and the car
                    # would U-turn back toward it (-> wrong-way off-road). Call
                    # it arrived a few metres early and stop, before that.
                    info['reached_destination'] = comp >= ROUTE_ARRIVE_FRAC
        if not self.extended_obs:
            return flat
        features, at_light = traffic_light_features(self._world_map, self.env.ego)
        if info is not None:
            speed = float(obs_dict['ego_state'][3])
            info['tl_features'] = features.tolist()   # 前瞻燈號(供 obs + 接近紅燈減速懲罰)
            # 違規/終止:用「真正越過停止線」(monitor 內以 stop waypoints 判定,需 ego),與前瞻解耦
            info['red_light_violation'] = self._red_light_monitor.check(self.env.ego, speed)
        if not self.predictive_obs:
            return np.concatenate([flat, features])   # 312-dim (pre-V9 policies)
        # V9 predictive perception, appended AFTER the 312-dim obs (obs[:312]
        # untouched). Over ALL world vehicles within 40 m, in the ego frame.
        pred = self._predictive_features(flat, float(features[4]))
        if not self.risk_obs:
            return np.concatenate([flat, features, pred])  # 317-dim
        # RL-核心避撞:5 扇區風險特徵(10 維)附加在最後 → 327 維(obs[:317] 不動);供 RL
        # 直接學避撞,並把最小 TTC 放進 info 給 avoid_v1 獎勵用。
        rf = self._risk_features()
        if info is not None:
            info['min_ttc'] = min_ttc(rf)
        return np.concatenate([flat, features, pred, rf])

    def _predictive_features(self, flat, junction_dist):
        try:
            tr = self.env.ego.get_transform()
            eyaw = np.deg2rad(tr.rotation.yaw)
            c, s = np.cos(-eyaw), np.sin(-eyaw)
            ex, ey = tr.location.x, tr.location.y
            rows = []
            for v in self.env.world.get_actors().filter('vehicle.*'):
                if v.id == self.env.ego.id:
                    continue
                loc = v.get_transform().location
                dx, dy = loc.x - ex, loc.y - ey
                if dx * dx + dy * dy > 1600.0:          # 40 m
                    continue
                lx = c * dx - s * dy
                ly = s * dx + c * dy
                vyaw = np.deg2rad(v.get_transform().rotation.yaw)
                vel = v.get_velocity()
                rows.append([lx, ly, vyaw - eyaw, float(np.hypot(vel.x, vel.y))])
            route_xy = flat[271:307].reshape(12, 3)[:, :2]
            return predictive_features(
                np.asarray(rows, dtype=np.float64).reshape(-1, 4), route_xy, junction_dist
            )
        except Exception:
            return NEUTRAL_PRED_FEATURES

    def _risk_features(self):
        """蒐集 40 m 內所有世界車輛(ego frame)→ risk_features 10 維(5 扇區 × TTC/距離)。失敗回中性值。
        與 _predictive_features 同款蒐集法,但輸出為「供策略決策」的扇區 TTC/距離特徵。"""
        try:
            tr = self.env.ego.get_transform()
            eyaw = np.deg2rad(tr.rotation.yaw)
            c, s = np.cos(-eyaw), np.sin(-eyaw)
            ex, ey = tr.location.x, tr.location.y
            evel = self.env.ego.get_velocity()
            ego_speed = float(np.hypot(evel.x, evel.y))
            rows = []
            for v in self.env.world.get_actors().filter('vehicle.*'):
                if v.id == self.env.ego.id:
                    continue
                loc = v.get_transform().location
                dx, dy = loc.x - ex, loc.y - ey
                if dx * dx + dy * dy > 1600.0:       # 40 m
                    continue
                lx = c * dx - s * dy
                ly = s * dx + c * dy
                vyaw = np.deg2rad(v.get_transform().rotation.yaw)
                vel = v.get_velocity()
                rows.append([lx, ly, vyaw - eyaw, float(np.hypot(vel.x, vel.y))])
            return risk_features(np.asarray(rows, dtype=np.float64).reshape(-1, 4), ego_speed)
        except Exception:
            return NEUTRAL_RISK_FEATURES

    def reset(self):
        obs = self.env.reset()
        self.last_obs_dict = obs
        self._red_light_monitor.reset()
        self._stuck_counter = 0
        if self.use_route:
            self._setup_route()
        return self._build_obs(obs)

    def _setup_route(self):
        import random
        if self._route_planner is None:
            from carla_rl.wrappers.route_planner import RoutePlanner
            self._route_planner = RoutePlanner(self._world_map)
        ego = self.env.ego.get_location()
        sps = [sp.location for sp in self.env.vehicle_spawn_points]
        far = [loc for loc in sps if ego.distance(loc) > 80.0]
        dest = random.choice(far if far else sps)
        self._route_ok = self._route_planner.reset(ego, dest)
        self._route_completion = 0.0

    def step(self, action):
        # Pedal mutual exclusion: learned policies (MSE-mean BC in particular)
        # press throttle and brake simultaneously, and in CARLA the brake wins
        # — the car never moves. Expert data never has both; make the env
        # interface enforce it so every policy is treated consistently.
        action = np.asarray(action, dtype=np.float32).copy()
        if action[0] >= action[2]:
            action[2] = 0.0
        else:
            action[0] = 0.0
        env_applied_action = action.copy()   # 互斥後「真正送進底層」的三維控制
        obs, reward, cost, done, info = self.env.step(action)
        self.last_obs_dict = obs
        info = dict(info)
        info['cost'] = float(cost)
        # 讓上層 wrapper 用「真正執行的動作」算 applied_pedal/steer(buffer 標籤一致性)
        info['env_applied_action'] = env_applied_action
        info['speed'] = float(obs['ego_state'][3])
        info['lane_width'] = float(obs['lane_info'][0])
        info['lateral_offset'] = float(obs['lane_info'][1])
        flat = self._build_obs(obs, info)
        # path-aware lead (the env's straight-box obs[7] sees phantom leads in
        # 40% of turns — oncoming/cross traffic); 0.0 = no on-path lead
        from carla_rl.wrappers.lead_vehicle import path_aware_lead
        gap, lead_speed = path_aware_lead(flat[:307])
        info['front_gap'] = gap
        info['lead_speed'] = lead_speed
        info['lead_rel_speed'] = info['speed'] - lead_speed if gap > 0.0 else 0.0
        # Anti-idle: freezing on an open road (no lead) ends the episode as a
        # non-success — this is the ep5 failure mode (boxed in / never started).
        # 例外:正確停在紅燈前(前瞻紅燈旗標 + 約 15m 內)不可算卡住,否則會把「守紅燈」
        # 誤判成 stuck 終止 + 扣 stuck 懲罰,等於懲罰正確行為(紅燈學習失敗的元兇之一)。
        tl = info.get('tl_features')
        waiting_at_red = (
            tl is not None and len(tl) >= 5
            and float(tl[1]) >= 0.5 and float(tl[4]) <= 0.30
        )
        if info['speed'] < STUCK_SPEED and gap == 0.0 and not waiting_at_red:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        if self._stuck_counter >= STUCK_STEPS:
            done = True
            info['stuck'] = True
        # Arriving at the planned destination ends the episode as a success —
        # otherwise the car drives past the route end on degenerate waypoints
        # and gets flagged wrong-way/off-road (a false failure).
        if info.get('reached_destination'):
            done = True
        return flat, float(reward), bool(done), info

    def close(self):
        # Leave the server in async mode so it keeps running without our ticks.
        # NEVER touch a dead server: libcarla abort()s the whole process (not a
        # catchable exception) when applying settings on a lost connection.
        from carla_rl.utils.server import is_port_open

        if not is_port_open(self.params['port']):
            return
        try:
            self.env._set_synchronous_mode(False)
        except Exception:
            pass
