"""Per-episode metric accumulation.

Definitions (Phase 1):
  - success: episode reached max_time_episode with no collision / off-road
  - lane_invasion_steps: steps where lateral_offset > lane_width / 2
  - red_light_violations: ego affected by a red light while moving > 1 m/s,
    counted once per traffic light encounter. Taken from
    info['red_light_violation'] (emitted by CarlaGymEnv with extended_obs);
    falls back to a direct CARLA query for non-extended envs
  - smoothness: mean |steer_t - steer_{t-1}| and mean |lateral acceleration|
  - distance_m: integrated ego displacement
  - route_completion: fraction of the planned global route covered (use_route
    envs only; -1 when no route is planned). This is the real route-completion
    metric — distance_m remains as a route-agnostic fallback.

Phase 4 (cruise) additions, fed by info['front_gap'] (raw obs[7]; 0 = no lead
within 20 m):
  - free flow: steps with no lead, or a lead far enough that the headway
    target equals target_speed (gap >= d0 + T * target_speed), after a
    WARMUP_STEPS grace period for the initial acceleration. mean/std/% above
    the 5.5 m/s floor are reported over those steps.
  - headway violation: steps with a lead closer than d_safe = d0 + T * speed
    (same constants as the cruise reward), over steps with any lead.
  - pedal smoothness: mean |d(throttle - brake)| between consecutive steps.
"""

from dataclasses import dataclass, field, asdict

import numpy as np

from carla_rl.configs.reward_config import CRUISE_REWARD_CONFIG as _CRUISE

# Free-flow / headway metric constants, single-sourced from the cruise preset
# so the reward and the metrics always agree on what "safe gap" means.
_HEADWAY_T = _CRUISE.headway_time
_HEADWAY_D0 = _CRUISE.headway_offset
_TARGET_SPEED = _CRUISE.target_speed
_SPEED_FLOOR = _CRUISE.low_speed_floor
WARMUP_STEPS = 50  # skip initial acceleration when judging free-flow speed


@dataclass
class EpisodeResult:
    episode: int
    steps: int = 0
    total_reward: float = 0.0
    total_cost: float = 0.0
    collided: bool = False
    off_road: bool = False
    success: bool = False
    distance_m: float = 0.0
    avg_speed: float = 0.0
    lane_invasion_steps: int = 0
    red_light_violations: int = 0
    mean_abs_steer_delta: float = 0.0
    mean_abs_lat_accel: float = 0.0
    video: str = ''
    free_steps: int = 0
    lead_steps: int = 0
    mean_speed_free: float = 0.0
    speed_std_free: float = 0.0
    pct_above_floor_free: float = 0.0
    headway_violation_rate: float = 0.0
    mean_abs_pedal_delta: float = 0.0
    # straight-segment oscillation: measured only where the rolling-mean steer
    # is ~0 (no sustained turn). flip rate = sign reversals per step there.
    straight_abs_steer_delta: float = 0.0
    straight_steer_flip_rate: float = 0.0
    # WEAVE metrics — the oscillation the eye sees but |dsteer| misses: the
    # body yaw-rate swings left/right over ~1-2 s even when per-step steering
    # increments are small. yaw_flip_rate counts yaw-rate sign reversals (with
    # both sides above a noise floor); straight_mean_abs_yaw_rate should be
    # ~0 deg/s when truly driving straight.
    yaw_flip_rate: float = 0.0
    straight_mean_abs_yaw_rate: float = 0.0
    # mean |yaw_rate_t - yaw_rate_{t-1}| over the WHOLE episode (deg/s per step)
    # — catches lateral roughness everywhere, including cornering jitter that
    # the straight-only metric misses (the gap the user spotted on video).
    mean_abs_yaw_jerk: float = 0.0
    # fraction of the planned global route covered (0..1), the real route-
    # completion metric — only meaningful with use_route envs; -1 = no route.
    route_completion: float = -1.0
    # episode ended by the anti-idle guard (frozen on a clear road) — a failure
    stuck: bool = False

    def to_dict(self):
        return asdict(self)


class EpisodeTracker:
    def __init__(self, episode_index, max_time_episode):
        self.result = EpisodeResult(episode=episode_index)
        self.max_time_episode = max_time_episode
        self._speeds = []
        self._steer_deltas = []
        self._lat_accels = []
        self._prev_steer = 0.0
        self._prev_location = None
        self._violated_light_ids = set()
        self._free_speeds = []
        self._lead_violations = []
        self._pedal_deltas = []
        self._prev_pedal = None
        self._steer_window = []   # last N applied steers, for straight detection
        self._straight_deltas = []
        self._straight_flips = []
        self._yaw_rates = []          # deg/s, signed
        self._straight_yaw_rates = []
        self._prev_yaw_rate = 0.0
        self._yaw_flips = 0
        self._route_completion = None   # last reported route_completion, if any

    def update(self, action, reward, info, ego):
        r = self.result
        r.steps += 1
        r.total_reward += reward
        r.total_cost += info.get('cost', 0.0)

        # with a rate-limit/hybrid/1D wrapper the env applies a filtered action;
        # measure smoothness on what actually drove the car. applied_action 可能是
        # 三維 [t,s,b] 或一維 [pedal];steer/pedal 優先讀 info 的 applied_steer/applied_pedal。
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        applied_action = np.asarray(
            info.get('applied_action', raw_action), dtype=np.float32).reshape(-1)

        speed = float(info.get('speed', 0.0))
        self._speeds.append(speed)

        if 'applied_steer' in info:
            steer = float(info['applied_steer'])
        elif applied_action.size >= 2:
            steer = float(applied_action[1])
        else:
            steer = 0.0
        self._steer_deltas.append(abs(steer - self._prev_steer))

        # straight segment = rolling-mean steer ~0 over the last second
        # (oscillation averages out to ~0; a real turn holds a sustained sign)
        self._steer_window.append(steer)
        if len(self._steer_window) > 10:
            self._steer_window.pop(0)
        if len(self._steer_window) == 10 and abs(np.mean(self._steer_window)) < 0.03:
            self._straight_deltas.append(abs(steer - self._prev_steer))
            self._straight_flips.append(
                1.0 if steer * self._prev_steer < 0 and abs(steer) > 0.01 else 0.0
            )
        self._prev_steer = steer

        if 'applied_pedal' in info:
            pedal = float(info['applied_pedal'])
        elif applied_action.size >= 3:
            pedal = float(applied_action[0] - applied_action[2])
        else:
            pedal = float(applied_action[0])
        if self._prev_pedal is not None:
            self._pedal_deltas.append(abs(pedal - self._prev_pedal))
        self._prev_pedal = pedal

        gap = info.get('front_gap')  # absent on older envs -> metrics stay 0
        if gap is not None:
            if gap > 0.0:
                d_safe = _HEADWAY_D0 + _HEADWAY_T * speed
                self._lead_violations.append(1.0 if gap < d_safe else 0.0)
            # free flow: no lead, or lead far enough that v_des == target
            free = gap == 0.0 or gap >= _HEADWAY_D0 + _HEADWAY_T * _TARGET_SPEED
            # 但「合法停/減速在紅燈或黃燈前」不算自由路速度——否則守紅燈的 0 m/s 會被算進
            # free-flow,讓正確守燈看起來像速度退化(審查 #4)。tl[1]=red, tl[2]=yellow, tl[4]=dist。
            tl = info.get('tl_features')
            slowing_for_light = (
                tl is not None and len(tl) >= 5
                and (float(tl[1]) >= 0.5 or float(tl[2]) >= 0.5)
                and float(tl[4]) <= 0.30
            )
            if free and not slowing_for_light and r.steps > WARMUP_STEPS:
                self._free_speeds.append(speed)

        if ego is not None:
            loc = ego.get_location()
            if self._prev_location is not None:
                r.distance_m += float(
                    np.hypot(loc.x - self._prev_location[0], loc.y - self._prev_location[1])
                )
            self._prev_location = (loc.x, loc.y)

            accel = ego.get_acceleration()
            self._lat_accels.append(abs(accel.y))

            # weave: body yaw-rate oscillation (deg/s). A flip = sign reversal
            # with both samples above a 1.5 deg/s noise floor.
            yaw_rate = float(ego.get_angular_velocity().z)
            self._yaw_rates.append(yaw_rate)
            if (yaw_rate * self._prev_yaw_rate < 0
                    and abs(yaw_rate) > 1.5 and abs(self._prev_yaw_rate) > 1.5):
                self._yaw_flips += 1
            self._prev_yaw_rate = yaw_rate
            if (len(self._steer_window) == 10
                    and abs(np.mean(self._steer_window)) < 0.03):
                self._straight_yaw_rates.append(abs(yaw_rate))

            if 'red_light_violation' in info:
                if info['red_light_violation']:
                    r.red_light_violations += 1
            else:
                self._check_red_light(ego, speed)

        lane_width = info.get('lane_width', 0.0)
        lateral_offset = info.get('lateral_offset', 0.0)
        if lane_width > 0 and abs(lateral_offset) > lane_width / 2:
            r.lane_invasion_steps += 1

        if 'route_completion' in info:
            self._route_completion = float(info['route_completion'])

    def _check_red_light(self, ego, speed):
        try:
            if not ego.is_at_traffic_light():
                return
            light = ego.get_traffic_light()
            if light is None:
                return
            import carla
            if light.state == carla.TrafficLightState.Red and speed > 1.0:
                if light.id not in self._violated_light_ids:
                    self._violated_light_ids.add(light.id)
                    self.result.red_light_violations += 1
        except Exception:
            pass  # light queries are best-effort; never kill an eval episode

    def finalize(self, info):
        r = self.result
        r.collided = bool(info.get('is_collision', False))
        r.off_road = bool(info.get('is_off_road', False))
        # success with a planned route = REACHED the destination (surviving the
        # clock while stuck/idle is not success — that false-positive is exactly
        # what the route metric exposes). Without a route, fall back to the
        # Phase-1 definition: survived the full time budget. Both also require
        # no collision / off-road.
        reached = bool(info.get('reached_destination', False))
        routed = self._route_completion is not None
        survived = reached if routed else r.steps >= self.max_time_episode
        r.stuck = bool(info.get('stuck', False))
        # 成功須同時:無碰撞、無出界、未卡住、零闖紅燈,且存活(到點或撐滿時間)。
        # 把闖紅燈/卡住納入失敗,紅綠燈實驗的「成功率」才有意義。
        r.success = (
            (not r.collided) and (not r.off_road) and (not r.stuck)
            and (r.red_light_violations == 0) and survived
        )
        r.avg_speed = float(np.mean(self._speeds)) if self._speeds else 0.0
        r.mean_abs_steer_delta = float(np.mean(self._steer_deltas)) if self._steer_deltas else 0.0
        r.mean_abs_lat_accel = float(np.mean(self._lat_accels)) if self._lat_accels else 0.0
        r.free_steps = len(self._free_speeds)
        r.lead_steps = len(self._lead_violations)
        if self._free_speeds:
            r.mean_speed_free = float(np.mean(self._free_speeds))
            r.speed_std_free = float(np.std(self._free_speeds))
            r.pct_above_floor_free = float(
                np.mean(np.asarray(self._free_speeds) > _SPEED_FLOOR)
            )
        if self._lead_violations:
            r.headway_violation_rate = float(np.mean(self._lead_violations))
        if self._pedal_deltas:
            r.mean_abs_pedal_delta = float(np.mean(self._pedal_deltas))
        if self._straight_deltas:
            r.straight_abs_steer_delta = float(np.mean(self._straight_deltas))
            r.straight_steer_flip_rate = float(np.mean(self._straight_flips))
        if self._yaw_rates:
            r.yaw_flip_rate = float(self._yaw_flips / len(self._yaw_rates))
        if len(self._yaw_rates) > 1:
            r.mean_abs_yaw_jerk = float(np.mean(np.abs(np.diff(self._yaw_rates))))
        if self._straight_yaw_rates:
            r.straight_mean_abs_yaw_rate = float(np.mean(self._straight_yaw_rates))
        if self._route_completion is not None:
            r.route_completion = self._route_completion
        return r


def summarize(results):
    n = len(results)
    if n == 0:
        return {}
    summary = {
        'episodes': n,
        'success_rate': sum(r.success for r in results) / n,
        'collision_rate': sum(r.collided for r in results) / n,
        'off_road_rate': sum(r.off_road for r in results) / n,
        # 卡住率:長程評估的主要失效模式(非碰撞、非出界的提早終止)。每集 stuck
        # 由 CarlaGymEnv 的 anti-idle 終止設定,finalize 已收集。
        'stuck_rate': sum(r.stuck for r in results) / n,
        'red_light_violations_per_episode': sum(r.red_light_violations for r in results) / n,
        'mean_lane_invasion_steps': float(np.mean([r.lane_invasion_steps for r in results])),
        'mean_total_reward': float(np.mean([r.total_reward for r in results])),
        'mean_total_cost': float(np.mean([r.total_cost for r in results])),
        'mean_steps': float(np.mean([r.steps for r in results])),
        'mean_distance_m': float(np.mean([r.distance_m for r in results])),
        'mean_avg_speed': float(np.mean([r.avg_speed for r in results])),
        'mean_abs_steer_delta': float(np.mean([r.mean_abs_steer_delta for r in results])),
        'mean_abs_lat_accel': float(np.mean([r.mean_abs_lat_accel for r in results])),
    }
    # Cruise metrics: average only over episodes that have the relevant steps,
    # so all-zero placeholder episodes don't drag the means down.
    free = [r for r in results if r.free_steps > 0]
    lead = [r for r in results if r.lead_steps > 0]
    summary.update({
        'mean_speed_free': float(np.mean([r.mean_speed_free for r in free])) if free else 0.0,
        'mean_speed_std_free': float(np.mean([r.speed_std_free for r in free])) if free else 0.0,
        'mean_pct_above_floor_free':
            float(np.mean([r.pct_above_floor_free for r in free])) if free else 0.0,
        'mean_headway_violation_rate':
            float(np.mean([r.headway_violation_rate for r in lead])) if lead else 0.0,
        'mean_abs_pedal_delta': float(np.mean([r.mean_abs_pedal_delta for r in results])),
        'straight_abs_steer_delta':
            float(np.mean([r.straight_abs_steer_delta for r in results])),
        'straight_steer_flip_rate':
            float(np.mean([r.straight_steer_flip_rate for r in results])),
        'yaw_flip_rate': float(np.mean([r.yaw_flip_rate for r in results])),
        'straight_mean_abs_yaw_rate':
            float(np.mean([r.straight_mean_abs_yaw_rate for r in results])),
        'mean_abs_yaw_jerk': float(np.mean([r.mean_abs_yaw_jerk for r in results])),
    })
    # route completion: only over episodes that actually had a planned route
    routed = [r for r in results if r.route_completion >= 0.0]
    if routed:
        summary['mean_route_completion'] = float(
            np.mean([r.route_completion for r in routed])
        )
        summary['route_reached_rate'] = float(
            np.mean([r.route_completion >= 0.9 for r in routed])
        )
    return summary
