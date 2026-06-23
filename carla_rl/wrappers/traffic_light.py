"""Traffic-light and intersection features queried from the CARLA API.

5 features, appended after the 307 dataset-compatible dims:
  [0] light_present  - 1.0 if a traffic light is AHEAD within LOOKAHEAD_DIST
  [1] red            - one-hot state of that upcoming light (all zero if none)
  [2] yellow
  [3] green
  [4] dist_norm      - distance to the upcoming light (or, if none, to the next
                       junction), clipped to MAX_JUNCTION_DIST and normalized to
                       [0, 1]; 1.0 = nothing within range

LOOKAHEAD (2026-06-22): features [0:4] now describe the *upcoming* light ahead in
the travel direction (scanned from every light's stop waypoints), NOT only the
light the ego is already at. Rationale: the policy must SEE a red before it can
brake for it — with the old at-light-only signal it only learned the color once
already in the trigger zone (too late), so it could never learn to stop. The
actual at-light state (for counting/terminating real violations) is returned
separately as `at_light` and used by RedLightMonitor — decoupled from the obs.

The offline dataset (collected with lights frozen green) has no equivalent
signal; for BC, pad dataset observations with NEUTRAL_FEATURES.
"""

import math

import numpy as np

N_FEATURES = 5
MAX_JUNCTION_DIST = 50.0
WAYPOINT_STEP = 2.0
LOOKAHEAD_DIST = 40.0     # m,前方紅綠燈偵測範圍
LANE_HALFWIDTH = 2.5      # m,stop waypoint 橫向偏移在此內才算「ego 正前方車道」

# no light ahead, no state, nothing in range
NEUTRAL_FEATURES = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def distance_to_junction(world_map, ego):
    """Walk lane waypoints forward until a junction; returns clipped meters."""
    waypoint = world_map.get_waypoint(ego.get_location())
    dist = 0.0
    while waypoint is not None and dist < MAX_JUNCTION_DIST:
        if waypoint.is_junction:
            return dist
        next_wps = waypoint.next(WAYPOINT_STEP)
        waypoint = next_wps[0] if next_wps else None
        dist += WAYPOINT_STEP
    return MAX_JUNCTION_DIST


def upcoming_light(ego, max_dist=LOOKAHEAD_DIST):
    """前方行進方向、max_dist 內最近的紅綠燈。回傳 (state 或 None, 距離 m)。

    掃每個紅綠燈的 stop waypoints(停止線位置),取「在 ego 前方(沿車頭方向 fwd>0)、
    橫向偏移 < LANE_HALFWIDTH(在 ego 車道正前方)、且距離最近」者。這讓策略在「還沒到
    路口」時就看得到前方燈號顏色 → 有空間提前減速。
    """
    world = ego.get_world()
    tr = ego.get_transform()
    ex, ey = tr.location.x, tr.location.y
    fyaw = math.radians(tr.rotation.yaw)
    fx, fy = math.cos(fyaw), math.sin(fyaw)

    best_state, best_d = None, max_dist
    for tl in world.get_actors().filter('*traffic_light*'):
        try:
            stop_wps = tl.get_stop_waypoints()
        except Exception:
            continue
        for sw in stop_wps:
            loc = sw.transform.location
            dx, dy = loc.x - ex, loc.y - ey
            fwd = dx * fx + dy * fy            # 沿行進方向的距離
            if fwd <= 0.0 or fwd > max_dist:
                continue
            lat = abs(-dx * fy + dy * fx)      # 橫向偏移
            if lat > LANE_HALFWIDTH:
                continue
            if fwd < best_d:
                best_d, best_state = fwd, tl.state
    return best_state, best_d


def traffic_light_features(world_map, ego):
    import carla

    features = np.zeros(N_FEATURES, dtype=np.float32)

    state, dist = upcoming_light(ego)          # 前瞻:前方 40m 最近燈
    if state is not None:
        features[0] = 1.0
        if state == carla.TrafficLightState.Red:
            features[1] = 1.0
        elif state == carla.TrafficLightState.Yellow:
            features[2] = 1.0
        elif state == carla.TrafficLightState.Green:
            features[3] = 1.0
        features[4] = dist / MAX_JUNCTION_DIST
    else:
        # 前方無燈 → 用到下個路口的距離(保留路口資訊)
        features[4] = distance_to_junction(world_map, ego) / MAX_JUNCTION_DIST

    # 實際「在燈下」的燈物件(供違規監測/終止用),與前瞻觀測解耦
    at_light = ego.get_traffic_light() if ego.is_at_traffic_light() else None
    return features, at_light


class RedLightMonitor:
    """Detects running a red light: ego has actually CROSSED the stop line of a red
    light (not merely entered its trigger box) while moving > 1 m/s. Counted once
    per light encounter, reset per episode.

    用該紅燈的 stop waypoints 判定:取與 ego 同車道(橫向 < LANE 半寬)的停止線,計算 ego
    沿停止線朝向(行進方向)越過多少;越線 > CROSS_MARGIN(0.25 m)才算違規。只進 trigger box
    還沒到停止線時不觸發(修正第三輪審查 #2)。"""

    SPEED_THRESHOLD = 1.0
    CROSS_MARGIN = 0.25       # m,車身越過停止線多少才算闖
    LANE_HALF = 3.0           # m,stop waypoint 橫向偏移在此內才算 ego 該車道

    def __init__(self):
        self._violated_ids = set()

    def reset(self):
        self._violated_ids.clear()

    def check(self, ego, speed):
        import math
        import carla
        if ego is None or speed <= self.SPEED_THRESHOLD:
            return False
        if not ego.is_at_traffic_light():
            return False
        light = ego.get_traffic_light()
        if light is None or light.state != carla.TrafficLightState.Red:
            return False
        if light.id in self._violated_ids:
            return False
        try:
            stop_wps = light.get_stop_waypoints()
        except Exception:
            return False
        loc = ego.get_transform().location
        for sw in stop_wps:
            swt = sw.transform
            yaw = math.radians(swt.rotation.yaw)
            fx, fy = math.cos(yaw), math.sin(yaw)   # 停止線朝向 = 行進方向
            dx, dy = loc.x - swt.location.x, loc.y - swt.location.y
            along = dx * fx + dy * fy                # 沿行進方向:>0 = 已越過停止線
            lat = abs(-dx * fy + dy * fx)            # 橫向偏移
            if lat < self.LANE_HALF and along > self.CROSS_MARGIN:
                self._violated_ids.add(light.id)
                return True
        return False
