"""Path-aware lead-vehicle detection.

The env's built-in front-vehicle check is a straight 20 m x +-2.5 m box along
the ego's heading — during turns that box points into oncoming lanes and
junction cross-traffic, so 40% of in-turn 'lead' detections are phantoms
(measured on 300k dataset frames; 4.9% on straights). Phantom leads zero out
the cruise target speed and freeze the car mid-turn.

This module instead matches the 5 nearby vehicles (ego-frame coords,
obs[251:271]) against the planned path (12 waypoints, obs[271:307]): a
vehicle is a lead only if it is ahead AND within CORRIDOR meters of some
waypoint. Works identically on live obs vectors and dataset arrays — the
replay prefill must relabel with exactly the same function.
"""

import numpy as np

CORRIDOR = 2.2     # m, lateral half-width around the waypoint path
# Same-lane forward box: a vehicle directly ahead within this lateral half-width
# is ALSO a lead, even if the route corridor curves away from it. This is the V7
# fix for rear-ends at junctions — there the route waypoints bend, so a car
# straight ahead in the ego's lane (|local_y| <= ~1.4 m in the diagnosed hits)
# fell outside CORRIDOR and was missed. 1.8 m < half a 3.5 m lane, so it catches
# same-lane leads without flagging adjacent-lane or crossing traffic.
FWD_HALFWIDTH = 1.8
NEARBY_SLICE = slice(251, 271)
WAYPOINT_SLICE = slice(271, 307)
MAX_GAP = 20.0     # keep the env's 20 m sensing convention


SAME_DIR_COS = 0.25   # cos(rel_yaw) 下限:>0.25 ≈ 朝向與 ego 夾角 < 75°。排除對向(~180°,
                      # cos≈-1)與正側向橫切(~90°,cos≈0);橫切真風險由 risk 特徵側扇區管。


def world_forward_lead(rows, halfwidth=FWD_HALFWIDTH, max_gap=MAX_GAP):
    """從「全部世界車輛」(非僅 5 最近)挑同向、同車道、正前方最近的前車,回傳 (gap, lead_speed)。
    rows: M×4 [x, y, rel_yaw, speed],ego frame(x 前、y 左)。無前車回 (0.0, 0.0)。

    取代 env 的 obs[7]/[8] 前向探測箱:那是 ±2.5m 直箱、無朝向過濾,轉彎時把對向來車當前車誤煞;
    這裡用朝向閘門(SAME_DIR_COS)排除對向/橫切,只留真正會追撞的同向前車。掃全車輛是為了補
    path_aware_lead 只看 5 最近、busy 路口會漏掉第 6+ 近前車的缺口。"""
    veh = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if veh.shape[0] == 0:
        return 0.0, 0.0
    x, y, ryaw, spd = veh[:, 0], veh[:, 1], veh[:, 2], veh[:, 3]
    gap = np.hypot(x, y)
    same_dir = np.cos(ryaw) > SAME_DIR_COS          # 同向(排除對向/橫切)
    sel = (x > 0.0) & same_dir & (np.abs(y) < halfwidth) & (gap <= max_gap)
    if not np.any(sel):
        return 0.0, 0.0
    i = np.where(sel, gap, np.inf).argmin()
    lead_speed = float(spd[i] * np.cos(ryaw[i]))    # 前車沿 ego 朝向的速度分量
    return float(gap[i]), lead_speed


def path_aware_lead(obs_batch):
    """obs_batch: (..., >=307) raw obs. Returns (gap, lead_speed), each (...,).
    gap = 0.0 when no on-path lead within MAX_GAP (matches obs[7] convention).
    """
    obs = np.asarray(obs_batch, dtype=np.float64)
    squeeze = obs.ndim == 1
    if squeeze:
        obs = obs[None]

    nearby = obs[:, NEARBY_SLICE].reshape(-1, 5, 4)   # local x, y, relyaw, speed
    wps = obs[:, WAYPOINT_SLICE].reshape(-1, 12, 3)   # local x, y, relyaw

    valid = np.abs(nearby).sum(-1) > 0                # zero rows = padding
    ahead = nearby[:, :, 0] > 0.0
    # 非對向車:前車是「同向」追撞對象,對向車(朝向夾角 >120°,cos<-0.5)不該當前車。
    # 排除彎道時對向車落進走廊/同車道箱的殘留誤煞(rel_yaw = nearby[...,2])。
    not_oncoming = np.cos(nearby[:, :, 2]) > -0.5
    dist_to_path = np.linalg.norm(
        nearby[:, :, None, :2] - wps[:, None, :, :2], axis=-1
    ).min(-1)                                         # (N, 5)
    # lead = on the (possibly curving) route corridor, OR directly ahead in the
    # ego's own lane (the V7 junction fix — catches the car the route bends past)
    in_corridor = dist_to_path < CORRIDOR
    in_lane_ahead = np.abs(nearby[:, :, 1]) < FWD_HALFWIDTH
    on_path = valid & ahead & not_oncoming & (in_corridor | in_lane_ahead)

    gap_all = np.linalg.norm(nearby[:, :, :2], axis=-1)
    gap_all = np.where(on_path & (gap_all <= MAX_GAP), gap_all, np.inf)
    idx = gap_all.argmin(-1)
    rows = np.arange(len(obs))
    gap = gap_all[rows, idx]
    lead_speed = nearby[rows, idx, 3]

    none = ~np.isfinite(gap)
    gap = np.where(none, 0.0, gap)
    lead_speed = np.where(none, 0.0, lead_speed)

    if squeeze:
        return float(gap[0]), float(lead_speed[0])
    return gap, lead_speed
