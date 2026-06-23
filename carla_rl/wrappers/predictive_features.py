"""Predictive perception features (V9): give the policy look-ahead so it can
anticipate conflicts instead of reacting late. Pure function over ego-frame
vehicle kinematics + the planned route; appended AFTER the 312-dim obs. Used
identically online (all vehicles) and offline (the 5 stored nearby), like
path_aware_lead — single source of truth so BC / prefill / live agree.

The conflict check is "does a vehicle's constant-velocity prediction come within
MARGIN of the ego's planned route corridor within HORIZON seconds". The route is
the ego's own future path, so checking against it already encodes ego motion;
this yields anticipation features, NOT a precise space-time guarantee (that is
the safety planner's job).
"""

import numpy as np

N_PRED_FEATURES = 5
HORIZON = 3.0            # s, constant-velocity look-ahead
MARGIN = 2.5            # m, lateral conflict half-width
TTC_CLIP = 5.0          # s, normalization cap for min-TTC
DIST_CLIP = 30.0        # m, normalization cap for distance-to-conflict
CONV_CAP = 5.0          # normalization cap for converging-count
_DT = 0.2

# neutral = "no threat": 1.0 where larger is safer, 0.0 where 0 is safe
NEUTRAL_PRED_FEATURES = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def predictive_features(vehicles, route_pts, junction_dist=1.0):
    """vehicles: (M,4) ego-frame [local_x, local_y, rel_yaw, speed] (real, no pad).
    route_pts: (N,2) ego-frame [x, y] of the planned route ahead. junction_dist:
    obs[311] (1.0 = no junction in range). Returns N_PRED_FEATURES float32:
    [min_ttc_norm, n_converging_norm, dist_to_conflict_norm, time_to_conflict_norm,
     junction_occupied]."""
    veh = np.asarray(vehicles, dtype=np.float64).reshape(-1, 4)
    if veh.shape[0] == 0:
        out = NEUTRAL_PRED_FEATURES.copy()
        out[4] = _junction_occupied(veh, junction_dist)
        return out

    route = np.asarray(route_pts, dtype=np.float64).reshape(-1, 2)
    steps = int(HORIZON / _DT)
    min_ttc = min_dist = min_time = np.inf
    n_conv = 0
    for x, y, ryaw, spd in veh:
        vx, vy = spd * np.cos(ryaw), spd * np.sin(ryaw)
        for k in range(1, steps + 1):
            t = k * _DT
            px, py = x + vx * t, y + vy * t
            if px < -2.0:                       # behind the ego, ignore
                continue
            if len(route):
                d = np.min(np.hypot(route[:, 0] - px, route[:, 1] - py))
            else:
                d = np.hypot(px, py)
            if d < MARGIN:                      # predicted to be on the ego path
                n_conv += 1
                min_ttc = min(min_ttc, t)
                min_time = min(min_time, t)
                min_dist = min(min_dist, np.hypot(px, py))
                break
    f = np.empty(N_PRED_FEATURES, dtype=np.float32)
    f[0] = min(min_ttc, TTC_CLIP) / TTC_CLIP if np.isfinite(min_ttc) else 1.0
    f[1] = min(n_conv, CONV_CAP) / CONV_CAP
    f[2] = min(min_dist, DIST_CLIP) / DIST_CLIP if np.isfinite(min_dist) else 1.0
    f[3] = min(min_time, HORIZON) / HORIZON if np.isfinite(min_time) else 1.0
    f[4] = _junction_occupied(veh, junction_dist)
    return f


def _junction_occupied(veh, junction_dist):
    """1.0 if near a junction (obs[311] < 0.3) AND a vehicle sits in the box just
    ahead (0 < local_x < 12, |local_y| < 4); else 0.0."""
    if junction_dist >= 0.3 or veh.shape[0] == 0:
        return 0.0
    ahead = (veh[:, 0] > 0.0) & (veh[:, 0] < 12.0) & (np.abs(veh[:, 1]) < 4.0)
    return 1.0 if np.any(ahead) else 0.0
