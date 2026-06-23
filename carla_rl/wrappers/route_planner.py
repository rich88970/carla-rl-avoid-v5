"""Global route planning — the fix for roundabout-looping and junction
mis-turns. The env's reference waypoints are greedy (next()[0]) with NO
destination, so the car never knows which roundabout exit / junction branch to
take and can circle forever. This plans an actual route from the ego to a
random far destination (CARLA GlobalRoutePlanner) and supplies the next N route
points in ego-frame, which overwrite the greedy waypoints in the observation.
"""

import os
import sys

import numpy as np

# CARLA agents 模組在源碼建置的 PythonAPI 內(非 pip wheel)。路徑由環境變數
# CARLA_PYTHONAPI 提供(指向 <CARLA_ROOT>/PythonAPI/carla);見 INSTALL.md。
_carla_pythonapi = os.environ.get('CARLA_PYTHONAPI')
if _carla_pythonapi and _carla_pythonapi not in sys.path:
    sys.path.insert(0, _carla_pythonapi)


class RoutePlanner:
    def __init__(self, world_map, sampling=2.0, n_waypoints=12):
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except ImportError as exc:
            raise RuntimeError(
                '找不到 CARLA agents 模組。請設定 CARLA_PYTHONAPI,'
                '例如指向 <CARLA_ROOT>/PythonAPI/carla。'
            ) from exc

        self.grp = GlobalRoutePlanner(world_map, sampling)
        self.n = n_waypoints
        self.route = []   # list of carla.Location along the planned path
        self.idx = 0

    def reset(self, start_loc, end_loc):
        """Plan a route; returns True if a usable route was found."""
        try:
            traced = self.grp.trace_route(start_loc, end_loc)
            self.route = [wp.transform.location for wp, _ in traced]
        except Exception:
            self.route = []
        self.idx = 0
        return len(self.route) >= self.n

    def waypoints(self, ego_x, ego_y, ego_yaw):
        """Advance to the nearest route point ahead and return (N x 3 ego-frame
        waypoints [local_x, local_y, rel_yaw], completion_fraction)."""
        if not self.route:
            return None, 1.0
        # advance index to the closest point within a forward window (never
        # backward, so the car commits to progressing along the route)
        best_i, best_d = self.idx, 1e18
        for i in range(self.idx, min(self.idx + 30, len(self.route))):
            loc = self.route[i]
            d = (loc.x - ego_x) ** 2 + (loc.y - ego_y) ** 2
            if d < best_d:
                best_d, best_i = d, i
        self.idx = best_i

        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        wps = np.zeros((self.n, 3), dtype=np.float32)
        n_route = len(self.route)
        for k in range(self.n):
            i = min(self.idx + k, n_route - 1)
            loc = self.route[i]
            lx = c * (loc.x - ego_x) - s * (loc.y - ego_y)
            ly = s * (loc.x - ego_x) + c * (loc.y - ego_y)
            nxt = self.route[min(i + 1, n_route - 1)]
            seg = np.arctan2(nxt.y - loc.y, nxt.x - loc.x) - ego_yaw
            seg = np.arctan2(np.sin(seg), np.cos(seg))   # wrap to [-pi, pi]
            wps[k] = [lx, ly, seg]
        completion = self.idx / max(1, n_route - 1)
        return wps, completion
