"""Offline checks for world_forward_lead 前車偵測的朝向/車道過濾。

world_forward_lead 取代 DSAFE 控制器原本併用的 env obs[7]/[8] 前向探測箱:那是 ±2.5m
直箱、無朝向過濾,轉彎時會把「對向車道來車」當前車誤煞。新函式掃 40m 內全部車輛但加朝向
閘門(只留同向)+ 車道閘門(只留 ego 車道內正前方),根除對向誤煞又保留同向前車防撞力。

Run:
    .\.venv\Scripts\python.exe -m carla_rl.scripts.test_world_forward_lead
"""

import numpy as np

from carla_rl.wrappers.lead_vehicle import world_forward_lead

PI = np.pi


def _gap(rows):
    g, _ = world_forward_lead(np.array(rows, dtype=float))
    return g


# rows: M×4 [x, y, rel_yaw, speed],ego frame(x 前、y 左)
# 同向、同車道、正前方 → 偵測為前車(回真實 gap)
assert abs(_gap([[10, 0, 0, 3.0]]) - 10.0) < 0.5
# 對向車道來車(|y| 一個車道 + 朝向 ~180°)→ 排除(這就是使用者看到的誤煞來源)
assert _gap([[10, -3.5, PI, 8.0]]) == 0.0
# 正前方同車道的對向(頭對頭)→ 也排除(罕見;真撞線由 ttc_shield 風險特徵接手)
assert _gap([[10, 0, PI, 8.0]]) == 0.0
# 路口正側向橫切(~90°)→ 排除(由 risk 側扇區處理,非縱向前車)
assert _gap([[10, 0, PI / 2, 5.0]]) == 0.0
# 鄰車道同向(|y| ≈ 一個車道)→ 排除(不為隔壁車道車煞)
assert _gap([[10, 3.5, 0, 3.0]]) == 0.0
# 兩台同向前車 → 取最近
assert abs(_gap([[15, 0, 0, 4.0], [8, 0.5, 0.1, 2.0]]) - 8.0) < 0.5
# 超過 MAX_GAP(20m)→ 排除
assert _gap([[25, 0, 0, 3.0]]) == 0.0
# 無車 → 無前車
assert _gap([]) == 0.0

# 前車速度回傳沿 ego 朝向的分量(同向 ~= 原速)
_, ls = world_forward_lead(np.array([[10, 0, 0, 3.0]], dtype=float))
assert abs(ls - 3.0) < 1e-6

print("world_forward_lead heading/lane filter tests PASS")
