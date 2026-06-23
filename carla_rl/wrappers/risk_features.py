"""碰撞風險特徵,供 RL 策略「直接決策避撞」用(取代為規則 planner 設計的 V9 特徵)。
對 ego frame 內鄰車做等速預測:其他車相對 ego 當前速度的相對運動,算每扇區最近預測
TTC 與當前距離。純函式,易單元測試。

v3(2026-06-18):由 3 扇區(前/左前/右前,僅前向 180°)擴成 **5 扇區**,加入左、右
正側向(±[75,135]°)→ 看得到路口「橫切」來車(avoid_v2 殘留碰撞的主因之一)。"""
import numpy as np

N_RISK_FEATURES = 10            # 5 扇區 × (ttc, dist)
RISK_HORIZON = 5.0             # s,預測視野;ttc 上限(視為安全)
RISK_MARGIN = 3.0             # m,keep-out 半徑(預測閉合距離小於此才算會撞)
RISK_MAX_DIST = 40.0          # m,距離正規化上限
# 橫向閘門:只有「最接近時刻的橫向偏移 |cy| < 此值」才算「真的在 ego 車道內/侵入路徑」。
# 用來排除「對向車道/鄰道平行錯車」的不可能風險——它們橫向恆隔約一個車道(~3.5m),
# 縱向雖逼近,但永遠不在 ego 路徑上。車道半寬 ~1.75m + buffer。
LATERAL_GATE = 2.5            # m
# 無風險的中性值(ttc 與 dist 都 = 1 = 最安全/最遠);離線 prefill 用
NEUTRAL_RISK_FEATURES = np.ones(N_RISK_FEATURES, dtype=np.float32)

# 扇區邊界(bearing 度,0=正前、+左、-右)。涵蓋前向 + 兩側 ±135°,只留正後方不看。
_SECTORS = [
    lambda b: np.abs(b) <= 30.0,                       # 0 前
    lambda b: (b > 30.0) & (b <= 75.0),                # 1 左前
    lambda b: (b < -30.0) & (b >= -75.0),              # 2 右前
    lambda b: (b > 75.0) & (b <= 135.0),               # 3 左側(橫切)
    lambda b: (b < -75.0) & (b >= -135.0),             # 4 右側(橫切)
]


def risk_features(vehicles, ego_speed):
    """vehicles: M×4 [x, y, rel_yaw, speed],ego frame(x 前、y 左)。ego 沿 +x 以
    ego_speed 前進。回傳 10 維 = 5 扇區各 [ttc, dist],皆正規化到 [0,1]
    (ttc/RISK_HORIZON、dist/RISK_MAX_DIST;1 = 最安全/最遠/該扇區無車)。"""
    out = np.ones(N_RISK_FEATURES, dtype=np.float32)
    veh = np.asarray(vehicles, dtype=np.float64).reshape(-1, 4)
    if veh.shape[0] == 0:
        return out
    x, y = veh[:, 0], veh[:, 1]
    # 其他車相對 ego(速度 = [ego_speed, 0])的相對速度
    rvx = veh[:, 3] * np.cos(veh[:, 2]) - ego_speed
    rvy = veh[:, 3] * np.sin(veh[:, 2])
    vv = rvx * rvx + rvy * rvy
    # 最近接近時間 t* = -(p·v)/(v·v),夾在 [0, horizon]
    t_star = np.where(vv > 1e-6, -(x * rvx + y * rvy) / np.maximum(vv, 1e-6), RISK_HORIZON)
    t_star = np.clip(t_star, 0.0, RISK_HORIZON)
    cx_c = x + rvx * t_star
    cy_c = y + rvy * t_star            # 最接近時刻的「橫向」偏移(ego frame y)
    closest = np.hypot(cx_c, cy_c)
    dist_now = np.hypot(x, y)
    # 真風險 = 預測閉合進 margin 「且」最接近時仍在 ego 車道內(|cy|<LATERAL_GATE)
    # 「且」非對向車(朝向夾角 <120°)。
    #  - LATERAL_GATE 排除對向/鄰道「平行錯車」(橫向恆 ~3.5m > gate)。
    #  - 朝向閘門 not_oncoming 再排除「彎道時對向車掃進前扇區」的殘留誤煞:對向車朝向與 ego
    #    夾角 ~180°(cos≈-1),門檻 -0.5(>120°)濾掉;同向(cos≈1)與路口正側向橫切(cos≈0)
    #    都保留 → 仍抓得到路口橫切真風險。
    not_oncoming = np.cos(veh[:, 2]) > -0.5
    in_path = (closest < RISK_MARGIN) & (np.abs(cy_c) < LATERAL_GATE) & (vv > 1e-6) & not_oncoming
    ttc = np.where(in_path, t_star, RISK_HORIZON)
    bearing = np.degrees(np.arctan2(y, x))   # 0 = 正前、+ 左、- 右
    for i, sel in enumerate(_SECTORS):
        m = sel(bearing)
        if np.any(m):
            out[2 * i] = np.clip(ttc[m].min() / RISK_HORIZON, 0.0, 1.0)
            out[2 * i + 1] = np.clip(dist_now[m].min() / RISK_MAX_DIST, 0.0, 1.0)
    return out


def min_ttc(risk_feat):
    """從 risk 特徵取所有扇區最小 TTC(正規化 [0,1]),供獎勵的 TTC 懲罰用。"""
    rf = np.asarray(risk_feat, dtype=np.float64)
    return float(rf[0::2].min())   # 偶數索引 = 各扇區 ttc
