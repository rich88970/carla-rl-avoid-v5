"""側向緊急閃避(evasion_steer)邏輯測試。
跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.test_evasion"""
from carla_rl.wrappers.lateral_control import (
    EVADE_OFFSET, evasion_steer,
)

# risk 區塊索引: [ttc_f,dist_f, ttc_lf,dist_lf, ttc_rf,dist_rf, ttc_l,dist_l, ttc_r,dist_r]
def rf(ttc_f=1.0, dist_f=1.0, lf=(1.0, 1.0), rf_=(1.0, 1.0), l=(1.0, 1.0), r=(1.0, 1.0)):
    return [ttc_f, dist_f, lf[0], lf[1], rf_[0], rf_[1], l[0], l[1], r[0], r[1]]

# 1) 正前方安全(ttc/dist 高)→ 不閃,steer 原樣
s, fired = evasion_steer(rf(ttc_f=1.0, dist_f=1.0), 0.1)
assert s == 0.1 and fired is False
# 2) 正前方迫近 但兩側都不夠空(dist 低)→ 不閃(只靠煞車)
s, fired = evasion_steer(rf(ttc_f=0.05, dist_f=0.1, lf=(0.2, 0.2), rf_=(0.2, 0.2),
                            l=(0.2, 0.2), r=(0.2, 0.2)), 0.0)
assert fired is False
# 3) 正前方迫近 且「左側」非常空曠 → 往左閃(+offset,與 Pure-Pursuit 同號)
s, fired = evasion_steer(rf(ttc_f=0.05, dist_f=0.1, lf=(1.0, 1.0), l=(1.0, 1.0),
                            rf_=(0.2, 0.2), r=(0.2, 0.2)), 0.0)
assert fired is True and abs(s - EVADE_OFFSET) < 1e-9
# 4) 正前方迫近 且「右側」非常空曠 → 往右閃(-offset)
s, fired = evasion_steer(rf(ttc_f=0.05, dist_f=0.1, rf_=(1.0, 1.0), r=(1.0, 1.0),
                            lf=(0.2, 0.2), l=(0.2, 0.2)), 0.0)
assert fired is True and abs(s + EVADE_OFFSET) < 1e-9
# 5) 選較空曠側:兩側都過門檻但左更空 → 往左
s, fired = evasion_steer(rf(ttc_f=0.05, dist_f=0.1, lf=(1.0, 1.0), l=(1.0, 1.0),
                            rf_=(0.9, 0.6), r=(0.9, 0.6)), 0.0)
assert fired is True and s > 0
# 6) 空曠但「不安全」(該側 ttc 低,有車正逼近該側)→ 不往那側閃
s, fired = evasion_steer(rf(ttc_f=0.05, dist_f=0.1, lf=(0.1, 1.0), l=(0.1, 1.0),
                            rf_=(0.2, 0.2), r=(0.2, 0.2)), 0.0)
assert fired is False
# 7) 輸出夾在 [-1,1]
s, fired = evasion_steer(rf(ttc_f=0.05, dist_f=0.1, lf=(1.0, 1.0), l=(1.0, 1.0)), 0.9)
assert -1.0 <= s <= 1.0 and fired is True

print("evasion_steer tests PASS")
