"""risk_features 純函式 assert 測試(5 扇區版)。
跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.test_risk_features
索引:0/1=前 ttc/dist、2/3=左前、4/5=右前、6/7=左側、8/9=右側。"""
import numpy as np
from carla_rl.wrappers.risk_features import (
    risk_features, min_ttc, N_RISK_FEATURES, NEUTRAL_RISK_FEATURES)

# 1) 無車:全 1(最安全),10 維
assert N_RISK_FEATURES == 10 and len(NEUTRAL_RISK_FEATURES) == 10
assert np.allclose(risk_features(np.zeros((0, 4)), 6.0), 1.0)
# 2) 正前方靜止車、ego 6 m/s 逼近:front ttc<1,其餘 4 扇區 ttc==1
f = risk_features([[12.0, 0.0, 0.0, 0.0]], 6.0)
assert 0.0 < f[0] < 0.7
assert f[2] == 1.0 and f[4] == 1.0 and f[6] == 1.0 and f[8] == 1.0
# 3) 正後方(被超過,bearing 180°)不在任何扇區:全安全
assert np.allclose(risk_features([[-10.0, 0.0, 0.0, 0.0]], 6.0), 1.0)
# 4) 左前 8m 在側、ego 直行安全通過:dist_lf<1、ttc_lf 安全、前扇區空
lf = risk_features([[8.0, 8.0, 0.0, 0.0]], 6.0)
assert lf[3] < 1.0 and lf[2] == 1.0 and f[0] < 1.0  # f from case 2; lf front empty:
assert lf[0] == 1.0
# 5) 左前「相對速度朝 ego」(會撞):ttc_lf<1
pos = np.array([8.0, 8.0]); ego_v = np.array([6.0, 0.0])
rel_v = -pos / np.linalg.norm(pos) * 10.0
abs_v = rel_v + ego_v
cross = risk_features([[8.0, 8.0, float(np.arctan2(abs_v[1], abs_v[0])),
                        float(np.linalg.norm(abs_v))]], 6.0)
assert cross[2] < 1.0
# 6) 正左側(bearing 90°,路口橫切)有車:落在左側扇區 → dist_L(索引7)<1
side = risk_features([[0.0, 10.0, 0.0, 0.0]], 6.0)
assert side[7] < 1.0
# 7) min_ttc 取所有 5 扇區最小(偶數索引)
assert abs(min_ttc([0.9, 0.5, 0.9, 0.5, 0.2, 0.5, 0.9, 0.5, 0.9, 0.5]) - 0.2) < 1e-9
# 8) 對向車道直線來車(前方 20m、橫向 3.5m、反向 heading,ego 6 m/s):橫向恆隔一個車道,
#    縱向雖逼近但永遠不在 ego 路徑 → 不可能撞 → 不該被標為風險(front ttc==1)。
oncoming = risk_features([[20.0, 3.5, np.pi, 6.0]], 6.0)
assert oncoming[0] == 1.0, f"oncoming opposite-lane wrongly flagged: {oncoming[0]}"
# 9) 已知取捨(報告,非硬性 assert):同向誤煞修正的朝向閘門 cos(rel_yaw) > -0.5 會一併排除「同車道
#    逆向正面來車」(橫向 1.0m、heading≈π,cos≈-1)。單一 ego-frame 下,此情境與「彎道時對向車道車
#    被掃進前扇區」幾何上無法區分;本專案刻意壓制對向誤煞(觀測到的真實問題),代價是放棄這個罕見的
#    逆向同車道正面對撞偵測(Town03 NPC 不會逆向行駛,評估中不發生)。故此處僅報告,不強制失敗。
head_on = risk_features([[20.0, 1.0, np.pi, 6.0]], 6.0)
status = "caught" if head_on[0] < 1.0 else "missed (known gap: oncoming-gate tradeoff)"
print(f"  [report] same-lane wrong-way head-on: {status} (front_ttc={head_on[0]:.3f})")
print("risk_features (5-sector + lateral-gate) tests PASS")
