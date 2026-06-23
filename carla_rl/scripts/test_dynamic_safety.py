"""統一安全控制器(動態安全距離 + 分級煞車 + TTC)性質測試。
跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.test_dynamic_safety"""
from carla_rl.wrappers.lateral_control import (
    DSAFE_D0, dynamic_safe_gap, graduated_brake,
)

# 1) 動態安全距離隨速單調遞增,靜止 = d0
assert abs(dynamic_safe_gap(0.0) - DSAFE_D0) < 1e-9
assert dynamic_safe_gap(0.0) < dynamic_safe_gap(3.0) < dynamic_safe_gap(6.0)

# 2) 無前車(gap<=0)→ 指令原樣通過
assert graduated_brake(0.0, 6.0, 0.0, 0.8, 0.0) == (0.8, 0.0)
assert graduated_brake(-1.0, 6.0, 0.0, 0.8, 0.0) == (0.8, 0.0)

# 3) 安全距離外 → 不介入(RL 自由給油)
far = dynamic_safe_gap(6.0) + 5.0
t, b = graduated_brake(far, 6.0, 1.0, 0.8, 0.0)
assert t == 0.8 and b == 0.0

# 4) 動態縮放降卡住:靜止時 d_safe=d0(5.5)< 舊固定 7m → 對「6m 外停住的前車」不誤煞
t, b = graduated_brake(6.0, 0.0, 0.0, 0.5, 0.0)   # v=0, gap=6 > d_safe(0)=5.5
assert b == 0.0 and t == 0.5                        # 不介入(舊的固定 7m 會誤煞 → 卡住)

# 5) 動態縮放降碰撞:高速時 d_safe 大 → 對「10m 外前車」提早分級煞車
t, b = graduated_brake(10.0, 6.0, 0.0, 0.8, 0.0)   # v=6, d_safe(6)≈14.2 > 10 → 介入
assert b > 0.0 and t < 0.8

# 6) 分級:侵入越深 → 煞越大力、油門收越多(非二元)
t_near, b_near = graduated_brake(3.0, 6.0, 0.0, 0.8, 0.0)
t_far, b_far = graduated_brake(9.0, 6.0, 0.0, 0.8, 0.0)
assert b_near > b_far > 0.0          # 3m 比 9m 煞更大力
assert t_near < t_far                 # 3m 比 9m 油門收更多
assert 0.0 <= b_near <= 1.0

# 7) TTC 硬底線:逼近快(closing 大、TTC<2s)→ 全力煞、油門歸零
t, b = graduated_brake(4.0, 6.0, 4.0, 1.0, 0.0)    # ttc=4/4=1.0s < 2.0
assert b >= 0.9 and t == 0.0

# 8) 不削弱既有煞車:回傳 brake 不小於指令 brake
t, b = graduated_brake(8.0, 6.0, 0.0, 0.0, 0.5)
assert b >= 0.5

# 9) 緊急內圈(gap<=d0):果斷全力煞、收油——止住低速潛行(94% 失效),不靠溫和分級
t, b = graduated_brake(DSAFE_D0 - 0.5, 0.5, 0.0, 0.6, 0.0)   # 低速潛行進到 d0 內
assert b >= 1.0 and t == 0.0
t, b = graduated_brake(DSAFE_D0 - 2.0, 1.5, 0.0, 0.9, 0.0)   # 更深入仍全力
assert b >= 1.0 and t == 0.0

print("dynamic safety controller tests PASS")
