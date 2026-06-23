"""3D SAC 油門/煞車投影回歸測試(project_3d_pedals)。

確保舊三維 checkpoint 的踏板互斥規則正確:依大小保留較大踏板、平手偏油門、邊界裁切。
修正前的「brake>0 即清零油門」會把 Gaussian actor 的微量煞車放大成全斷油 → 50/100/150 車零移動。
"""

from carla_rl.wrappers.lateral_control import project_3d_pedals


def check(throttle, brake, exp_t, exp_b):
    at, ab = project_3d_pedals(throttle, brake)
    assert abs(at - exp_t) < 1e-9, (throttle, brake, at, ab)
    assert abs(ab - exp_b) < 1e-9, (throttle, brake, at, ab)


check(0.90, 0.05, 0.90, 0.00)   # 高油門、小煞車 → 保留油門(修正的核心案例)
check(0.10, 0.80, 0.00, 0.80)   # 小油門、高煞車 → 保留煞車
check(0.40, 0.40, 0.40, 0.00)   # 平手 → 沿 CarlaGymEnv 規則偏油門
check(1.50, -0.20, 1.00, 0.00)  # 邊界裁切
check(-0.50, 1.30, 0.00, 1.00)  # 邊界裁切
check(0.00, 0.00, 0.00, 0.00)   # 皆零

print('3D pedal projection tests PASS')
