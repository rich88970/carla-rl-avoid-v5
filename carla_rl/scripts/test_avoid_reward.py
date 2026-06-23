"""avoid_v1 獎勵性質 assert 測試。跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.test_avoid_reward"""
from carla_rl.configs.reward_config import REWARD_PRESETS, survive_shaping

cfg = REWARD_PRESETS['avoid_v1']
EDS = 8.0


def ttc_pen(mt):
    """wrapper 套用的 TTC 懲罰公式(與 reward_shaping 一致)。"""
    return cfg.avoid_ttc_weight * max(0.0, cfg.avoid_ttc_floor_norm - mt)


# 1) preset 設定正確:保留 env 速度項、survive、有 TTC 與卡住懲罰
assert cfg.replace_speed_term is False and cfg.survive is True
assert cfg.avoid_ttc_weight < 0.0 and 0.0 < cfg.avoid_ttc_floor_norm < 1.0
assert cfg.survive_stuck_penalty < 0.0
# 2) survive 部分(progress)仍正確:停住=0、越快越高(8>0)
d0, _ = survive_shaping(0.0, 0.0, 1.0, cfg, EDS)
d8, _ = survive_shaping(8.0, 0.0, 1.0, cfg, EDS)
assert float(d0) == 0.0 and float(d8) > float(d0)
# 3) TTC 懲罰:安全(ttc_norm=1)不罰;門檻(0.4)不罰;低於門檻才罰且越危險越重
assert ttc_pen(1.0) == 0.0
assert ttc_pen(0.4) == 0.0
assert ttc_pen(0.1) < 0.0
assert ttc_pen(0.0) < ttc_pen(0.1)
print("avoid_v1 reward tests PASS")
