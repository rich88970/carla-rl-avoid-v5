"""avoid_v4 距離缺口懲罰(gap_deficit_penalty)性質測試。
跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.test_avoid_gap_reward"""
from carla_rl.configs.reward_config import REWARD_PRESETS, gap_deficit_penalty

cfg = REWARD_PRESETS['avoid_v4']


def gp(gap, v):
    return float(gap_deficit_penalty(gap, v, cfg))


# 1) preset:= avoid_v3 基底 + 距離缺口懲罰開啟,floor=7
assert cfg.survive is True and cfg.replace_speed_term is False
assert cfg.avoid_gap_weight < 0.0 and cfg.avoid_gap_floor == 7.0
assert cfg.avoid_ttc_weight < 0.0 and cfg.avoid_collision_penalty < 0.0  # 保留 v3 訊號
# 2) 無 on-path 前車(gap<=0)→ 不罰
assert gp(0.0, 6.0) == 0.0
assert gp(-1.0, 6.0) == 0.0
# 3) 前車距 >= floor(7m)→ 不罰
assert gp(7.0, 6.0) == 0.0
assert gp(10.0, 6.0) == 0.0
# 4) 前車距 < floor 且移動中 → 罰,且越近越重
assert gp(5.0, 6.0) < 0.0
assert gp(3.0, 6.0) < gp(5.0, 6.0)        # 3m 比 5m 更該罰(更負)
# 5) 速度淡出:同樣近距,停住(v=0)不罰,移動越快罰越重(min(1,v/2) 上限在 v=2 飽和)
assert gp(3.0, 0.0) == 0.0                # 被卡住停住 → 不罰(避免 escape-or-suicide)
assert gp(3.0, 1.0) > gp(3.0, 2.0)        # v=1(scale .5)比 v=2(scale 1)罰得輕(較不負)
assert gp(3.0, 2.0) == gp(3.0, 4.0)       # v>=2 飽和,scale 都是 1
# 6) 數值正確:gap=4, v=4 → weight*1*(7-4)=weight*3
assert abs(gp(4.0, 4.0) - cfg.avoid_gap_weight * 3.0) < 1e-9
# 7) 與 TTC 互補:gap 罰只看距離(不需逼近速度),近距低速也會觸發(正是 TTC 盲區)
assert gp(2.0, 0.6) < 0.0                 # 1m/s 內潛行頂前車的情境 → 有罰(TTC 會誤判為安全)
print("avoid_v4 gap-deficit reward tests PASS")
