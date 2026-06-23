"""survive_shaping 純函式的 assert 測試(無需 CARLA)。

survive_v1 的最終設計(2026-06-17,經兩次迭代修正):
  - 保留 env 內建速度項(replace_speed_term=False)當前進獎勵;
  - survive_shaping 只加一個 progress 小獎勵 = min(v,8)/8(封頂 8、停住為 0、恆非負);
  - 不用 alive 加分(會被「停住farming」)、不用 crawl 扣分(會誘發早撞自殺);
  - 卡住一次性懲罰在 wrapper 套用(讀 info['stuck']),不在此純函式內。
本測試驗證 progress 的關鍵性質:停住=0(不可farming、不誘發自殺)、越快越高、封頂、恆非負。
跑:  .\.venv\Scripts\python.exe -m carla_rl.scripts.test_survive_reward
"""
from carla_rl.configs.reward_config import REWARD_PRESETS, survive_shaping

cfg = REWARD_PRESETS['survive_v1']
EDS = 8.0  # env desired_speed(survive_v1 搭配 8)


def d(v, gap, jd):
    """survive_shaping 的 delta。最終設計下 = progress = min(v,8)/8(恆非負)。"""
    delta, _ = survive_shaping(v, gap, jd, cfg, EDS)
    return float(delta)


# 1) 停住 = 0:不可被 farming(非正),也不誘發自殺(非負)
assert abs(d(0.0, 0.0, 1.0)) < 1e-9
# 2) 越快分越高(速度梯度 → 不會太慢):8 > 3 > 0
assert d(8.0, 0.0, 1.0) > d(3.0, 0.0, 1.0) > d(0.0, 0.0, 1.0)
# 3) 數值:progress = min(v,8)/8
assert abs(d(8.0, 0.0, 1.0) - 1.0) < 1e-6
assert abs(d(3.0, 0.0, 1.0) - 3.0 / 8) < 1e-6
# 4) v>=v_target 封頂(12 與 8 相同)
assert abs(d(12.0, 0.0, 1.0) - d(8.0, 0.0, 1.0)) < 1e-6
# 5) 恆非負(progress 永遠 >= 0 → 此項不會造成負的每步獎勵 → 不誘發自殺)
for v in (0.0, 1.0, 4.0, 7.9, 8.0, 20.0):
    assert d(v, 0.0, 1.0) >= 0.0
# 6) alive/crawl 已關閉:delta 與 gap/junction 無關(不再有防爬行扣分)
assert abs(d(3.0, 6.0, 0.2) - d(3.0, 0.0, 1.0)) < 1e-9
# 7) preset 設定正確:保留 env 速度項、關閉 alive/crawl、有卡住懲罰
assert cfg.replace_speed_term is False
assert cfg.survive_alive_bonus == 0.0
assert cfg.survive_crawl_penalty == 0.0
assert cfg.survive_stuck_penalty < 0.0

print("survive_shaping tests PASS")
