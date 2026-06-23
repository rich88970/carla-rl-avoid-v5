"""Offline checks for avoid_v6 safe-clear speed bonus.

Run:
    .\.venv\Scripts\python.exe -m carla_rl.scripts.test_avoid_free_speed_reward
"""

from carla_rl.configs.reward_config import REWARD_PRESETS, survive_shaping


cfg = REWARD_PRESETS["avoid_v6"]
EDS = 10.0


def shaped(v, gap=0.0, jd=1.0, min_ttc=1.0):
    delta, terms = survive_shaping(v, gap, jd, cfg, EDS, min_ttc=min_ttc)
    return float(delta), {k: float(v) for k, v in terms.items()}


# Preset intent: less broad progress than avoid_v5, plus gated free-road speed.
assert cfg.survive is True
assert cfg.survive_progress_weight == 2.5
assert cfg.avoid_free_speed_weight > 0.0

# Free road: speed bonus starts at the floor and saturates at target.
d_floor, t_floor = shaped(cfg.avoid_free_speed_floor)
d_mid, t_mid = shaped(7.5)
d_target, t_target = shaped(cfg.avoid_free_speed_target)
d_fast, t_fast = shaped(12.0)
assert t_floor["free_speed"] == 0.0
assert 0.0 < t_mid["free_speed"] < cfg.avoid_free_speed_weight
assert abs(t_target["free_speed"] - cfg.avoid_free_speed_weight) < 1e-9
assert abs(t_fast["free_speed"] - cfg.avoid_free_speed_weight) < 1e-9
assert d_target > d_mid > d_floor

# Any risk gate blocks only the extra speed bonus, leaving base progress.
_, t_lead = shaped(9.0, gap=8.0)
_, t_low_ttc = shaped(9.0, min_ttc=0.4)
_, t_junction = shaped(9.0, jd=0.2)
assert t_lead["free_speed"] == 0.0
assert t_low_ttc["free_speed"] == 0.0
assert t_junction["free_speed"] == 0.0

# Clear high-speed state must pay more than the same speed under risk.
d_clear, _ = shaped(9.0)
d_risky, _ = shaped(9.0, gap=8.0)
assert d_clear > d_risky

print("avoid_v6 safe-clear speed reward tests PASS")
