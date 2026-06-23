"""Score a 30-episode eval against the PLAN_V2 strict acceptance bar.

Usage: python carla_rl/scripts/score_v2.py [eval_dir_name]   (default sac_v2_eval)
"""

import glob
import json
import sys

eval_dir = sys.argv[1] if len(sys.argv) > 1 else 'sac_v2_eval'
s = json.load(open(glob.glob(rf'carla_rl\logs\{eval_dir}\*_summary.json')[0]))

BAR = {
    'collision_rate': (0.0, '<='),
    'off_road_rate': (0.0, '<='),
    'success_rate': (0.9, '>='),
    'mean_speed_free': (5.5, '>='),
    'mean_speed_std_free': (0.8, '<='),
    'mean_pct_above_floor_free': (0.9, '>='),
    'mean_headway_violation_rate': (0.05, '<='),
    'mean_abs_steer_delta': (0.02, '<='),
    'mean_abs_pedal_delta': (0.05, '<='),
    'straight_mean_abs_yaw_rate': (1.85, '<='),
    'yaw_flip_rate': (0.01, '<='),
}

print(f"n = {s['episodes']} episodes")
passes = 0
for k, (t, op) in BAR.items():
    v = s[k]
    ok = v <= t if op == '<=' else v >= t
    passes += ok
    print(f"{'PASS' if ok else 'FAIL'}  {k}: {round(v, 4)}  (bar {op} {t})")
print(f'\n{passes}/{len(BAR)} criteria met')
print('extras: mean_steps', round(s['mean_steps'], 1),
      '| mean_avg_speed', round(s['mean_avg_speed'], 2),
      '| straight_steer_flip', round(s['straight_steer_flip_rate'], 3),
      '| straight_abs_steer_delta', round(s['straight_abs_steer_delta'], 4))
