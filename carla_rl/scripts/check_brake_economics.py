"""Show that under cruise_v2, NOT braking near a lead pays more than braking
(diagnoses the user-observed 'never brakes for the car ahead')."""

from carla_rl.configs.reward_config import (
    CRUISE_V2_REWARD_CONFIG as cfg,
    cruise_shaping,
    speed_tent,
)

print('lead vehicle CLOSE at gap=8 m (d_safe at 6.5 m/s = 15.4 m, so this is tailgating):')
for label, v, rel in [('keep 6.5 m/s (no brake)', 6.5, 5.0),
                      ('brake to 3.0 m/s', 3.0, 1.5),
                      ('brake to 1.5 m/s', 1.5, 0.0)]:
    d, t = cruise_shaping(v, 8.0, 0.5, 0.5, cfg, 8.0, curve=0.0, rel_speed=rel)
    tent = float(speed_tent(v, 8.0))
    total = tent + float(d) - 0.5
    vdes = float(t['v_des'])
    print(f'  {label:26s} env_tent={tent:+.2f}  shaping={float(d):+.2f}  '
          f'TOTAL={total:+.2f}/step  (v_des={vdes:.1f})')
