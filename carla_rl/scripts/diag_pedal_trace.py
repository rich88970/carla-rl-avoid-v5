"""逐步油門診斷:直接記錄「策略原始輸出油門/煞車」對比「安全層套用後」每一步的值。

目的:對 100 車卡住給「直接證據」(非只用消去法)。若**關掉安全層(DSAFE off)後策略原始
油門仍 ≈ 0**,即證明僵住來自 SAC 策略本身(在比訓練更高的車流密度為分布外、收油),與安全層無關。

條件由環境變數控制(與 run_eval 一致):
  CARLA_DSAFE=1 CARLA_DSAFE_D0=7 CARLA_TTC_SHIELD=0.4  → 安全層開
  (不設這些)                                          → 安全層關(純策略)
伺服器路徑亦由環境變數提供(CARLA_UE4_EDITOR / CARLA_UPROJECT;見 INSTALL.md)。

用法(務必走 big-stack,torch+CARLA 同進程):
  python -m carla_rl.scripts.diag_pedal_trace --label dsafe_on  --vehicles 100 --episodes 3
"""

import argparse
import csv
import os

import numpy as np

from carla_rl.utils.bigstack import run_with_big_stack


def _run(args):
    from carla_rl.configs.env_params import make_params
    from carla_rl.scripts.run_eval import SACPolicy
    from carla_rl.utils.server import restart_server
    from carla_rl.wrappers.gym_compat import CarlaGymEnv
    from carla_rl.wrappers.lateral_control import HybridSteerWrapper

    params = make_params(town=args.town, number_of_vehicles=args.vehicles,
                         traffic='off', max_time_episode=args.max_steps)
    # headless(-nullrhi):高密度逐回合 churn 會觸發伺服器 SkeletalMesh 當機,無渲染可避開
    restart_server(port=params['port'], town=args.town, nullrhi=True)

    policy = SACPolicy(checkpoint=args.checkpoint)
    dsafe = os.environ.get('CARLA_DSAFE', '') == '1'
    print(f'[trace] label={args.label}  vehicles={args.vehicles}  DSAFE={"ON" if dsafe else "OFF"}')

    # 與 run_eval 的 SAC env_factory 完全一致(避撞策略:Pure-Pursuit 轉向 + throttle_ema 0.4)
    def make_env():
        env = CarlaGymEnv(params, use_route=False, predictive_obs=True, risk_obs=True)
        return HybridSteerWrapper(env, throttle_ema=policy.throttle_ema)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f'pedal_trace_{args.label}.csv')
    rows = []
    env = make_env()
    for ep in range(args.episodes):
        obs = env.reset()
        if hasattr(policy, 'on_episode_start'):
            policy.on_episode_start(env)
        done = False
        step = 0
        while not done:
            action = policy.act(obs, env)            # 策略原始輸出 [throttle, steer, brake]
            obs, _, done, info = env.step(action)
            g = lambda k: round(float(info.get(k, np.nan)), 4)
            # 動作鏈三段:raw_policy(actor 原始)→ policy(投影+EMA,未進安全層)→ final(實際執行)
            rows.append({
                'episode': ep, 'step': step,
                'speed': g('speed'),
                'raw_policy_throttle': g('raw_policy_throttle'),
                'raw_policy_brake': g('raw_policy_brake'),
                'policy_throttle': g('policy_throttle'),
                'policy_brake': g('policy_brake'),
                'final_throttle': g('final_throttle'),
                'final_brake': g('final_brake'),
                'raw_pedal': round(g('raw_policy_throttle') - g('raw_policy_brake'), 4),
                'applied_pedal': g('applied_pedal'),
                'front_gap': g('front_gap'),
                'min_ttc': g('min_ttc'),
                'shield_ttc': g('shield_ttc'),
                'dsafe_d': g('dsafe_d'),
                'ttc_shield_fired': int(bool(info.get('ttc_shield_fired', False))),
                'dsafe_active': int(bool(info.get('dsafe_active', False))),
            })
            step += 1

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 摘要:策略原始 vs 套用後油門 + 安全層介入比例,結論一目了然
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    n = len(rows)
    print(f'[trace] {args.label}: {n} steps over {args.episodes} eps')
    print(f'  raw_policy_throttle mean={g("raw_policy_throttle").mean():.4f}  '
          f'max={g("raw_policy_throttle").max():.4f}')
    print(f'  raw_policy_brake    mean={g("raw_policy_brake").mean():.4f}')
    print(f'  policy_throttle     mean={g("policy_throttle").mean():.4f}  (投影+EMA 後、未進安全層)')
    print(f'  final_throttle      mean={g("final_throttle").mean():.4f}  max={g("final_throttle").max():.4f}  (實際執行)')
    print(f'  ttc_shield_fired    frac={g("ttc_shield_fired").mean():.3f}')
    _dd = g("dsafe_d"); _dd = _dd[~np.isnan(_dd)]
    print(f'  dsafe_active        frac={g("dsafe_active").mean():.3f}  d_safe mean='
          f'{(_dd.mean() if _dd.size else float("nan")):.2f}')
    print(f'  speed (m/s)         mean={g("speed").mean():.4f}  max={g("speed").max():.4f}')
    print(f'[trace] CSV -> {csv_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, help='dsafe_on / dsafe_off 之類')
    ap.add_argument('--checkpoint', default='carla_rl/checkpoints/sac_avoid_v5.pth')
    ap.add_argument('--vehicles', type=int, default=100)
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--town', default='Town03')
    ap.add_argument('--max-steps', type=int, default=2000)
    ap.add_argument('--out', default='carla_rl/logs/pedal_trace')
    args = ap.parse_args()
    run_with_big_stack(lambda: _run(args))


if __name__ == '__main__':
    main()
