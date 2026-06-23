"""診斷 avoid 策略的碰撞「卡在哪」:感知問題還是策略問題?
跑策略,維護最近 N 步的 (speed, min_ttc, throttle, brake);每次撞車就 dump 這個視窗。
判讀:撞前 min_ttc 若早就變低(感知看到了)但 throttle 仍高/brake 低 → 策略沒煞(策略問題);
若 min_ttc 一直接近 1(沒看到)直到撞上 → 感知盲區(感知問題)。

用法: python -m carla_rl.scripts.diagnose_avoid_collisions <ckpt> [episodes]
"""
import sys
from collections import deque

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.utils.server import ensure_server, restart_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import HybridSteerWrapper

WIN = 15


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r'carla_rl\checkpoints\sac_avoid_v3.pth'
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    from carla_rl.scripts.run_eval import SACPolicy
    policy = SACPolicy(checkpoint=ckpt)

    params = make_params(number_of_vehicles=50, desired_speed=8, max_time_episode=3000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)

    def build():
        env = CarlaGymEnv(params, use_route=False, predictive_obs=True, risk_obs=True)
        return HybridSteerWrapper(env, throttle_ema=0.4)   # 純 RL,無安全層

    env = build()
    ncol = 0
    seen_then_nobrake = 0   # 撞前看到(min_ttc 低)卻沒煞
    blind = 0               # 撞前沒看到(min_ttc 高)
    ep = 0
    while ep < episodes:
        try:
            obs = env.reset()
            if hasattr(policy, 'on_episode_start'):
                policy.on_episode_start(env)
            win = deque(maxlen=WIN)
            done = False
            while not done:
                a = policy.act(obs, env)
                obs, _, done, info = env.step(a)
                applied = info.get('applied_action', a)
                win.append((float(obs[3]), float(info.get('min_ttc', 1.0)),
                            float(applied[0]), float(applied[2])))
                if info.get('is_collision'):
                    ncol += 1
                    pre = list(win)
                    # 撞前 8 步的最小 min_ttc + 當時的油門/煞車
                    last8 = pre[-8:]
                    min_ttc_pre = min(t for _, t, _, _ in last8)
                    mean_thr = np.mean([th for _, _, th, _ in last8])
                    mean_brk = np.mean([bk for _, _, _, bk in last8])
                    saw = min_ttc_pre < 0.7      # 感知有警示(門檻同 avoid_v3)
                    if saw and mean_brk < 0.3:
                        seen_then_nobrake += 1
                    elif not saw:
                        blind += 1
                    print(f'[collision {ncol}] 撞前8步 min_ttc={min_ttc_pre:.2f} '
                          f'mean_thr={mean_thr:.2f} mean_brk={mean_brk:.2f} '
                          f'{"SAW-but-no-brake" if (saw and mean_brk<0.3) else ("BLIND" if not saw else "saw+braked")}',
                          flush=True)
                    break
            ep += 1
        except RuntimeError as exc:
            print(f'[diag] CARLA error: {exc}; restart', flush=True)
            try:
                env.close()
            except Exception:
                pass
            restart_server(port=params['port'], nullrhi=True)
            env = build()
    env.close()
    print(f'\nDIAG DONE: {ncol} collisions over {episodes} eps | '
          f'SAW-but-no-brake={seen_then_nobrake} (策略問題), BLIND={blind} (感知問題)', flush=True)


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack
    run_with_big_stack(main)
