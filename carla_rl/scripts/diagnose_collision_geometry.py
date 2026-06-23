"""診斷 avoid 策略「撞到的是什麼方位」——前方/側向/後方,以判斷碰撞是否「ego 可由煞車避免」。

systematic-debugging Phase 1 證據蒐集:薄護盾把碰撞從 0.64 壓到 0.40 後就到頂,提早煞(0.6)
反而更差。最有力的未驗證假設:果斷硬煞把「前方追撞」換成了「被後車追撞」(CARLA 車流跟很近,
ego 突然 0.8 煞車會被後車撞)→ 煞車策略有天花板。要驗證就必須知道每次碰撞的「幾何方位」。

env 的碰撞感測器只存了 intensity,丟掉了 event.other_actor 與 event.normal_impulse。這支腳本
另外掛一個碰撞感測器,擷取「撞到誰、撞擊法向量」,並在撞車當下記錄對方車相對 ego 的方位/速度、
ego 當時車速與油門煞車、以及風險感知是否「看到」(min_ttc)。輸出前/側/後分類統計。

用法: python -m carla_rl.scripts.diagnose_collision_geometry <ckpt> [episodes]
"""
import sys
from collections import deque, Counter

import carla
import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.utils.server import ensure_server, restart_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import HybridSteerWrapper
from carla_rl.wrappers.risk_features import N_RISK_FEATURES

WIN = 8


def _bearing_class(deg):
    """對方車相對 ego 的方位角(0=正前、±180=正後)→ 前/側/後分類。"""
    a = abs(deg)
    if a <= 50.0:
        return 'FRONT'
    if a >= 130.0:
        return 'REAR'
    return 'SIDE'


def _ego_frame(ego, loc):
    """world 座標點 loc 轉到 ego frame (x 前, y 左)。回傳 (lx, ly)。"""
    tr = ego.get_transform()
    yaw = np.deg2rad(tr.rotation.yaw)
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dy = loc.x - tr.location.x, loc.y - tr.location.y
    return c * dx - s * dy, s * dx + c * dy


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r'carla_rl\checkpoints\sac_avoid_v3.pth'
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    from carla_rl.scripts.run_eval import SACPolicy
    policy = SACPolicy(checkpoint=ckpt)

    params = make_params(number_of_vehicles=50, desired_speed=8, max_time_episode=3000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)

    def build():
        env = CarlaGymEnv(params, use_route=False, predictive_obs=True, risk_obs=True)
        return HybridSteerWrapper(env, throttle_ema=0.4)   # 純 RL,無安全層

    env = build()
    holder = {'event': None}
    sensor = {'s': None}

    def attach_sensor():
        if sensor['s'] is not None:
            try:
                sensor['s'].stop()
                sensor['s'].destroy()
            except Exception:
                pass
        bp = env.world.get_blueprint_library().find('sensor.other.collision')
        s = env.world.spawn_actor(bp, carla.Transform(), attach_to=env.ego)
        s.listen(lambda e: holder.__setitem__('event', e))
        sensor['s'] = s

    cls_count = Counter()
    braked_front = 0
    front_lead_speed = []     # 前方碰撞時對方車速(判斷是否追撞靜止/慢車)
    ep = 0
    ncol = 0
    while ep < episodes:
        try:
            obs = env.reset()
            attach_sensor()
            holder['event'] = None
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
                    ev = holder['event']
                    ego = env.ego
                    espeed = float(obs[3])
                    pre = list(win)[-WIN:]
                    min_ttc_pre = min(t for _, t, _, _ in pre)
                    mean_brk = float(np.mean([bk for _, _, _, bk in pre]))
                    mean_thr = float(np.mean([th for _, _, th, _ in pre]))
                    cls, info_str = 'UNKNOWN', ''
                    if ev is not None and ev.other_actor is not None:
                        oa = ev.other_actor
                        olx, oly = _ego_frame(ego, oa.get_transform().location)
                        bearing = float(np.degrees(np.arctan2(oly, olx)))
                        cls = _bearing_class(bearing)
                        ovel = oa.get_velocity()
                        ospeed = float(np.hypot(ovel.x, ovel.y))
                        # 撞擊法向量(方向向量,只旋轉不平移)轉到 ego frame
                        imp = ev.normal_impulse
                        yaw = np.deg2rad(ego.get_transform().rotation.yaw)
                        c, s = np.cos(-yaw), np.sin(-yaw)
                        ilx, ily = c * imp.x - s * imp.y, s * imp.x + c * imp.y
                        imp_bear = float(np.degrees(np.arctan2(ily, ilx)))
                        info_str = (f'other="{oa.type_id.split(".")[-1]}" bearing={bearing:6.1f} '
                                    f'ospeed={ospeed:4.1f} imp_bear={imp_bear:6.1f}')
                        if cls == 'FRONT':
                            front_lead_speed.append(ospeed)
                            if mean_brk > 0.3:
                                braked_front += 1
                    cls_count[cls] += 1
                    print(f'[col {ncol:2d}] {cls:6s} ego_v={espeed:4.1f} min_ttc={min_ttc_pre:.2f} '
                          f'brk={mean_brk:.2f} thr={mean_thr:.2f} step={info.get("step","?")} | {info_str}',
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
    try:
        if sensor['s'] is not None:
            sensor['s'].stop(); sensor['s'].destroy()
    except Exception:
        pass
    env.close()
    print('\n=== COLLISION GEOMETRY ===', flush=True)
    print(f'episodes={episodes} collisions={ncol}', flush=True)
    for k in ('FRONT', 'SIDE', 'REAR', 'UNKNOWN'):
        print(f'  {k:7s}: {cls_count.get(k, 0)}', flush=True)
    print(f'FRONT 碰撞中 ego 有煞車(brk>0.3): {braked_front}/{cls_count.get("FRONT",0)}', flush=True)
    if front_lead_speed:
        fls = np.asarray(front_lead_speed)
        print(f'FRONT 對方車速: mean={fls.mean():.1f} <1m/s(近靜止)={int((fls<1).sum())}/{len(fls)}', flush=True)


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack
    run_with_big_stack(main)
