"""Data-driven diagnosis of V6 collisions + freezes (we can't watch video, so
log the STATE). Attaches a collision sensor that records WHAT was hit and from
which DIRECTION (ego frame), plus per-step speed / applied pedals / front_gap /
distance-to-junction / nearest-vehicle, then classifies every collision and
every stuck episode.

Collision questions answered:
  - direction of impact (front / rear / side) -> rear-end vs cross-traffic vs being hit
  - was it at a junction? (obs[311] < junction gate)
  - was the ego moving / accelerating into it, and was there a lead it ignored?
  - was the other actor a vehicle or a static object (wall/pole = steering fault)?

Stuck questions answered:
  - is a vehicle very close but OUTSIDE the forward 20 m corridor (boxed in), or
    is the road clear and the policy simply won't throttle (open-road freeze)?
  - is the policy braking or throttling-but-blocked during the freeze?

Usage: python -m carla_rl.scripts.diagnose_collisions [episodes]
"""

import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.utils.server import ensure_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import HybridSteerWrapper, PurePursuitController

CKPT = r'carla_rl\logs\sac_v6\sac_step55000.pth'
JUNCTION_GATE = 0.4   # obs[311] below this == near a junction (matches cruise_safe)

# Findings are APPENDED here so samples survive the un-catchable libcarla abort:
# an orchestrator re-runs this script across server crashes and they accumulate.
FINDINGS = Path(__file__).resolve().parents[1] / 'logs' / 'diag_findings.jsonl'


def _append_finding(rec):
    with open(FINDINGS, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def nearest_vehicle(obs):
    """Return (local_x, local_y, distance) of the closest real nearby vehicle."""
    nb = np.asarray(obs[251:271], dtype=np.float64).reshape(5, 4)
    d = np.hypot(nb[:, 0], nb[:, 1])
    pad = (nb[:, 0] == 0) & (nb[:, 1] == 0) & (nb[:, 3] == 0)
    d[pad] = 1e9
    i = int(np.argmin(d))
    return float(nb[i, 0]), float(nb[i, 1]), float(d[i])


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    from carla_rl.scripts.run_eval import SACPolicy
    policy = SACPolicy(checkpoint=CKPT)

    params = make_params(number_of_vehicles=50, desired_speed=10, max_time_episode=1000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)

    def build():
        e = CarlaGymEnv(params, use_route=policy.use_route)
        if policy.lateral_control:
            e = HybridSteerWrapper(e, PurePursuitController(),
                                   throttle_ema=policy.throttle_ema)
        return e

    env = build()

    collisions = []   # dicts
    stucks = []       # dicts
    n_coll = n_stuck = n_arr = n_other = 0

    captured = {'event': None}

    def attach_sensor():
        import carla
        captured['event'] = None
        bp = env.world.get_blueprint_library().find('sensor.other.collision')
        sensor = env.world.spawn_actor(bp, carla.Transform(), attach_to=env.ego)

        def on_coll(ev):
            if captured['event'] is not None:
                return  # first contact only
            tf = env.ego.get_transform()
            yaw = np.deg2rad(tf.rotation.yaw)
            o = ev.other_actor.get_transform().location
            dx, dy = o.x - tf.location.x, o.y - tf.location.y
            lx = np.cos(-yaw) * dx - np.sin(-yaw) * dy
            ly = np.sin(-yaw) * dx + np.cos(-yaw) * dy
            captured['event'] = {
                'other': ev.other_actor.type_id,
                'lx': float(lx), 'ly': float(ly),
            }

        sensor.listen(on_coll)
        return sensor

    ep = 0
    restarts = 0
    while ep < episodes:
        try:
            obs = env.reset()
            policy.on_episode_start(env)
            sensor = attach_sensor()
            hist = deque(maxlen=40)
            done, info = False, {}
            while not done:
                a = policy.act(obs, env)
                obs, _, done, info = env.step(a)
                ap = info.get('applied_action', a)
                nlx, nly, nd = nearest_vehicle(obs)
                hist.append({
                    'speed': float(obs[3]), 'thr': float(ap[0]), 'brk': float(ap[2]),
                    'gap': float(info.get('front_gap', 0.0)), 'junc': float(obs[311]),
                    'nlx': nlx, 'nly': nly, 'nd': nd,
                })
            try:
                sensor.stop(); sensor.destroy()
            except Exception:
                pass
        except RuntimeError as exc:
            restarts += 1
            print(f'[diag] CARLA error ep{ep}: {exc}; restart {restarts}', flush=True)
            try:
                env.close()
            except Exception:
                pass
            from carla_rl.utils.server import restart_server
            restart_server(port=params['port'], nullrhi=True)
            env = build()
            continue

        last = hist[-1]
        if info.get('is_collision'):
            n_coll += 1
            ev = captured['event'] or {}
            lx, ly = ev.get('lx', 0.0), ev.get('ly', 0.0)
            if abs(lx) >= abs(ly):
                direction = 'FRONT' if lx > 0 else 'REAR'
            else:
                direction = 'SIDE'
            recent = list(hist)[-10:]
            collisions.append({
                'ep': ep, 'dir': direction, 'other': ev.get('other', '?'),
                'at_junction': last['junc'] < JUNCTION_GATE,
                'speed': last['speed'],
                'mean_thr': np.mean([h['thr'] for h in recent]),
                'mean_brk': np.mean([h['brk'] for h in recent]),
                'had_lead_frac': np.mean([h['gap'] > 0 for h in recent]),
                'lx': lx, 'ly': ly,
            })
            c = collisions[-1]
            _append_finding({'type': 'collision', **{k: (float(v) if isinstance(v, np.floating) else v)
                                                      for k, v in c.items()}})
            print(f"ep{ep:>2} COLLISION  {direction:5s} other={c['other']:<22s} "
                  f"junction={c['at_junction']} v={c['speed']:.1f} "
                  f"thr={c['mean_thr']:.2f} brk={c['mean_brk']:.2f} "
                  f"had_lead={c['had_lead_frac']:.0%}", flush=True)
        elif info.get('stuck'):
            n_stuck += 1
            frozen = [h for h in hist if h['speed'] < 0.3]
            nd = np.min([h['nd'] for h in frozen]) if frozen else 999
            thr = np.mean([h['thr'] for h in frozen]) if frozen else 0
            brk = np.mean([h['brk'] for h in frozen]) if frozen else 0
            kind = 'boxed_in(<6m)' if nd < 6 else ('near(<10m)' if nd < 10 else 'OPEN_ROAD')
            stucks.append({'ep': ep, 'nearest_d': nd, 'thr': thr, 'brk': brk, 'kind': kind})
            _append_finding({'type': 'stuck', 'nearest_d': float(nd),
                             'thr': float(thr), 'brk': float(brk), 'kind': kind})
            print(f"ep{ep:>2} STUCK      {kind:14s} nearest_veh={nd:.1f}m "
                  f"thr={thr:.2f} brk={brk:.2f}", flush=True)
        elif info.get('reached_destination'):
            n_arr += 1
        else:
            n_other += 1
        ep += 1

    env.close()

    print('\n==================== DIAGNOSIS ====================')
    print(f'episodes={episodes}  collisions={n_coll}  stuck={n_stuck}  '
          f'arrived={n_arr}  other={n_other}')
    if collisions:
        dirs = Counter(c['dir'] for c in collisions)
        at_junc = sum(c['at_junction'] for c in collisions)
        statics = sum(1 for c in collisions if not c['other'].startswith('vehicle'))
        print(f'\nCOLLISIONS ({n_coll}):')
        print(f'  direction: {dict(dirs)}')
        print(f'  at junction: {at_junc}/{n_coll}    static-object hits: {statics}/{n_coll}')
        print(f'  mean speed at impact: {np.mean([c["speed"] for c in collisions]):.1f} m/s')
        print(f'  had a lead in last 10 steps: '
              f'{np.mean([c["had_lead_frac"] for c in collisions]):.0%}')
        print(f'  mean throttle/brake before impact: '
              f'{np.mean([c["mean_thr"] for c in collisions]):.2f} / '
              f'{np.mean([c["mean_brk"] for c in collisions]):.2f}')
    if stucks:
        kinds = Counter(s['kind'] for s in stucks)
        print(f'\nSTUCK ({n_stuck}):')
        print(f'  kind: {dict(kinds)}')
        print(f'  mean nearest vehicle: {np.mean([s["nearest_d"] for s in stucks]):.1f} m')
        print(f'  mean throttle/brake while frozen: '
              f'{np.mean([s["thr"] for s in stucks]):.2f} / '
              f'{np.mean([s["brk"] for s in stucks]):.2f}')
    print('DIAGNOSIS DONE')


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
