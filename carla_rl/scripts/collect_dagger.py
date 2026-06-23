"""Recovery-DAgger collection (V11). Classic DAgger can't query the CARLA TM for
arbitrary states, so instead: run the current STUDENT (hier-BC = pure-pursuit
steer + BC pedals) for a random number of steps so it drifts into its own
(often over-speed) states, then HAND OFF to the fast TM autopilot, which
recovers and drives on while we record (obs_317, TM_action). Those labels live
in the student's state distribution — exactly what DAgger needs to fix the
longitudinal covariate shift (the reckless over-speed that causes collisions).

Saves recovery (obs, action) to data/dagger_iter<N>.npz. Resilient + incremental.

Usage: python -m carla_rl.scripts.collect_dagger <student_ckpt> <out_npz> [episodes]
"""

import sys

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.utils.server import ensure_server, restart_server
from carla_rl.wrappers.gym_compat import CarlaGymEnv
from carla_rl.wrappers.lateral_control import PurePursuitController

SPEED_PCT = -30.0          # fast TM expert (same as the demos / ceiling test)
RECOVERY_WINDOW = 60       # record only the first N steps after handoff: the
                           # brake-and-stabilize maneuver lives in the student's
                           # drifted (over-speed) state distribution; recording
                           # the long cruising tail would just dilute the
                           # correction signal with autopilot_fast duplicates

# DANGER-TRIGGERED handoff. A random-K handoff only catches states the student
# already SURVIVED — the truly dangerous ones end the episode in a collision
# during drift, yielding no label. So instead run the student until it enters a
# plausibly-pre-collision state (over-speed AND a close/closing lead, or
# over-speed approaching a junction), THEN hand off so the expert demonstrates
# the avoidance from exactly the state that was about to crash.
DANGER_SPEED = 6.0   # m/s — only over-speed states are interesting
DANGER_GAP = 12.0    # m — a lead this close at speed is a brake candidate
DANGER_TTC = 3.0     # s — or one we'd hit this soon if closing
DANGER_JUNC = 0.25   # obs[311] normalized — approaching an (occupiable) junction
MIN_DRIFT = 10       # let the student build up state before we look
MAX_DRIFT = 300      # no danger by here: open road, abandon (base demos cover it)


def _danger(obs):
    """True when the student is over-speed and a collision is plausibly near."""
    if float(obs[3]) < DANGER_SPEED:
        return False
    gap, clos = float(obs[7]), float(obs[8])
    if gap > 0.0 and (gap < DANGER_GAP or (clos > 0.5 and gap / clos < DANGER_TTC)):
        return True
    return float(obs[311]) < DANGER_JUNC


def main():
    student_ckpt = sys.argv[1] if len(sys.argv) > 1 else r'carla_rl\checkpoints\bc_autopilot.pth'
    out = sys.argv[2] if len(sys.argv) > 2 else r'data\dagger_iter1.npz'
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    from carla_rl.scripts.run_eval import BCPolicy
    student = BCPolicy(checkpoint=student_ckpt)        # provides BC pedals
    controller = PurePursuitController()

    params = make_params(number_of_vehicles=50, desired_speed=10, max_time_episode=1000)
    ensure_server(port=params['port'], town=params['town'], nullrhi=True)

    def build():
        return CarlaGymEnv(params, use_route=False, predictive_obs=True)

    env = build()
    obs_buf, act_buf = [], []
    ep = 0
    restarts = 0
    while ep < episodes:
        try:
            obs = env.reset()
            controller.reset()
            done = False
            triggered = False
            # --- student drift phase (hier-BC) until a danger state appears ---
            drift = 0
            for drift in range(1, MAX_DRIFT + 1):
                if done:
                    break
                bc = student.act(obs, env)
                steer = controller.steer(obs)
                obs, _, done, _ = env.step(
                    np.array([bc[0], steer, bc[2]], dtype=np.float32))
                if drift >= MIN_DRIFT and _danger(obs):
                    triggered = True
                    break
            rec = 0
            hspd = float(obs[3])
            hgap = float(obs[7])
            if triggered and not done:
                # --- hand off to the fast TM, record its avoidance maneuver ---
                import carla
                env.ego.set_autopilot(True)
                carla.Client('localhost', params['port']).get_trafficmanager(8000) \
                    .vehicle_percentage_speed_difference(env.ego, SPEED_PCT)
                w = env.ego.get_control()  # warmup tick (TM takes the wheel)
                obs, _, done, _ = env.step(
                    np.array([w.throttle, w.steer, w.brake], dtype=np.float32))
                while not done and rec < RECOVERY_WINDOW:
                    ctrl = env.ego.get_control()
                    a = np.array([ctrl.throttle, ctrl.steer, ctrl.brake], dtype=np.float32)
                    obs_buf.append(np.asarray(obs, dtype=np.float32))
                    act_buf.append(a)
                    rec += 1
                    obs, _, done, _ = env.step(a)
            ep += 1
            tag = 'crash-in-drift' if (triggered and done and rec == 0) else \
                  ('triggered' if triggered else 'no-danger')
            print(f'ep {ep}/{episodes}: drift {drift} [{tag}] handoff spd={hspd:.1f} '
                  f'gap={hgap:.1f}, recovery {rec} labels, total {len(obs_buf)}', flush=True)
            if ep % 10 == 0 and obs_buf:
                np.savez(out, obs=np.asarray(obs_buf), act=np.asarray(act_buf))
        except RuntimeError as exc:
            restarts += 1
            print(f'[dagger] CARLA error: {exc}; restart {restarts}', flush=True)
            try:
                env.close()
            except Exception:
                pass
            restart_server(port=params['port'], nullrhi=True)
            env = build()

    env.close()
    np.savez(out, obs=np.asarray(obs_buf), act=np.asarray(act_buf))
    print(f'DAGGER COLLECT DONE: {len(obs_buf)} recovery labels from {episodes} eps, '
          f'{restarts} restarts -> {out}', flush=True)


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
