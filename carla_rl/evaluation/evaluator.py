"""Run a policy for N episodes and produce per-episode CSV, summary JSON, and videos.

A policy is any object with:
    on_episode_start(env) -> None   (optional hook)
    act(obs_vec, env) -> action [throttle, steer, brake]

The local CARLA server intermittently crashes during actor churn, so the
episode loop takes an env_factory: on a RuntimeError it restarts the server,
rebuilds the env, and re-runs the interrupted episode. Completed episode
results are never lost.
"""

import csv
import json
import threading
import time
from pathlib import Path

import numpy as np

from carla_rl.evaluation.metrics import EpisodeTracker, summarize
from carla_rl.evaluation.recorder import VideoRecorder
from carla_rl.utils.server import restart_server

# A healthy episode runs at most ~max_time_episode steps (~1-2 min wall). If
# reset/step makes NO progress for this long, the CARLA client has hung on an
# unresponsive server — it retries the socket forever instead of raising, so the
# RuntimeError restart path below never fires (this silently wedged a 30-ep eval
# for 40 min). The watchdog converts that hang into a RuntimeError so the server
# gets restarted. Generous, so only a genuine hang trips it.
EPISODE_TIMEOUT_S = 240
_BIG_STACK = 64 * 1024 * 1024  # torch + CARLA need it (see utils/bigstack)


class _WatchdogTimeout(RuntimeError):
    """An episode made no progress within the timeout (server hang)."""


def _run_watched(fn, timeout, *args, **kwargs):
    """Run fn on a big-stack daemon thread; raise _WatchdogTimeout if it does not
    return within `timeout` s. The hung worker is abandoned — the caller restarts
    the server, which drops the socket and lets that thread unwind and exit."""
    box = {}

    def worker():
        try:
            box['value'] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
            box['error'] = exc

    threading.stack_size(_BIG_STACK)
    t = threading.Thread(target=worker, name='episode-watchdog', daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise _WatchdogTimeout(
            f'no progress for {timeout}s — CARLA client hung on an unresponsive server'
        )
    if 'error' in box:
        raise box['error']
    return box['value']


class RandomPolicy:
    """Throttle-biased random actions: actually drives (and crashes), giving a
    meaningful floor. Pure uniform throttle+brake mostly cancels out and the
    car never moves."""

    def on_episode_start(self, env):
        env.ego.set_autopilot(False)

    def act(self, obs, env):
        return np.array(
            [
                np.random.uniform(0.3, 1.0),
                np.random.uniform(-0.6, 0.6),
                0.0,
            ],
            dtype=np.float32,
        )


class AutopilotPolicy:
    """Expert mode: CARLA traffic-manager autopilot drives, we log its controls.
    speed_pct sets the ego's TM percentage-speed-difference (NEGATIVE = faster
    than the limit); None = TM default (~30% below limit, ~7 m/s free)."""

    def __init__(self, speed_pct=None):
        self.speed_pct = speed_pct

    def on_episode_start(self, env):
        env.ego.set_autopilot(True)
        if self.speed_pct is not None:
            import carla
            tm = carla.Client('localhost', env.params['port']).get_trafficmanager(8000)
            tm.vehicle_percentage_speed_difference(env.ego, float(self.speed_pct))

    def act(self, obs, env):
        control = env.ego.get_control()
        return np.array([control.throttle, control.steer, control.brake], dtype=np.float32)


def _run_episode(env, policy, ep_index, recorder, video_path):
    obs = env.reset()
    if hasattr(policy, 'on_episode_start'):
        policy.on_episode_start(env)

    if recorder is not None:
        recorder.attach(env.world, env.ego, video_path)

    tracker = EpisodeTracker(ep_index, env.params['max_time_episode'])
    done = False
    info = {}
    try:
        while not done:
            action = policy.act(obs, env)
            obs, reward, done, info = env.step(action)
            tracker.update(action, reward, info, env.ego)
    finally:
        if recorder is not None:
            recorder.detach()
            frames = recorder.save()
            if frames:
                tracker.result.video = str(video_path)

    return tracker.finalize(info)


def evaluate(env_factory, policy, episodes, out_dir, run_name,
             record_first_n=1, max_restarts=3, reward_config=None, nullrhi=False,
             episode_timeout=EPISODE_TIMEOUT_S):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = env_factory()
    recorder = VideoRecorder(fps=int(round(1.0 / env.params['dt'])))

    # episodes are appended to the CSV as they complete — a native carla abort
    # can kill this process at any reset, and end-of-run writes lost the data
    csv_file = open(out_dir / f'{run_name}_episodes.csv', 'w', newline='')
    csv_writer = None

    results = []
    restarts = 0
    ep = 0
    while ep < episodes:
        recording = ep < record_first_n
        try:
            t0 = time.time()
            result = _run_watched(
                _run_episode, episode_timeout,
                env, policy, ep,
                recorder if recording else None,
                out_dir / f'{run_name}_ep{ep}.mp4' if recording else None,
            )
        except RuntimeError as exc:
            restarts += 1
            if restarts > max_restarts:
                print(f'[{run_name}] giving up after {max_restarts} server restarts')
                raise
            print(f'[{run_name}] CARLA error during episode {ep}: {exc}')
            print(f'[{run_name}] restarting server (attempt {restarts}/{max_restarts})...')
            # On a watchdog hang the server is unresponsive, so env.close() would
            # hang too — skip it. restart_server kills the server processes, which
            # also drops the abandoned worker thread's socket so it can exit.
            if not isinstance(exc, _WatchdogTimeout):
                try:
                    env.close()
                except Exception:
                    pass
            # 必須帶 town:否則跨城鎮 eval 崩潰重啟會退回預設 Town03(env 的 world-reuse
            # patch 會沿用,導致在錯誤地圖上評估)。
            restart_server(port=env.params['port'], town=env.params['town'],
                           nullrhi=nullrhi)
            env = env_factory()
            continue  # re-run the interrupted episode

        results.append(result)
        row = result.to_dict()
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()
        print(
            f"[{run_name}] episode {ep}: steps={result.steps} reward={result.total_reward:.1f} "
            f"collided={result.collided} off_road={result.off_road} success={result.success} "
            f"avg_speed={result.avg_speed:.2f} m/s ({time.time() - t0:.0f}s wall)"
        )
        ep += 1

    env.close()
    csv_file.close()

    summary = summarize(results)
    summary['run_name'] = run_name
    summary['server_restarts'] = restarts
    summary['env_params'] = env.params
    if reward_config is not None:
        summary['reward_config'] = reward_config

    with open(out_dir / f'{run_name}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return results, summary
