"""Evaluate a baseline policy in EasyCarla-RL.

Usage (CARLA server must be running):
    python -m carla_rl.scripts.run_eval --policy random --episodes 5
    python -m carla_rl.scripts.run_eval --policy autopilot --episodes 5 --video 2
    python -m carla_rl.scripts.run_eval --policy dql --episodes 5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from carla_rl.configs.env_params import make_params
from carla_rl.configs.reward_config import REWARD_PRESETS
from carla_rl.evaluation import AutopilotPolicy, RandomPolicy, evaluate
from carla_rl.utils.server import ensure_server
from carla_rl.wrappers import CarlaGymEnv, ShapedRewardWrapper, SmoothActionWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / 'EasyCarla-RL' / 'example'


class DQLPolicy:
    """Pretrained Diffusion-QL agent shipped with EasyCarla-RL."""

    def __init__(self):
        sys.path.insert(0, str(EXAMPLE_DIR))
        import torch
        from agents.ql_diffusion import Diffusion_QL

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Diffusion_QL(
            state_dim=307,
            action_dim=3,
            max_action=1.0,
            device=device,
            discount=0.99,
            tau=0.005,
            eta=0.01,
            beta_schedule='vp',
            n_timesteps=5,
        )
        self.model.load_model(str(EXAMPLE_DIR / 'params_dql'), id=200)
        print(f"Loaded pretrained DQL checkpoint (device={device})")

    def on_episode_start(self, env):
        env.ego.set_autopilot(False)

    def act(self, obs, env):
        # The model was trained on the 307-dim dataset layout; the wrapper's
        # extension features are appended after, so slice them off.
        return np.asarray(self.model.sample_action(obs[:307]), dtype=np.float32)


class BCPolicy:
    """Behavior-cloned actor (trained on the 307-dim dataset slice)."""

    DEFAULT_CHECKPOINT = REPO_ROOT / 'carla_rl' / 'checkpoints' / 'bc_actor.pth'

    def __init__(self, checkpoint=None, steer_gain=1.0):
        import torch
        from carla_rl.models import GaussianActor, ObsNormalizer

        ckpt = torch.load(checkpoint or self.DEFAULT_CHECKPOINT,
                          map_location='cpu', weights_only=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = GaussianActor(ckpt['obs_dim'], hidden=tuple(ckpt['hidden']))
        self.actor.load_state_dict(ckpt['actor'])
        self.actor.to(self.device).eval()
        self.normalizer = ObsNormalizer.from_state_dict(ckpt['normalizer'])
        self.obs_dim = ckpt['obs_dim']
        # MSE regression-to-mean attenuates steering to ~0.52x the expert's;
        # the gain compensates globally at inference
        self.steer_gain = steer_gain
        self.torch = torch
        print(f"Loaded BC checkpoint (val_mse={ckpt['val_mse']:.5f}, epoch {ckpt['epoch']}, "
              f"steer_gain={steer_gain})")

    def on_episode_start(self, env):
        env.ego.set_autopilot(False)

    def act(self, obs, env):
        obs_n = self.normalizer(obs[: self.obs_dim])
        with self.torch.no_grad():
            obs_t = self.torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
            action = self.actor.mean_action(obs_t).squeeze(0).cpu().numpy()
        action[1] = np.clip(action[1] * self.steer_gain, -1.0, 1.0)
        return action


class SACPolicy:
    """Fine-tuned SAC actor (lives on the full 312-dim observation)."""

    DEFAULT_CHECKPOINT = REPO_ROOT / 'carla_rl' / 'checkpoints' / 'sac_latest.pth'

    def __init__(self, checkpoint=None):
        import torch
        from carla_rl.agents.sac import SAC

        path = checkpoint or self.DEFAULT_CHECKPOINT
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        hidden = tuple(ckpt.get('hidden', (256, 256)))
        meta = ckpt.get('meta', {})
        # the env stack must match training: a smooth-trained agent expects
        # the prev-action dims, the steer rate limit, and the steer EMA filter;
        # a hierarchical agent expects pure-pursuit steering
        self.smooth_steer = float(meta.get('smooth_steer', 0.0))
        self.steer_ema = float(meta.get('steer_ema', 0.0))
        self.lateral_control = bool(meta.get('lateral_control', False))
        self.use_route = bool(meta.get('use_route', False))
        self.throttle_ema = float(meta.get('throttle_ema', 0.0))
        self.safety_brake = bool(meta.get('safety_brake', False))
        self.safety_planner = bool(meta.get('safety_planner', False))
        self.signed_pedal = bool(meta.get('signed_pedal', False))
        self.ckpt_obs_dim = int(ckpt['obs_dim'])
        if self.signed_pedal:
            # 一維 signed-pedal:actor 1 維、critic 輸入 obs+1、邊界 [-1,1]
            self.agent = SAC(int(ckpt['obs_dim']), act_dim=1, hidden=hidden,
                             critic_layernorm=ckpt.get('critic_layernorm', False),
                             action_low=[-1.0], action_high=[1.0])
        else:
            self.agent = SAC(int(ckpt['obs_dim']), hidden=hidden,
                             critic_layernorm=ckpt.get('critic_layernorm', False))
        self.agent.load(path)
        print(f'Loaded SAC checkpoint (hidden={hidden}, obs_dim={ckpt["obs_dim"]}, '
              f'signed_pedal={self.signed_pedal}, steps={meta.get("env_steps")})')

    def on_episode_start(self, env):
        env.ego.set_autopilot(False)

    def act(self, obs, env):
        return self.agent.act(obs, deterministic=True)


POLICIES = {
    'random': RandomPolicy,
    'autopilot': AutopilotPolicy,
    'dql': DQLPolicy,
    'bc': BCPolicy,
    'sac': SACPolicy,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=sorted(POLICIES), required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--video', type=int, default=1, help='record first N episodes')
    parser.add_argument('--town', default='Town03')
    parser.add_argument('--vehicles', type=int, default=100)
    parser.add_argument('--desired-speed', type=float, default=None,
                        help='override env desired_speed (match training, e.g. 12 for v4)')
    parser.add_argument('--traffic', choices=['on', 'off'], default='off')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--shaped', action='store_true',
                        help='apply ShapedRewardWrapper (logs reward_env vs reward_shaped)')
    parser.add_argument('--steer-gain', type=float, default=1.0,
                        help='bc policy only: multiply predicted steer')
    parser.add_argument('--reward-preset', choices=sorted(REWARD_PRESETS), default='default',
                        help='shaping preset; a non-default preset implies --shaped')
    parser.add_argument('--use-route', action='store_true',
                        help='force global route planning on (otherwise inferred from a SAC '
                             'checkpoint\'s meta); enables the route_completion metric')
    parser.add_argument('--max-restarts', type=int, default=3,
                        help='server restarts tolerated before giving up (raise for long '
                             'multi-episode evals on this crash-prone source build)')
    parser.add_argument('--safety-brake', action='store_true',
                        help='hierarchical only: force-brake when a detected lead is imminent '
                             '(V7 longitudinal safety override against rear-ends)')
    parser.add_argument('--safety-planner', action='store_true',
                        help='hierarchical only: V9 predictive planner — cap speed to keep a '
                             'safe margin from all predicted vehicle paths over a 3 s horizon')
    parser.add_argument('--autopilot-speed-pct', type=float, default=None,
                        help='autopilot only: ego TM percentage-speed-difference (NEGATIVE = '
                             'faster than the limit); tests whether a faster expert keeps <0.05')
    parser.add_argument('--lateral-control', action='store_true',
                        help='force hierarchical pure-pursuit steering (override the policy meta) '
                             '— e.g. run a full-control BC with its weavy steering replaced')
    parser.add_argument('--checkpoint', default=None,
                        help='bc/sac policy: explicit checkpoint path')
    parser.add_argument('--nullrhi', action='store_true',
                        help='run on a headless server (stable; SkeletalMesh crash cannot '
                             'fire) — metrics only, videos would be black; forces --video 0')
    parser.add_argument('--out', default=None, help='output dir (default: carla_rl/logs/<ts>_<policy>)')
    args = parser.parse_args()
    if args.nullrhi:
        args.video = 0

    speed_override = {'desired_speed': args.desired_speed} if args.desired_speed else {}
    params = make_params(
        town=args.town,
        number_of_vehicles=args.vehicles,
        traffic=args.traffic,
        max_time_episode=args.max_steps,
        **speed_override,
    )
    out_dir = args.out or (
        REPO_ROOT / 'carla_rl' / 'logs'
        / f"{datetime.now():%Y%m%d_%H%M%S}_{args.policy}"
    )

    if args.nullrhi:
        # a rendered server may already be up; replace it so the run is stable
        from carla_rl.utils.server import restart_server
        restart_server(port=params['port'], town=args.town, nullrhi=True)
    else:
        ensure_server(port=params['port'], town=args.town)

    preset = REWARD_PRESETS[args.reward_preset]
    shaped = args.shaped or args.reward_preset != 'default'

    if args.policy == 'bc':
        policy = BCPolicy(checkpoint=args.checkpoint, steer_gain=args.steer_gain)
    elif args.policy == 'sac':
        policy = SACPolicy(checkpoint=args.checkpoint)
    else:
        policy = POLICIES[args.policy]()
    if args.policy == 'autopilot' and args.autopilot_speed_pct is not None:
        policy.speed_pct = args.autopilot_speed_pct

    smooth_steer = getattr(policy, 'smooth_steer', 0.0)
    steer_ema = getattr(policy, 'steer_ema', 0.0)
    lateral_control = getattr(policy, 'lateral_control', False) or args.lateral_control
    use_route = getattr(policy, 'use_route', False) or args.use_route
    throttle_ema = getattr(policy, 'throttle_ema', 0.0)
    safety_brake = getattr(policy, 'safety_brake', False) or args.safety_brake
    safety_planner = getattr(policy, 'safety_planner', False) or args.safety_planner
    signed_pedal = getattr(policy, 'signed_pedal', False)
    # 依 policy 的 obs_dim 明確配對環境結構(planner 與 obs 維度無關):
    #   312 = 307 原始 + 5 紅綠燈/路口
    #   317 = 312 + 5 predictive
    #   327 = 317 + 10 risk(5 扇區 × TTC/距離)← 最終避撞策略 sac_avoid_v5
    _pdim = getattr(policy, 'ckpt_obs_dim', None) or getattr(policy, 'obs_dim', 312)
    if _pdim == 312:
        predictive_obs = False
        risk_obs = False
    elif _pdim == 317:
        predictive_obs = True
        risk_obs = False
    elif _pdim == 327:
        predictive_obs = True
        risk_obs = True
    else:
        raise ValueError(
            f'不支援的 checkpoint obs_dim={_pdim};預期為 312、317 或 327。')

    def env_factory():
        env = CarlaGymEnv(params, use_route=use_route, predictive_obs=predictive_obs,
                          risk_obs=risk_obs)
        if lateral_control:
            from carla_rl.wrappers.lateral_control import HybridSteerWrapper
            env = HybridSteerWrapper(env, throttle_ema=throttle_ema,
                                     safety_brake=safety_brake,
                                     safety_planner=safety_planner,
                                     signed_pedal=signed_pedal)
        elif smooth_steer > 0.0:
            env = SmoothActionWrapper(env, max_steer_delta=smooth_steer,
                                      steer_ema=steer_ema)
        if shaped:
            env = ShapedRewardWrapper(env, preset)
        return env
    _, summary = evaluate(
        env_factory, policy, args.episodes, out_dir,
        run_name=args.policy, record_first_n=args.video,
        reward_config=preset.to_dict() if shaped else None,
        nullrhi=args.nullrhi, max_restarts=args.max_restarts,
    )

    print('\n=== Summary ===')
    for key, value in summary.items():
        if key != 'env_params':
            print(f'{key}: {value}')
    print(f'Logs written to {out_dir}')


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
