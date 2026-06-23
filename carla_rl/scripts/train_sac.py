"""SAC fine-tuning of the BC policy in the live env.

Stage 1 of the curriculum: dataset conditions (Town03, 100 vehicles, lights
off) to minimize distribution shift from the BC warm start. The actor starts
as the BC policy (input layer widened 307 -> 312 with zero columns); a critic
warmup phase trains only the critic so random Q-values don't wreck the BC
actor in the first updates.

Server crashes (known SkeletalMesh assert) are survived: the env is rebuilt
and the interrupted episode discarded; buffer and networks live in this
process and are untouched.

Usage:
    python -m carla_rl.scripts.train_sac --total-steps 50000
    python -m carla_rl.scripts.train_sac --total-steps 2000 --critic-warmup 500   # smoke
Evaluate checkpoints separately: python -m carla_rl.scripts.run_eval --policy sac
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from carla_rl.agents.sac import SAC, ReplayBuffer
from carla_rl.configs.env_params import make_params
from carla_rl.configs.reward_config import REWARD_PRESETS
from carla_rl.data.offline_dataset import iter_prefill_transitions
from carla_rl.utils.server import ensure_server, restart_server
from carla_rl.wrappers import CarlaGymEnv, ShapedRewardWrapper, SmoothActionWrapper, obs_dim
from carla_rl.wrappers.traffic_light import NEUTRAL_FEATURES
from carla_rl.wrappers.predictive_features import NEUTRAL_PRED_FEATURES
from carla_rl.wrappers.risk_features import NEUTRAL_RISK_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / 'carla_rl' / 'checkpoints'
BC_CHECKPOINT = CHECKPOINT_DIR / 'bc_actor.pth'


def parse_curriculum(spec):
    """車流密度課程字串 -> [(step, vehicles), ...](依 step 由小到大排序)。
    例:"0:0,30000:50,60000:100" -> [(0,0),(30000,50),(60000,100)]。"""
    phases = []
    for part in spec.split(','):
        s, v = part.split(':')
        phases.append((int(s), int(v)))
    return sorted(phases)


def curriculum_vehicles(phases, step):
    """回傳目前 step 所屬階段的車流數(最後一個 step<=當前步 的階段值)。"""
    n = phases[0][1]
    for s, v in phases:
        if step >= s:
            n = v
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=50_000)
    parser.add_argument('--critic-warmup', type=int, default=5_000,
                        help='env steps with critic-only updates')
    parser.add_argument('--prefill', type=int, default=100_000,
                        help='dataset transitions preloaded into the replay buffer')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=400_000)
    parser.add_argument('--updates-per-step', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--bc-checkpoint', default=str(BC_CHECKPOINT))
    parser.add_argument('--resume', default=None,
                        help='resume from a SAC checkpoint instead of BC init '
                             '(replay buffer is not preserved; prefill still applies)')
    parser.add_argument('--render', action='store_true',
                        help='use a rendered server (default: -nullrhi headless ??'
                             'the SkeletalMesh render assert cannot fire without a renderer)')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--reward-preset', choices=sorted(REWARD_PRESETS), default='default',
                        help='reward shaping preset (cruise = Phase 4 objective)')
    parser.add_argument('--vehicles', type=int, default=None,
                        help='override number_of_vehicles (default: env default 100)')
    parser.add_argument('--target-entropy', type=float, default=None,
                        help='SAC entropy target (default: -act_dim = -3.0)')
    parser.add_argument('--critic-layernorm', action='store_true',
                        help='LayerNorm in the twin-Q critic — the fix for the V6 Q-value '
                             'divergence (critic_loss 90 -> 2000+). Free: the critic is not '
                             'BC-warm-started, so LN does not disturb the actor init.')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='multiply reward into the buffer (NOT the logged ep_return) to '
                             'keep Q targets O(1); V6 ran ~6.5 reward/step, inflating Q. '
                             'try 0.2 (~1/step)')
    parser.add_argument('--curriculum', default=None,
                        help='車流密度課程,格式 "step:veh,step:veh,...";例如 '
                             '"0:0,30000:50,60000:100"(由疏到密,跨步數門檻重建 env 切換車流數)。'
                             '空 = 用固定 --vehicles')
    parser.add_argument('--latest-checkpoint', default='sac_latest.pth',
                        help='rolling-latest filename under carla_rl/checkpoints; '
                             'give each run family its own (stage-1 owns sac_latest.pth)')
    parser.add_argument('--smooth-steer', type=float, default=0.0,
                        help='enable SmoothActionWrapper with this per-step steer rate '
                             'limit (e.g. 0.1); appends the applied action to the obs '
                             '(+3 dims) so smoothness penalties become learnable')
    parser.add_argument('--steer-ema', type=float, default=0.0,
                        help='low-pass gain on steering inside SmoothActionWrapper '
                             '(0.8 = 0.35 Hz cutoff at 10 Hz; kills weave frequency)')
    parser.add_argument('--bc-anchor', type=float, default=0.0,
                        help='TD3+BC behavior anchor: regularize the policy toward the '
                             'frozen BC actor. Value = RL/Q-term scale (smaller hugs BC '
                             'harder; 0 disables = plain SAC). The diagnosis showed plain '
                             'SAC turns the smooth BC policy jittery; this is the fix.')
    parser.add_argument('--lateral-control', action='store_true',
                        help='hierarchical: steering by PurePursuitController, RL learns '
                             'throttle/brake only (the fix for steering weave)')
    parser.add_argument('--use-route', action='store_true',
                        help='global route planning: waypoints follow a planned route to a '
                             'destination (the fix for roundabout-looping); episode ends as a '
                             'success on arrival')
    parser.add_argument('--throttle-ema', type=float, default=0.0,
                        help='hierarchical only: EMA-smooth the THROTTLE at execution (brake '
                             'stays instant) to kill the longitudinal lurch; 0 = off, ~0.4 typical')
    parser.add_argument('--safety-brake', action='store_true',
                        help='hierarchical only: longitudinal safety override (force-brake on an '
                             'imminent detected lead). Train WITH it so the policy co-adapts (V7).')
    parser.add_argument('--desired-speed', type=float, default=None,
                        help='override env desired_speed (cruise_fast uses 12 so the speed '
                             'tent peaks just above the 11 m/s target)')
    parser.add_argument('--traffic', choices=['on', 'off'], default=None,
                        help="紅綠燈:'on'=正常循環(學停紅燈必須,搭配 avoid_light_v1 獎勵);"
                             "'off'(預設 None→make_params 預設 off)=凍結綠燈(資料集條件)")
    parser.add_argument('--signed-pedal', action='store_true',
                        help='SAC 改一維 signed pedal u in [-1,1] (u>=0 油門、u<0 煞車);方向盤由 '
                             'Pure Pursuit。需配 --lateral-control。從 --resume 的三維 actor 轉一維'
                             '(只轉特徵層+縱向頭+normalizer,critic/buffer 全新)。修正動作介面不一致。')
    parser.add_argument('--buffer-save-every', type=int, default=10_000,
                        help='persist the replay buffer alongside the latest checkpoint '
                             'every N steps (0 disables)')
    parser.add_argument('--allow-bufferless-resume', action='store_true',
                        help='DANGEROUS: resume without a saved buffer (prefill-only '
                             'replay diverged three earlier runs)')
    args = parser.parse_args()

    run_name = args.run_name or f"sac_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = REPO_ROOT / 'carla_rl' / 'logs' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.vehicles is not None:
        overrides['number_of_vehicles'] = args.vehicles
    if args.desired_speed is not None:
        overrides['desired_speed'] = args.desired_speed
    if args.traffic is not None:
        overrides['traffic'] = args.traffic   # 'on' = 紅綠燈正常循環(學停紅燈必須)
    params = make_params(**overrides)  # defaults = dataset conditions
    reward_config = REWARD_PRESETS[args.reward_preset]
    latest_path = CHECKPOINT_DIR / args.latest_checkpoint
    smooth = args.smooth_steer > 0.0
    dim = obs_dim(params) + (SmoothActionWrapper.EXTRA_DIMS if smooth else 0)

    act_dim = 1 if args.signed_pedal else 3
    # signed-pedal 前置條件:方向盤必須交給 Pure Pursuit(--lateral-control),且不可再用 smooth-steer
    # (那是三維動作平滑器,與一維縱向不相容)。
    if args.signed_pedal and not args.lateral_control:
        raise SystemExit('--signed-pedal 必須搭配 --lateral-control')
    if args.signed_pedal and args.smooth_steer > 0.0:
        raise SystemExit('--signed-pedal 不可再搭配 --smooth-steer')
    if args.signed_pedal and args.bc_anchor > 0.0:
        # BC anchor 的 frozen actor 是三維;一維 actor 與三維 BC action 做 MSE 會錯誤 broadcast。
        raise SystemExit('--signed-pedal 暫不支援 --bc-anchor(三維 BC 與一維動作維度不符)')

    import torch
    if args.resume:
        ckpt_head = torch.load(args.resume, map_location='cpu', weights_only=False)
    resume_meta = ckpt_head.get('meta', {}) if args.resume else {}
    resume_is_signed = bool(resume_meta.get('signed_pedal', False))
    start_step = 0
    if args.signed_pedal and resume_is_signed:
        # --resume 已是一維 checkpoint → 真正續訓(直接 load,不可再轉一次,否則 2 列輸出頭會 IndexError)
        hidden = tuple(ckpt_head['hidden'])
        agent = SAC(dim, act_dim=1, hidden=hidden,
                    critic_layernorm=ckpt_head.get('critic_layernorm', False),
                    action_low=[-1.0], action_high=[1.0])
        agent.load(args.resume)
        start_step = int(resume_meta.get('env_steps', 0))
        print(f'signed-pedal SAC: 續訓一維 checkpoint {args.resume} '
              f'(hidden={hidden}, env_steps={start_step})')
    elif args.signed_pedal:
        # --resume 是三維 → 轉一維:只轉特徵層+縱向頭+normalizer;critic/target/alpha/buffer 全新。
        # 屬「全新訓練」(start_step=0),不接續舊三維步數。
        if not args.resume:
            raise SystemExit('--signed-pedal 需要 --resume <checkpoint>(三維轉一維或一維續訓)')
        hidden = tuple(ckpt_head['hidden'])
        agent = SAC(dim, act_dim=1, hidden=hidden,
                    critic_layernorm=args.critic_layernorm,
                    action_low=[-1.0], action_high=[1.0])
        agent.init_signed_pedal_from_3d(args.resume)
        print(f'signed-pedal SAC: 1D actor 由三維 {args.resume} 轉成(hidden={hidden}, '
              f'critic/buffer 全新, target_entropy={agent.target_entropy})')
    elif args.resume:
        hidden = tuple(ckpt_head['hidden'])
        agent = SAC(dim, hidden=hidden,
                    critic_layernorm=ckpt_head.get('critic_layernorm', False))
        agent.load(args.resume)
        start_step = int(ckpt_head.get('meta', {}).get('env_steps', 0))
        print(f'resumed SAC from {args.resume} (hidden={hidden}, env_steps={start_step})')
    else:
        bc_hidden = tuple(torch.load(args.bc_checkpoint, map_location='cpu',
                                     weights_only=False)['hidden'])
        agent = SAC(dim, hidden=bc_hidden, critic_layernorm=args.critic_layernorm)
        bc_info = agent.load_bc_checkpoint(args.bc_checkpoint)
        print(f"actor initialized from BC checkpoint (val_mse={bc_info['val_mse']:.5f}, "
              f"hidden={bc_hidden})")

    if args.target_entropy is not None:
        # SAC.load() never restores target_entropy, so this works on resume too
        agent.target_entropy = float(args.target_entropy)

    # behavior anchor — independent of resume/BC-init (the frozen BC actor is
    # always loaded from bc_checkpoint, never the evolving policy)
    if args.bc_anchor > 0.0:
        agent.set_bc_anchor(args.bc_checkpoint, args.bc_anchor)
        print(f'BC anchor enabled (RL scale={args.bc_anchor}, bc_dim={agent.bc_dim})')

    # A resume of an EVOLVED policy (past the initial warmup) gets a mandatory
    # critic-only re-adaptation window. A resume still inside the initial
    # warmup (policy ~= BC, actor never updated) must NOT keep pushing the
    # window forward on every early crash, or frequent pre-warmup crashes mean
    # the actor never starts learning — keep the absolute threshold there.
    critic_warmup = args.critic_warmup
    evolved_resume = args.resume and start_step >= args.critic_warmup
    if evolved_resume:
        critic_warmup = start_step + 5_000
        print(f'post-resume critic-only window until step {critic_warmup}')

    buffer = ReplayBuffer(args.buffer_size, dim, act_dim=act_dim)
    buffer_path = latest_path.with_suffix('.buffer.npz')
    signed_convert = args.signed_pedal and not resume_is_signed   # 三維→一維:動作維度不同
    if signed_convert:
        # 三維轉一維:舊 buffer 是三維 action,維度不符,絕不可載入 → 一律重新 prefill。
        print('signed-pedal 3D→1D:忽略任何既有 buffer,改以一維 prefill 重建。')
    elif args.signed_pedal and resume_is_signed:
        # 一維續訓:從「--resume 同名」的 .buffer.npz 載入(不是 --latest-checkpoint 名)。
        resume_buf = Path(args.resume).with_suffix('.buffer.npz')
        if resume_buf.exists():
            n = buffer.load(resume_buf)
            print(f'replay buffer restored: {n} transitions from {resume_buf.name}')
        elif args.allow_bufferless_resume:
            print(f'一維續訓但無 {resume_buf.name}:--allow-bufferless-resume → prefill 重建')
        else:
            raise SystemExit(f'REFUSING 一維續訓 without {resume_buf.name} '
                             f'(pass --allow-bufferless-resume to override)')
    elif args.resume:
        if buffer_path.exists():
            n = buffer.load(buffer_path)
            print(f'replay buffer restored: {n} transitions from {buffer_path.name}')
        elif args.allow_bufferless_resume or not evolved_resume:
            # still in warmup => policy ~= BC => prefill rebuild is safe (the
            # divergence risk is only an EVOLVED policy on a prefill-only buffer)
            print(f'bufferless resume at step {start_step} < warmup '
                  f'{args.critic_warmup}: policy still ~BC, rebuilding via prefill')
        else:
            raise SystemExit(
                f'REFUSING to resume an evolved policy without {buffer_path.name} '
                f'(see PLAN_V2; pass --allow-bufferless-resume to override)'
            )
    if buffer.size == 0 and args.prefill > 0:
        print(f'prefilling buffer with {args.prefill} dataset transitions...')
        for obs307, act, rew, next307, done, prev_act in iter_prefill_transitions(
            limit=args.prefill, steer_jerk_weight=reward_config.steer_jerk_weight,
            config=reward_config,
        ):
            # 307 dataset + 5 traffic-light + 5 V9 predictive (neutral offline;
            # the policy learns them online). expand_first_layer zero-pads the
            # 307-dim BC actor to obs_dim, so no BC re-clone is needed.
            # 307 dataset + 5 燈 + 5 V9 預測 + 10 risk(5 扇區;離線中性,policy 線上學);
            # obs_dim 預設含 risk=True → 327;expand_first_layer 把 307 BC 零填充到 327。
            obs_parts = [obs307, NEUTRAL_FEATURES, NEUTRAL_PRED_FEATURES, NEUTRAL_RISK_FEATURES]
            next_parts = [next307, NEUTRAL_FEATURES, NEUTRAL_PRED_FEATURES, NEUTRAL_RISK_FEATURES]
            if smooth:
                obs_parts.append(prev_act)   # zeros at episode starts
                next_parts.append(act)       # next state's prev action = this action
            obs_full = agent.normalize(np.concatenate(obs_parts))
            next_full = agent.normalize(np.concatenate(next_parts))
            if args.signed_pedal:
                # 三維 [throttle,steer,brake] → 一維 u = clip(throttle - brake, -1, 1)
                u = float(np.clip(act[0] - act[2], -1.0, 1.0))
                store_act = np.array([u], dtype=np.float32)
            else:
                store_act = act
            buffer.add(obs_full, store_act, rew * args.reward_scale, next_full, done)
        print(f'buffer size: {buffer.size}')

    with open(out_dir / 'config.json', 'w') as f:
        json.dump({'args': vars(args), 'env_params': params,
                   'reward_config': reward_config.to_dict()}, f, indent=2)

    log_path = out_dir / 'train_log.csv'
    # resume 時「接續」日誌(append),不要覆寫:否則每次崩潰 resume 都清空 train_log,
    # watchdog 永遠累積不到足夠集數判斷平台/下降(早停失效)。只有全新訓練才寫表頭。
    resume_log = bool(args.resume) and log_path.exists() and log_path.stat().st_size > 0
    log_file = open(log_path, 'a' if resume_log else 'w', newline='')
    logger = csv.DictWriter(log_file, fieldnames=[
        'step', 'episode', 'ep_return', 'ep_env_return', 'ep_steps',
        'critic_loss', 'q1_mean', 'actor_loss', 'alpha', 'entropy', 'restarts', 'wall_min',
        # 1D signed-pedal 監控(spec 9):平均速度、碰撞/闖紅燈次數、最小 TTC、raw/applied pedal、護盾介入
        'mean_speed', 'collision_count', 'red_light_violation_count', 'min_ttc',
        'raw_pedal_mean', 'applied_pedal_mean', 'shield_intervention_count',
    ])
    if not resume_log:
        logger.writeheader()
    # best-checkpoint(spec 9):優先「零闖紅燈 > 零碰撞 > 高均速」,不只存最後一個。
    best_key = None
    best_path = latest_path.with_name(latest_path.stem + '_best.pth')

    def make_meta(s):
        # single source of checkpoint meta — run_eval rebuilds the env stack
        # from these, so every save site MUST use this (drift here silently
        # makes eval load the wrong env: missing use_route / throttle_ema).
        return {'env_steps': s, 'smooth_steer': args.smooth_steer,
                'steer_ema': args.steer_ema, 'reward_preset': args.reward_preset,
                'lateral_control': args.lateral_control, 'use_route': args.use_route,
                'throttle_ema': args.throttle_ema, 'safety_brake': args.safety_brake,
                'signed_pedal': args.signed_pedal}

    # 車流密度課程:cur_vehicles 是可變容器,env_factory 每次重建時讀目前車流數。
    curriculum = parse_curriculum(args.curriculum) if args.curriculum else None
    cur_vehicles = {'n': params['number_of_vehicles']}

    def env_factory():
        p = dict(params)
        p['number_of_vehicles'] = cur_vehicles['n']
        env = CarlaGymEnv(p, use_route=args.use_route)
        if args.lateral_control:
            from carla_rl.wrappers.lateral_control import HybridSteerWrapper
            env = HybridSteerWrapper(env, throttle_ema=args.throttle_ema,
                                     safety_brake=args.safety_brake,
                                     signed_pedal=args.signed_pedal)  # steer = pure-pursuit
        elif smooth:
            env = SmoothActionWrapper(env, max_steer_delta=args.smooth_steer,
                                      steer_ema=args.steer_ema)
        return ShapedRewardWrapper(env, reward_config)

    nullrhi = not args.render
    ensure_server(port=params['port'], town=params['town'], nullrhi=nullrhi)
    env = env_factory()

    step = start_step
    episode = 0
    restarts = 0
    t_start = time.time()
    metrics = {}

    while step < args.total_steps:
        try:
            # 車流課程:跨步數門檻時切換車流數。在「執行中」的伺服器上重生大量 NPC 會
            # 觸發 SkeletalMesh abort 把進程打死(0 restarts);因此改為「重啟伺服器」
            # (已知可靠的全新開機路徑)再以新車流數重建 env(下一集生效)。
            if curriculum is not None:
                want = curriculum_vehicles(curriculum, step)
                if want != cur_vehicles['n']:
                    print(f'[curriculum] step {step}: vehicles {cur_vehicles["n"]} -> {want} '
                          f'(restarting server)', flush=True)
                    cur_vehicles['n'] = want
                    try:
                        env.close()
                    except Exception:
                        pass
                    restart_server(port=params['port'], town=params['town'], nullrhi=nullrhi)
                    env = env_factory()
            obs = env.reset()
            norm_obs = agent.normalize(obs)
            ep_return = 0.0
            ep_env_return = 0.0
            ep_steps = 0
            done = False
            # 1D 監控累積器
            ep_speed_sum = 0.0; ep_collisions = 0; ep_red = 0; ep_min_ttc = 1.0
            ep_raw_pedal_sum = 0.0; ep_applied_pedal_sum = 0.0; ep_shield = 0
            ep_saw_red = False   # 本集是否「遇到過紅燈」(前瞻 red 旗標)→ best-ckpt 只採計遇紅集

            while not done and step < args.total_steps:
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                # 監控:速度/碰撞/闖紅燈/TTC/pedal/護盾介入
                ep_speed_sum += float(info.get('speed', next_obs[3]))
                if info.get('is_collision'):
                    ep_collisions += 1
                if info.get('red_light_violation'):
                    ep_red += 1
                _tl = info.get('tl_features')
                if _tl is not None and len(_tl) >= 2 and float(_tl[1]) >= 0.5:
                    ep_saw_red = True
                ep_min_ttc = min(ep_min_ttc, float(info.get('min_ttc', 1.0)))
                raw_pedal = (float(action[0]) if args.signed_pedal
                             else float(action[0]) - float(action[2]))
                applied_pedal = float(info.get('applied_pedal', raw_pedal))
                ep_raw_pedal_sum += raw_pedal
                ep_applied_pedal_sum += applied_pedal
                if abs(applied_pedal - raw_pedal) > 0.05:   # 安全層改了動作 = 護盾介入
                    ep_shield += 1

                norm_next = agent.normalize(next_obs)
                # timeout endings are not true terminals for bootstrapping
                is_terminal = done and ep_steps + 1 < params['max_time_episode']
                # store the ACTUALLY EXECUTED action (e.g. pure-pursuit steer in
                # the hybrid stack, rate-limited steer in the smooth stack), so
                # the critic learns Q of what really drove the car
                stored = info.get('applied_action', action)
                buffer.add(norm_obs, stored, reward * args.reward_scale, norm_next,
                           float(is_terminal))
                norm_obs = norm_next
                obs = next_obs

                for _ in range(args.updates_per_step):
                    metrics = agent.update(
                        buffer, args.batch_size,
                        update_actor=step >= critic_warmup,
                    )

                ep_return += reward
                ep_env_return += info.get('reward_env', reward)
                ep_steps += 1
                step += 1

                if args.checkpoint_every and step % args.checkpoint_every == 0:
                    meta = make_meta(step)
                    agent.save(latest_path, meta)
                    agent.save(out_dir / f'sac_step{step}.pth', meta)
                if args.buffer_save_every and step % args.buffer_save_every == 0:
                    buffer.save(buffer_path)

            episode += 1
            mean_speed = ep_speed_sum / max(ep_steps, 1)
            row = {
                'step': step, 'episode': episode,
                'ep_return': round(ep_return, 2),
                'ep_env_return': round(ep_env_return, 2),
                'ep_steps': ep_steps,
                'critic_loss': round(metrics.get('critic_loss', 0.0), 4),
                'q1_mean': round(metrics.get('q1_mean', 0.0), 3),
                'actor_loss': round(metrics.get('actor_loss', 0.0), 4),
                'alpha': round(metrics.get('alpha', 0.0), 4),
                'entropy': round(metrics.get('entropy', 0.0), 3),
                'restarts': restarts,
                'wall_min': round((time.time() - t_start) / 60, 1),
                'mean_speed': round(mean_speed, 3),
                'collision_count': ep_collisions,
                'red_light_violation_count': ep_red,
                'min_ttc': round(ep_min_ttc, 3),
                'raw_pedal_mean': round(ep_raw_pedal_sum / max(ep_steps, 1), 3),
                'applied_pedal_mean': round(ep_applied_pedal_sum / max(ep_steps, 1), 3),
                'shield_intervention_count': ep_shield,
            }
            logger.writerow(row)
            log_file.flush()
            agent.save(latest_path, make_meta(step))
            # best-checkpoint:只在 actor 已過 warmup、回合夠長、且「本集確實遇到紅燈」時評比
            # ——否則沒遇紅燈的高速集 red=0 會被誤選為 best(審查 #6)。排序「闖紅燈少 > 碰撞少 > 均速高」。
            # 註:這仍非固定 seed 評估,僅為訓練中粗篩;正式 best 請用 --checkpoint-every 存檔後手動定點評估。
            if step >= critic_warmup and ep_steps >= 200 and ep_saw_red:
                key = (ep_red, ep_collisions, -mean_speed)
                if best_key is None or key < best_key:
                    best_key = key
                    agent.save(best_path, make_meta(step))
                    print(f'  [best] ep{episode} saw_red red={ep_red} coll={ep_collisions} '
                          f'speed={mean_speed:.2f} -> saved {best_path.name}')
            print(f"step {step}/{args.total_steps} ep {episode}: "
                  f"return={ep_return:.1f} (env {ep_env_return:.1f}) steps={ep_steps} "
                  f"critic={row['critic_loss']} alpha={row['alpha']}")

        except RuntimeError as exc:
            restarts += 1
            print(f'CARLA error ({exc}); restarting server ({restarts})...')
            try:
                env.close()
            except Exception:
                pass
            restart_server(port=params['port'], town=params['town'], nullrhi=nullrhi)
            env = env_factory()

    meta = make_meta(step)
    agent.save(latest_path, meta)
    agent.save(out_dir / 'sac_final.pth', meta)
    if args.buffer_save_every:
        buffer.save(buffer_path)
    env.close()
    log_file.close()
    print(f'done: {step} steps, {episode} episodes, {restarts} restarts. '
          f'Logs: {out_dir}')


if __name__ == '__main__':
    from carla_rl.utils.bigstack import run_with_big_stack

    run_with_big_stack(main)
