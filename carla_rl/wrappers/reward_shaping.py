"""Additive reward shaping on top of the env's built-in reward.

Wrap a CarlaGymEnv (extended_obs must be on for the red-light term to fire).
Both rewards are logged per step: info['reward_env'] (raw) and
info['reward_shaped'] (what this wrapper returns), so training curves can be
compared against unshaped baselines.

With a cruise-style config (replace_speed_term / pedal_jerk_weight active)
the wrapper also applies cruise_shaping() — the same pure function the replay
prefill relabeler uses — on the RAW post-step obs (obs[3]=speed, obs[7]=front
gap) and the PRE-exclusion action, and logs info['v_des'] /
info['reward_terms'] for metrics and debugging.
"""

import gym
import numpy as np

from carla_rl.configs.reward_config import (
    DEFAULT_REWARD_CONFIG,
    cross_traffic_risk,
    cruise_shaping,
    gap_deficit_penalty,
    survive_shaping,
)


class ShapedRewardWrapper(gym.Wrapper):
    def __init__(self, env, config=None):
        super().__init__(env)
        self.config = config or DEFAULT_REWARD_CONFIG
        self._prev_steer = 0.0
        self._prev_pedal = None  # None = first step of episode (zero jerk)
        self._prev_yaw_rate = 0.0

    def reset(self, **kwargs):
        self._prev_steer = 0.0
        self._prev_pedal = None
        self._prev_yaw_rate = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        cfg = self.config

        shaped = reward

        # 動作維度無關:優先讀「實際執行」的 steer/pedal(hybrid/1D 模式由 wrapper 放進 info),
        # 沒有才退回原始 action(舊三維直接控制)。1D signed-pedal 時 action 只有 1 維,不能讀 [1]/[2]。
        steer = info.get('applied_steer')
        if steer is None:
            steer = float(action[1]) if len(action) > 1 else 0.0
        steer = float(steer)
        shaped += cfg.steer_jerk_weight * abs(steer - self._prev_steer)
        self._prev_steer = steer

        if info.get('red_light_violation'):
            shaped += cfg.red_light_penalty
            # 闖紅燈 = 終止回合(與撞車同級):車子不能「闖過去後繼續賺速度獎勵」,
            # 一旦真的在紅燈下穿越就結束 + 大懲罰 → 讓「停在紅燈前」成為最佳選擇。
            if cfg.red_light_terminal:
                done = True
                info['red_light_terminated'] = True
        # 密集減速懲罰:前方有紅燈(tl[1]=前瞻 red)且「已接近停止線」(tl[4]<=creep_dist_norm)仍移動
        # → 每步按速度扣分。距離閘門避免「40m 外一看到紅燈就停」的超早停;遠處不罰、接近才煞。
        if cfg.red_light_creep_weight != 0.0:
            tl = info.get('tl_features')
            if (tl is not None and len(tl) >= 5 and tl[1] >= 0.5
                    and float(tl[4]) <= cfg.red_light_creep_dist_norm):
                spd = float(info.get('speed', obs[3]))
                if spd > 0.5:
                    shaped += cfg.red_light_creep_weight * spd

        if cfg.progress_weight != 0.0:
            # forward progress this step, in meters
            shaped += cfg.progress_weight * info.get('speed', 0.0) * self.env.params['dt']

        if cfg.survive:
            # 長程存活獎勵:保留 env 速度項(replace_speed_term=False),加 progress 小獎勵。
            # survive 分支需先於下面的 cruise 分支(兩者互斥)。
            delta, terms = survive_shaping(
                float(obs[3]), float(info['front_gap']),
                float(obs[311]) if len(obs) >= 312 else 1.0,
                cfg, float(self.env.params['desired_speed']),
                min_ttc=float(info.get('min_ttc', 1.0)),
            )
            shaped += float(delta)
            # 卡住一次性懲罰:env 的 stuck 終止本身無懲罰,會被「停住farming」鑽。
            # 只在卡住終止的那一步套用;遠輕於撞車(-100),不誘發自殺。
            if cfg.survive_stuck_penalty != 0.0 and info.get('stuck'):
                shaped += cfg.survive_stuck_penalty
                terms['stuck_penalty'] = cfg.survive_stuck_penalty
            # avoid_v1/v2:TTC 接近懲罰(RL 自學避撞的梯度)。min_ttc 由 gym_compat 放進 info。
            if cfg.avoid_ttc_weight != 0.0:
                mt = float(info.get('min_ttc', 1.0))
                ttc_pen = cfg.avoid_ttc_weight * max(0.0, cfg.avoid_ttc_floor_norm - mt)
                shaped += ttc_pen
                terms['ttc_pen'] = ttc_pen
            # avoid_v2:加重撞車終止懲罰(env -100 相對 progress 太便宜)。
            if cfg.avoid_collision_penalty != 0.0 and info.get('is_collision'):
                shaped += cfg.avoid_collision_penalty
                terms['collision_penalty'] = cfg.avoid_collision_penalty
            # avoid_v4:距離缺口懲罰(根因修法)。前車距 info['front_gap'](path-aware)低於
            # avoid_gap_floor 就按缺口線性扣分、隨速度淡出——TTC 型訊號近距低速盲,這補上「距離型」
            # 密集梯度,逼 RL 學會在前車後停住而非潛行頂上去(診斷:94% 碰撞屬此)。
            if cfg.avoid_gap_weight != 0.0:
                gap_pen = float(gap_deficit_penalty(
                    float(info['front_gap']), float(obs[3]), cfg))
                shaped += gap_pen
                terms['gap_pen'] = gap_pen
            # pedal jerk:survive 分支原本沒套用(審查 #3)。用「實際執行」的 applied_pedal 算抖動。
            if cfg.pedal_jerk_weight != 0.0:
                pedal = info.get('applied_pedal')
                if pedal is None:
                    pedal = float(action[0]) - (float(action[2]) if len(action) > 2 else 0.0)
                pedal = float(pedal)
                prev_pedal = pedal if self._prev_pedal is None else self._prev_pedal
                pj = cfg.pedal_jerk_weight * abs(pedal - prev_pedal)
                shaped += pj
                terms['pedal_jerk'] = pj
                self._prev_pedal = pedal
            info['reward_terms'] = {k: float(v) for k, v in terms.items()}
        elif cfg.replace_speed_term or cfg.speed_weight != 0.0 or cfg.pedal_jerk_weight != 0.0:
            # 實際執行的 pedal(applied_pedal=throttle-brake);1D 模式 action 只有 1 維,
            # 不能讀 action[0]/[2]。沒有 info 才退回原始三維動作。
            pedal = info.get('applied_pedal')
            if pedal is None:
                pedal = float(action[0]) - (float(action[2]) if len(action) > 2 else 0.0)
            pedal = float(pedal)
            prev_pedal = pedal if self._prev_pedal is None else self._prev_pedal
            # obs[288] = relative yaw of the ~10 m-lookahead waypoint.
            # gap/rel_speed come from path-aware lead detection (info), NOT
            # obs[7]/obs[8] — the env's straight-box lead is phantom-prone in
            # turns and froze the car mid-corner.
            delta, terms = cruise_shaping(
                float(obs[3]), float(info['front_gap']), pedal, prev_pedal, cfg,
                float(self.env.params['desired_speed']),
                curve=abs(float(obs[288])),
                rel_speed=float(info.get('lead_rel_speed', 0.0)),
                yaw_rate=float(obs[4]), prev_yaw_rate=self._prev_yaw_rate,
                junction_dist=float(obs[311]) if len(obs) >= 312 else 1.0,
            )
            shaped += float(delta)
            self._prev_pedal = pedal
            self._prev_yaw_rate = float(obs[4])
            info['v_des'] = float(terms['v_des'])
            info['reward_terms'] = {k: float(val) for k, val in terms.items()}

        # junction cross-traffic avoidance — needs the extended obs (obs[311] =
        # dist-to-junction, obs[251:271] = nearby vehicles). Online only.
        if cfg.junction_ttc_weight != 0.0 and len(obs) >= 312:
            jc = cross_traffic_risk(obs[251:271], float(obs[3]), float(obs[311]), cfg)
            shaped += jc
            info['junction_cross'] = jc
            if 'reward_terms' in info:
                info['reward_terms']['junction_cross'] = jc

        info['reward_env'] = float(reward)
        info['reward_shaped'] = float(shaped)
        return obs, float(shaped), done, info
