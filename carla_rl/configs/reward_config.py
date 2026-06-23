"""Tunable shaping terms applied ON TOP of the env's built-in reward.

The base reward (speed tracking, lane deviation, lateral-accel smoothness,
stationary penalty, collision/off-road -100) lives in
EasyCarla-RL/easycarla/envs/carla_env.py and is left untouched by default.

The cruise preset (Phase 4) REPLACES the env's speed term: the env tent
(+v up to desired_speed, then -(v - desired_speed)) has a reward cliff at
desired_speed that fights constant-speed cruising. cruise_shaping() subtracts
the tent exactly and adds a smooth tracking bump centered on a time-headway-
aware target speed. All cruise fields are inert at their defaults, so
DEFAULT_REWARD_CONFIG keeps stage-1 behavior bit-identical (its supervisor
may relaunch train_sac with this code at any time).

Cautions from the project plan:
  - don't raise speed/progress rewards blindly (unsafe driving)
  - steer-jerk penalty is the first knob against steering oscillation
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RewardConfig:
    # penalty per unit of |steer_t - steer_{t-1}| (counters oscillation)
    steer_jerk_weight: float = -0.5
    # one-time penalty for running a red light (event from RedLightMonitor)
    red_light_penalty: float = -50.0
    # 密集紅燈懲罰:在紅燈觸發區內「每步」按速度扣分(weight*speed,weight 負)。比 once-per-
    # encounter 的 red_light_penalty 密集得多,提供「看到紅燈就煞停」的連續梯度(燈轉綠即歸零→放行)。
    red_light_creep_weight: float = 0.0
    # 闖紅燈是否終止回合(與撞車同級)。True 時實際紅燈穿越會結束 episode + red_light_penalty,
    # 堵住「闖過去後繼續賺速度獎勵」的漏洞 → 讓停紅燈成為最佳策略。需搭配前瞻燈號觀測才學得起來。
    red_light_terminal: bool = False
    # 密集減速懲罰的「距離閘門」(正規化 dist,1.0=50m):只有前方紅燈距離 tl[4] <= 此值才罰移動。
    # 預設 1.0 = 不限距離(會在 40m 前瞻一看到紅燈就罰 → 超早停);設 ~0.24(=12m)才會「接近停止線
    # 才煞」、遠處照常開。
    red_light_creep_dist_norm: float = 1.0
    # reward per meter of forward progress; 0 by default because the env's
    # speed term already rewards motion — enable deliberately, not blindly
    progress_weight: float = 0.0

    # --- cruise extension (Phase 4). All inert at defaults; penalty weights
    # are negative, matching the steer_jerk_weight convention. Invariant:
    # headway_offset + headway_time * target_speed <= 20 m, because obs[7]
    # only sees a lead within 20 m — this keeps v_des and the tailgate term
    # continuous at the sensor cutoff.
    replace_speed_term: bool = False  # subtract env tent, add the terms below
    target_speed: float = 6.5         # v_target (m/s), cruise speed when free
    speed_weight: float = 0.0         # tracking-bump height (cruise: +4.0)
    speed_sigma: float = 0.6          # bump width (m/s); tighter = more constant-speed
    headway_time: float = 1.6         # T (s): required gap grows with speed
    headway_offset: float = 5.0       # d0 (m); obs[7] is center-to-center, contact ~4.7
    low_speed_floor: float = 5.5      # required free-flow speed (m/s)
    low_speed_weight: float = 0.0     # per m/s below floor when unjustified (cruise: -2.0)
    overspeed_margin: float = 1.0     # m/s above target before penalty starts
    overspeed_weight: float = 0.0     # per m/s above target+margin (cruise: -2.0)
    tailgate_weight: float = 0.0      # per unit gap-deficit fraction (cruise: -4.0)
    pedal_jerk_weight: float = 0.0    # per |d(throttle - brake)| (cruise: -0.3)
    # curve-aware target speed: v_des shrinks by 1/(1 + gain * |waypoint yaw|)
    # using the ~10 m-lookahead waypoint's relative heading (raw obs[288]).
    # 0.0 = inert. Why: demanding 6.5 m/s through corners under a tight steer
    # rate cap is physically unturnable (stage-4 off-road 0.4); slowing for
    # curves is correct driving per the project plan's "proper speed".
    curve_gain: float = 0.0
    # scale the tracking bump by (v_des/target)^power: slow zones must pay
    # LESS than full-speed cruising or the policy farms low-v_des areas
    # (stage-6 drove 3.5 m/s everywhere: bump +8 at a curve apex beat +1.5 on
    # straights). power=2 makes a 3.5 m/s apex worth +2.3-3.5 < straight +1.5.
    bump_vdes_power: float = 0.0
    # weave penalty: fires when the body yaw-rate (obs[4], DEG/s — verified on
    # the dataset: |p90| ~19) REVERSES SIGN with both samples above a noise
    # floor. This is exactly the eval's weave metric turned into a training
    # signal; legitimate turns hold one sign and are never charged.
    yaw_flip_weight: float = 0.0
    yaw_flip_floor: float = 1.5  # deg/s, matches the eval metric
    # time-to-collision penalty: when closing on a lead (obs[8] = ego speed -
    # lead speed > 0), penalize weight * max(0, ttc_floor - gap/closing_speed).
    # Distance-based tailgating misses fast approaches; TTC catches them.
    ttc_floor: float = 2.0
    ttc_weight: float = 0.0
    # --- junction cross-traffic avoidance (V6). The forward TTC/headway terms
    # above only watch the lead car; side/crossing traffic at junctions is
    # invisible to them. This penalizes a predicted collision with any nearby
    # vehicle (from `nearby_vehicles`, omnidirectional within 50 m) whose path
    # closes on the ego — but ONLY when within junction_dist_norm_thresh of a
    # junction, so an adjacent-lane car on a straight never triggers a slowdown.
    junction_ttc_weight: float = 0.0     # per second below cross_ttc_floor (e.g. -4.0)
    junction_dist_norm_thresh: float = 0.4  # obs[311] gate: 0.4 * 50 m = within 20 m
    cross_ttc_floor: float = 3.0         # only react inside this time-to-collision (s)
    cross_miss_radius: float = 3.0       # only if predicted closest approach < this (m)
    # --- junction-aware target speed (V8): the both-solution. Collisions are
    # 100% at junctions, so drive FAST on the open road but slow APPROACHING a
    # junction (obs[311] = dist-to-junction, 1.0 = none in 50 m). v_des is capped
    # toward junction_target_speed as obs[311] falls below junction_slow_norm,
    # so free-flow speed stays high (the speed goal) while junctions are entered
    # slow (collisions down + the safety brake is effective at low speed).
    junction_slow_norm: float = 0.0      # obs[311] below which to start slowing; 0 = off
    junction_target_speed: float = 5.0   # m/s cap right at the junction

    # --- 長程存活獎勵(survive_v1)。survive=True 時 ShapedRewardWrapper 改走
    # survive_shaping():極簡「安全前進」獎勵,不堆密集塑形(CaRL 教訓:密集
    # shaped reward 會被鑽且不 scale)。目標是「長時間穩定行駛、不出問題、又不太慢」。
    survive: bool = False
    survive_v_target: float = 8.0       # 前進獎勵封頂(m/s):min(v,target)/target ∈ [0,1]
    survive_progress_weight: float = 1.0  # progress 權重(avoid_v5 拉速度:開闊路無安全懲罰,加大此值直接拉高自由車速)
    survive_v_floor: float = 5.0        # 無障礙時低於此速 = 爬行(僅在 crawl_penalty!=0 時生效)
    survive_crawl_penalty: float = 0.0   # 預設關閉:它讓「停住」變負 → 反而誘發早撞自殺
    survive_junction_clear: float = 0.4  # obs[311] >= 此值才算「非接近路口」
    # --- RL-核心避撞(avoid_v1):TTC 接近懲罰,讓 RL 在快撞前就收油/煞車(自學避撞)。
    # min_ttc_norm = 三扇區最小預測 TTC / RISK_HORIZON ∈ [0,1];低於門檻線性扣分。
    # 由 ShapedRewardWrapper 讀 info['min_ttc'](gym_compat 放入)套用,不在純函式內。
    avoid_ttc_weight: float = 0.0        # 每單位(門檻-ttc_norm)的懲罰(負);例 -6.0
    avoid_ttc_floor_norm: float = 0.4    # 門檻(=2s/5s);ttc_norm < 0.4 才開始罰
    # Extra speed pull only in clear, low-risk scenes. This stays separate
    # from broad progress so it never pays while following a lead, when the
    # risk features say TTC is low, or near junctions.
    avoid_free_speed_weight: float = 0.0
    avoid_free_speed_floor: float = 6.0
    avoid_free_speed_target: float = 9.0
    avoid_free_speed_ttc_floor_norm: float = 0.85
    avoid_free_speed_junction_clear: float = 0.6
    # 撞車一次性終止懲罰(在 wrapper 讀 info['is_collision'] 套用)。env 自帶 -100 相對
    # 於 3000 步 ~6000 的 progress 太便宜 → RL 學會「開快、偶爾撞無所謂」。加重使撞車變貴。
    avoid_collision_penalty: float = 0.0  # 例 -500
    # 距離缺口懲罰(avoid_v4 的根因修法)。診斷:94% 碰撞是「低速潛行頂前車」,而 TTC 型訊號
    # 在近距低速時(TTC=距離/逼近速度)會把「1m 外 0.5m/s 逼近」算成 ~2s 看似安全 → RL 從沒拿到
    # 「近距要停住」的梯度。改用「距離」直接罰:前車距(path-aware,info['front_gap'])低於
    # avoid_gap_floor 就按缺口線性扣分,並以 min(1,v/2) 隨速度淡出(車已被前車卡住停住時不罰,
    # 避免教出 escape-or-suicide)。這在「逼近前車」的失效狀態提供密集、可靠的減速/停住梯度。
    avoid_gap_weight: float = 0.0        # 每公尺缺口的懲罰(負);例 -8.0
    avoid_gap_floor: float = 7.0         # 低於此前車距(m)才開始罰(= safety_brake 的 7m)
    # 每步存活加分:預設 0。教訓(2026-06-17):設為正值會被「停住farming」
    # (alive > crawl 時,停到 stuck 終止仍淨正,策略學會不開車)。改為保留 env 內建
    # 速度項(replace_speed_term=False)當前進獎勵——它本身就是「開車為正、停住≈0、
    # 撞車-100」的正確結構,不需 alive。
    survive_alive_bonus: float = 0.0
    # 卡住終止的一次性懲罰(在 wrapper 套用,讀 info['stuck'])。env 的 stuck 終止本身
    # 沒有懲罰,會被「停住farming」鑽;此懲罰讓「卡住」明確劣於「繼續開」,但遠輕於
    # 撞車(-100),故不會誘發自殺。
    survive_stuck_penalty: float = 0.0

    def to_dict(self):
        return asdict(self)


def speed_tent(v, desired_speed):
    """Exactly the env's built-in speed term (carla_env.py:710-715)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= desired_speed, v, -(v - desired_speed))


def cruise_desired_speed(gap, cfg, curve=0.0, junction_dist=1.0):
    """Time-headway + curvature- + junction-aware target speed. gap is raw
    obs[7]: 0.0 means no lead (real gaps are >= ~2.6 m, never 0). curve is
    |relative yaw| (rad) of the ~10 m-lookahead waypoint. junction_dist is
    obs[311] (1.0 = no junction within 50 m)."""
    gap = np.asarray(gap, dtype=np.float64)
    # the env's waypoint relative yaw is NOT wrapped (raw values cluster at 0
    # AND +-2pi for "straight ahead"); wrap to [-pi, pi] before measuring
    curve = np.asarray(curve, dtype=np.float64)
    curve = np.arctan2(np.sin(curve), np.cos(curve))
    target = cfg.target_speed / (1.0 + cfg.curve_gain * np.abs(curve))
    # junction-aware cap: slow APPROACHING a junction. frac=0 at the junction,
    # 1 at/beyond junction_slow_norm -> cap blends junction_target_speed..target.
    if cfg.junction_slow_norm > 0.0:
        jd = np.asarray(junction_dist, dtype=np.float64)
        frac = np.clip(jd / cfg.junction_slow_norm, 0.0, 1.0)
        jcap = cfg.junction_target_speed + frac * (cfg.target_speed - cfg.junction_target_speed)
        target = np.minimum(target, jcap)
    follow = np.minimum(
        target,
        np.maximum(0.0, (gap - cfg.headway_offset) / cfg.headway_time),
    )
    return np.where(gap > 0.0, follow, target)


def cruise_shaping(v, gap, pedal, prev_pedal, cfg, env_desired_speed, curve=0.0,
                   rel_speed=0.0, yaw_rate=0.0, prev_yaw_rate=0.0, junction_dist=1.0):
    """Reward delta added on top of the env reward.

    Inputs are RAW obs values (wrappers and the dataset relabeler never see
    normalized obs) taken from the POST-step obs — the env computes its
    reward from that same obs, and dataset rewards[i] came from
    next_observations[i]. pedal/prev_pedal are PRE-exclusion throttle-brake.

    Vectorized: scalars in -> 0-d arrays out (callers cast with float()),
    arrays in -> arrays out (offline relabeling). Returns (delta, terms).
    """
    v = np.asarray(v, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    v_des = cruise_desired_speed(gap, cfg, curve, junction_dist)
    delta = np.zeros_like(v)
    terms = {'v_des': v_des}

    if cfg.replace_speed_term or cfg.speed_weight != 0.0:
        # replace_speed_term subtracts the env tent (stage-3..4 designs);
        # cruise5 keeps the tent (linear "faster pays more" floor, positive
        # everywhere) and only ADDS the tracking bump on top.
        terms['tent_removed'] = (
            -speed_tent(v, env_desired_speed) if cfg.replace_speed_term
            else np.zeros_like(v)
        )
        bump_scale = 1.0
        if cfg.bump_vdes_power != 0.0:
            bump_scale = (v_des / cfg.target_speed) ** cfg.bump_vdes_power
        terms['speed_track'] = cfg.speed_weight * bump_scale * np.exp(
            -((v - v_des) ** 2) / (2.0 * cfg.speed_sigma ** 2)
        )
        # slow is only penalized when unjustified, i.e. the target itself is
        # at/above the floor (a close lead OR a sharp curve lowers v_des below
        # the floor and justifies slowing; identical to the old condition for
        # curve_gain=0 presets since a no-lead v_des is then always the target)
        unjustified = v_des >= cfg.low_speed_floor
        terms['low_speed'] = np.where(
            unjustified,
            cfg.low_speed_weight * np.maximum(0.0, cfg.low_speed_floor - v),
            0.0,
        )
        terms['overspeed'] = cfg.overspeed_weight * np.maximum(
            0.0, v - (cfg.target_speed + cfg.overspeed_margin)
        )
        d_safe = cfg.headway_offset + cfg.headway_time * v
        # fade out below 2 m/s: a stopped car in a jam is not "tailgating",
        # and punishing forced stops teaches escape-or-suicide behavior
        tailgate_speed_scale = np.minimum(1.0, v / 2.0)
        terms['tailgate'] = np.where(
            gap > 0.0,
            cfg.tailgate_weight * tailgate_speed_scale
            * np.maximum(0.0, (d_safe - gap) / d_safe),
            0.0,
        )
        delta = delta + terms['tent_removed'] + terms['speed_track'] \
            + terms['low_speed'] + terms['overspeed'] + terms['tailgate']

    if cfg.ttc_weight != 0.0:
        rel = np.asarray(rel_speed, dtype=np.float64)
        closing = (gap > 0.0) & (rel > 0.1)
        ttc = np.where(closing, gap / np.maximum(rel, 0.1), np.inf)
        terms['ttc'] = cfg.ttc_weight * np.maximum(0.0, cfg.ttc_floor - ttc)
        delta = delta + terms['ttc']

    if cfg.yaw_flip_weight != 0.0:
        yr = np.asarray(yaw_rate, dtype=np.float64)
        pyr = np.asarray(prev_yaw_rate, dtype=np.float64)
        flip = (yr * pyr < 0) & (np.abs(yr) > cfg.yaw_flip_floor) \
            & (np.abs(pyr) > cfg.yaw_flip_floor)
        terms['yaw_flip'] = np.where(flip, cfg.yaw_flip_weight, 0.0)
        delta = delta + terms['yaw_flip']

    if cfg.pedal_jerk_weight != 0.0:
        terms['pedal_jerk'] = cfg.pedal_jerk_weight * np.abs(
            np.asarray(pedal, dtype=np.float64)
            - np.asarray(prev_pedal, dtype=np.float64)
        )
        delta = delta + terms['pedal_jerk']

    return delta, terms


def survive_shaping(v, gap, junction_dist, cfg, env_desired_speed, min_ttc=1.0):
    """長程存活獎勵(survive_v1)。回傳 (delta, terms),加在 env 獎勵之上。

    參數均為 RAW 觀測值:
      v             = obs[3]   當前車速 (m/s)
      gap           = path-aware 前車距 (m);<=0 表「前方無車」
      junction_dist = obs[311] 正規化路口距離 (1.0 = 50m 內無路口)
    設計(CaRL 教訓:只說「要做什麼」=安全前進,不堆密集塑形):
      - progress:min(v,v_target)/v_target ∈ [0,1],越快(到 8)分越高 → 不會太慢,
        但 8 封頂 → 不鼓勵超速魯莽。
      - 防爬行:無障礙(無前車且非接近路口)卻 v<v_floor → 扣分(這就是「速度不能太慢」)。
        有前車或接近路口時的慢是「合理的」,不罰。
      - replace_speed_term=True 時先扣掉 env 內建速度 tent,只留此處 progress,避免雙重計分。
      - 撞/出界/卡住的懲罰來自 env「終止」(失去後續所有 progress),此處不另加密集懲罰。
    向量化:純量進 -> 0-d array 出(呼叫端用 float() 取值),陣列進 -> 陣列出。
    """
    v = np.asarray(v, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    jd = np.asarray(junction_dist, dtype=np.float64)
    mt = np.asarray(min_ttc, dtype=np.float64)
    # 扣掉 env 內建速度 tent(避免與下面 progress 雙重計分)
    tent_removed = (-speed_tent(v, env_desired_speed) if cfg.replace_speed_term
                    else np.zeros_like(v))
    progress = cfg.survive_progress_weight * np.minimum(v, cfg.survive_v_target) / cfg.survive_v_target
    # 無障礙 = 前方無車 且 非接近路口 → 此時的慢沒有正當理由
    unobstructed = (gap <= 0.0) & (jd >= cfg.survive_junction_clear)
    crawl = np.where(unobstructed & (v < cfg.survive_v_floor),
                     cfg.survive_crawl_penalty, 0.0)
    free_speed = np.zeros_like(v)
    if cfg.avoid_free_speed_weight != 0.0:
        clear = (
            (gap <= 0.0)
            & (jd >= cfg.avoid_free_speed_junction_clear)
            & (mt >= cfg.avoid_free_speed_ttc_floor_norm)
        )
        denom = max(cfg.avoid_free_speed_target - cfg.avoid_free_speed_floor, 1e-6)
        ramp = np.clip((v - cfg.avoid_free_speed_floor) / denom, 0.0, 1.0)
        free_speed = np.where(clear, cfg.avoid_free_speed_weight * ramp, 0.0)
    # 存活加分:保證每步淨獎勵為正(存活 >> 早撞自殺),維持正最優不變式
    alive = cfg.survive_alive_bonus
    delta = tent_removed + alive + progress + crawl + free_speed
    terms = {'tent_removed': tent_removed, 'alive': np.full_like(v, alive),
             'progress': progress, 'crawl': crawl, 'free_speed': free_speed}
    return delta, terms


def gap_deficit_penalty(gap, v, cfg):
    """距離缺口懲罰(avoid_v4):前車距 gap (m;<=0 表前方無 on-path 車)低於 avoid_gap_floor 時,
    按缺口 (floor-gap) 線性扣分,並乘上 min(1, v/2) 隨速度淡出。

    為何淡出:車被前車卡住「已停住」(v≈0)時,gap 小是「正確地停著」而非錯誤 → 不該罰,否則
    教出「逃離或撞車」。移動中(尤其潛行 0.5–2 m/s 進前車)才是要壓制的失效 → 此時 scale 漸增。
    最優點(停在 floor 後面 hold 住,gap≈floor → 缺口 0 → 0 罰)可達且無罰,故不誘發自殺。
    向量化:純量進→0-d array 出,陣列進→陣列出(離線 relabel 用)。回傳懲罰(<=0)。"""
    g = np.asarray(gap, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    scale = np.minimum(1.0, np.maximum(0.0, vv) / 2.0)
    pen = cfg.avoid_gap_weight * scale * np.maximum(0.0, cfg.avoid_gap_floor - g)
    return np.where(g > 0.0, pen, 0.0)


def cross_traffic_risk(nearby_flat, ego_speed, dist_junction_norm, cfg):
    """Penalty (<= 0) for cross/side traffic on a collision course near a
    junction. Scalar (online wrapper only); the offline dataset has no junction
    feature, so the relabeler never calls this.

    nearby_flat: the 20-vector `nearby_vehicles` (5 x [local_x, local_y,
    rel_yaw, speed]) in the ego frame (obs[251:271]). dist_junction_norm is
    obs[311] (1.0 = no junction within 50 m). Returns
    junction_ttc_weight * max(0, cross_ttc_floor - min_ttc) where min_ttc is the
    soonest predicted collision (closest approach < cross_miss_radius within
    cross_ttc_floor seconds); 0 when not near a junction or nothing converges.
    """
    if cfg.junction_ttc_weight == 0.0:
        return 0.0
    # gate on junction proximity so straights never slow down for adjacent lanes
    if dist_junction_norm >= cfg.junction_dist_norm_thresh:
        return 0.0
    nb = np.asarray(nearby_flat, dtype=np.float64).reshape(-1, 4)
    min_ttc = np.inf
    for x, y, ryaw, spd in nb:
        if x == 0.0 and y == 0.0 and spd == 0.0:
            continue  # zero-padded (fewer than max_nearby_vehicles detected)
        # relative velocity (other - ego) in the ego frame; ego moves along +x
        rvx = spd * np.cos(ryaw) - ego_speed
        rvy = spd * np.sin(ryaw)
        vv = rvx * rvx + rvy * rvy
        if vv < 1e-6:
            continue  # no relative motion -> never closes
        # time of closest approach of the two point trajectories
        t = -(x * rvx + y * rvy) / vv
        if t <= 0.0 or t > cfg.cross_ttc_floor:
            continue  # diverging, or collision too far in the future
        miss = np.hypot(x + rvx * t, y + rvy * t)
        if miss <= cfg.cross_miss_radius:
            min_ttc = min(min_ttc, t)
    if not np.isfinite(min_ttc):
        return 0.0
    return cfg.junction_ttc_weight * max(0.0, cfg.cross_ttc_floor - min_ttc)


DEFAULT_REWARD_CONFIG = RewardConfig()

# Phase 4 cruise objective: constant 6.5 m/s free-flow cruise (>5.5 required),
# time-headway following, smooth control. steer_jerk raised -0.5 -> -0.8.
#
# INVARIANT (learned from the stage-2 collapse): perfect cruising must earn a
# clearly POSITIVE per-step reward. The tent removal costs -target_speed at
# v=target, so speed_weight must exceed target_speed with margin — at 4.0 the
# optimum was -2.5/step and the policy learned to end episodes early
# (collision -100 beats living at negative rate); episode length regressed
# 315 -> 221 steps in the last quarter of training.
CRUISE_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=-0.8,
    replace_speed_term=True,
    speed_weight=8.0,   # perfect 6.5 m/s cruise: -6.5 + 8.0 = +1.5/step
    low_speed_weight=-2.0,
    overspeed_weight=-2.0,
    tailgate_weight=-4.0,
    pedal_jerk_weight=-0.3,
)

# Stage-4 refinement of cruise (weight-magnitude lessons from the stage-3
# eval): at steer_jerk -0.8 a rate-cap zigzag cost 0.08/step vs the +8 speed
# bump — 1% of signal, so the policy zigzagged at the cap (straight-segment
# flip rate 0.41); tailgate -4 left 55% headway violations / 70% collisions.
CRUISE2_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=-4.0,
    replace_speed_term=True,
    speed_weight=8.0,
    low_speed_weight=-2.0,
    overspeed_weight=-2.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
)

# Stage-5: cruise2 + curvature-aware target speed. Resolves the stage-4
# conflict (tight steer cap x full speed through corners = off-road 0.4): at
# |waypoint yaw| 0.3 rad the target drops 6.5 -> 4.1 m/s, so the 0.07 rate cap
# is sufficient to corner.
CRUISE3_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=-4.0,
    replace_speed_term=True,
    speed_weight=8.0,
    low_speed_weight=-2.0,
    overspeed_weight=-2.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
)

# Stage-7: target success >= 0.9. vs cruise3: bump scaled by (v_des/target)^2
# (kills stage-6's slow-farming exploit — full-speed straights pay best again)
# and a TTC penalty (distance-only tailgating missed fast approaches; 60% of
# stage-6 remaining failures were collisions while closing).
CRUISE4_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=-4.0,
    replace_speed_term=True,
    speed_weight=8.0,
    low_speed_weight=-2.0,
    overspeed_weight=-2.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
    bump_vdes_power=2.0,
    ttc_weight=-3.0,
)

# cruise5 — the corrected speed-reward GEOMETRY. cruise4's (v_des/target)^2
# bump scaling re-broke the positive-optimum invariant: correct reduced-speed
# driving (curves, car-following) netted negative and 7b collapsed in dense
# traffic. cruise5 KEEPS the env tent (linear positive floor: alive & moving
# always pays, faster pays more) and adds a LINEARLY v_des-scaled bump:
#   straight 6.5: ~+10.5/step   curve apex 3.5: ~+5.7   stopped in jam: ~0
# Positive everywhere driving is correct, monotone in v_des (no slow-farming),
# neutral at forced stop (no parking farm). low_speed term dropped — the tent
# already penalizes slow linearly.
CRUISE5_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=-4.0,
    replace_speed_term=False,
    speed_weight=4.0,
    bump_vdes_power=1.0,
    low_speed_weight=0.0,
    overspeed_weight=-2.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
    ttc_weight=-3.0,
)

# V2.2: cruise_v2 + direct weave penalty (-3 per yaw-rate sign flip — the
# eval's weave metric as a training signal; v2/v2.1 proved steering-side
# penalties and filters leave low-frequency commanded weaving intact).
CRUISE_V2B_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=-4.0,
    replace_speed_term=False,
    speed_weight=4.0,
    bump_vdes_power=1.0,
    low_speed_weight=0.0,
    overspeed_weight=-2.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
    ttc_weight=-3.0,
    yaw_flip_weight=-3.0,
)

# V2 rebuild: THE preset (see carla_rl/PLAN_V2.md). Identical to cruise5 —
# the geometry that passed every economics assertion and produced the
# project's healthiest curves (7a fresh run) — kept under its own name so the
# v2 run family never depends on legacy preset names. cruise1-4 are retained
# only for archived-run reproducibility; do not train on them.
CRUISE_V2_REWARD_CONFIG = CRUISE5_REWARD_CONFIG

# V4 (hierarchical): RL controls ONLY throttle/brake; steering is pure-pursuit.
# Free-flow target 11 m/s (user, safety-first: slow for leads). steer_jerk
# dropped (no RL steering to penalize). Use with env desired_speed=12 so the
# env speed tent rewards up to 12 (peak just above target, no penalty at 11)
# and overspeed only bites past 12. curve_gain=2 slows for corners (11 m/s
# cornering is unsafe). Headway: v_des = min(11, (gap-5)/1.6); a lead anywhere
# in the 20 m sensor range pulls the target down — safety-first by design.
CRUISE_FAST_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    target_speed=11.0,
    speed_weight=4.0,
    bump_vdes_power=1.0,
    low_speed_weight=0.0,
    overspeed_weight=-2.0,
    overspeed_margin=1.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
    ttc_weight=-3.0,
)

# V6 (safety-first, on global routes): cruise_fast's two open problems were
# (a) free speed stuck at ~6 and (b) junction cross-traffic collisions. Fixes:
#   - target_speed 11 -> 8 with speed_sigma 0.6 -> 1.0: the Gaussian bump at
#     target 11 was ~0 at 6 m/s (no gradient), so the policy coasted on the weak
#     linear tent. A REACHABLE target (8) with a wider bump actively pulls free
#     speed up to ~8. overspeed bites past 9 (modest, not 11 — user's choice).
#   - junction cross-traffic TTC penalty (the side-collision fix; gated on
#     junction proximity so straights are unaffected).
# Headway invariant holds: 5 + 1.6*8 = 17.8 <= 20 m sensor cutoff. Use with env
# desired_speed=10 (tent peaks at 10) and lateral_control + use_route.
CRUISE_SAFE_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,        # steering is pure-pursuit (no RL steer)
    replace_speed_term=False,
    target_speed=8.0,
    speed_weight=4.0,
    speed_sigma=1.0,              # wider than 0.6 so the bump pulls 6 -> 8
    bump_vdes_power=1.0,
    low_speed_weight=0.0,
    overspeed_weight=-2.0,
    overspeed_margin=1.0,         # penalty above 9 m/s
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
    ttc_weight=-3.0,
    junction_ttc_weight=-4.0,
    junction_dist_norm_thresh=0.4,
    cross_ttc_floor=3.0,
    cross_miss_radius=3.0,
)

# V7: trained WITH the longitudinal safety brake active (HybridSteerWrapper
# safety_brake), so the policy can drive a bit faster knowing the override
# backstops rear-ends. vs cruise_safe: target_speed 8 -> 9 (push speed up, the
# user's goal) and the junction cross-traffic penalty DROPPED (the diagnosis
# proved collisions are rear-ends, not side traffic — that term solved nothing).
# Everything else (headway/tailgate/ttc/curve/pedal) kept. Use with env
# desired_speed=11 (tent peaks above the 9 target; overspeed bites past 10).
CRUISE_V7_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    target_speed=9.0,
    speed_weight=4.0,
    speed_sigma=1.0,
    bump_vdes_power=1.0,
    low_speed_weight=0.0,
    overspeed_weight=-2.0,
    overspeed_margin=1.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
    ttc_weight=-3.0,
    junction_ttc_weight=0.0,   # dropped — diagnosis: no side collisions exist
)

# V8: the both-solution — fast on the open road, slow into junctions. vs
# cruise_v7: target_speed 9 -> 10 (push open-road speed up, the user's goal) AND
# junction-aware slowing on (cap 5 m/s within ~15 m of a junction, where 100% of
# collisions happen). Trained WITHOUT the safety brake (V7 proved training with
# it makes the policy reckless); the brake is an inference-time backup only.
# Use with env desired_speed=12 (tent peaks above the 10 target; overspeed past 11).
CRUISE_V8_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    target_speed=10.0,
    speed_weight=4.0,
    speed_sigma=1.0,
    bump_vdes_power=1.0,
    low_speed_weight=0.0,
    overspeed_weight=-2.0,
    overspeed_margin=1.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,
    ttc_weight=-3.0,
    junction_ttc_weight=0.0,
    junction_slow_norm=0.3,       # start slowing within ~15 m of a junction
    junction_target_speed=5.0,    # cap to 5 m/s at the junction
)

# V9 reward == V8 geometry (open-road target 10 + junction-slow, cross-traffic
# off). V9's novelty is the predictive PERCEPTION + PLANNER, not the reward, so
# the preset is aliased for clean run labeling.
CRUISE_V9_REWARD_CONFIG = CRUISE_V8_REWARD_CONFIG

# V12 (speed push on the STABLE critic). sac_v12 cruised at 6.82 vs its target
# 10 — with a non-diverging critic the policy now correctly prices collision
# risk and trades speed DOWN; the +4 bump (sigma 1.0) wasn't a strong enough
# pull. This DOUBLES the speed bump (4 -> 8) and WIDENS it (sigma 1.0 -> 2.0) so
# there is a strong gradient across 6 -> 10 m/s, while leaving every safety /
# curve / junction term identical to v8/v9 (the geometry that gave off_road 0.0
# and held collisions at 0.30). The experiment: can the fixed critic reach a
# higher free speed WITHOUT the collision blow-up that hit V6/V7 (which happened
# because the critic diverged)? Positive-optimum invariant holds: at v=10 open
# road tent(+10)+bump(+8) ~ +18/step; at 6.82 ~ +9 — so faster pays ~2x.
# Use with env desired_speed=12 (tent peaks above the 10 target).
CRUISE_V12_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    target_speed=10.0,
    speed_weight=8.0,             # doubled from v8/v9's 4.0 — the speed-push lever
    speed_sigma=2.0,             # widened from 1.0 so the bump pulls 6 -> 10
    bump_vdes_power=1.0,
    low_speed_weight=0.0,
    overspeed_weight=-2.0,
    overspeed_margin=1.0,
    tailgate_weight=-8.0,
    pedal_jerk_weight=-0.3,
    curve_gain=2.0,              # KEEP — preserves the off_road 0.0 cornering
    ttc_weight=-3.0,
    junction_ttc_weight=0.0,
    junction_slow_norm=0.3,      # KEEP junction safety (5 m/s into junctions)
    junction_target_speed=5.0,
)

# survive_v1(長程穩定駕駛專案):**保留** env 內建速度項當前進獎勵(replace_speed_term=False
# ——它本身就是「開車為正、停住≈0、撞車-100」的正確結構),再加一個 progress 小獎勵
# (min(v,8)/8,封頂在 8、停住為 0)強化「維持速度」,以及卡住一次性懲罰。**不用** alive
# 加分(會被停住farming)、**不用** crawl 扣分(會誘發自殺)。steer_jerk=0(Pure-Pursuit)。
# 搭配 env desired_speed=8 使用。目標:長時間零違規穩定行駛,且車速不太慢。
SURVIVE_V1_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=8.0,
    survive_alive_bonus=0.0,      # 保留 env 速度項當前進獎勵,不用 alive(會被 farming)
    survive_crawl_penalty=0.0,    # 關閉(會誘發自殺);速度梯度由 env tent + progress 提供
    survive_stuck_penalty=-10.0,  # 卡住一次性懲罰(<< 撞車 -100,不誘發自殺)
)

# avoid_v1(RL-核心避撞):= survive_v1(保留 env 速度項 + progress + 卡住懲罰;不震盪靠幾何轉向)
# 再加 TTC 接近懲罰,讓 RL 自學避撞。推論期不掛任何規則安全層。搭配 env desired_speed=8、obs 323。
AVOID_V1_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=8.0,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-6.0,        # TTC 低於門檻就罰(RL 自學避撞的梯度)
    avoid_ttc_floor_norm=0.4,     # ttc_norm<0.4(實際 <2s)才開始罰
    pedal_jerk_weight=-0.3,
)

# avoid_v2(加重避撞,診斷:撞車太便宜 → RL 學會「開快、偶爾撞無所謂」,裸跑碰撞 0.96)。
# 對策:(1) TTC 懲罰加重 5×(-6→-30)且提早觸發(門檻 0.4→0.6,即 <3s 就罰)→ 衝突逼近時
# 密集負訊號壓過速度誘因,逼 RL 提前煞;(2) 加大撞車終止懲罰 -500(env -100 太便宜)。
# 速度 target 仍 8(使用者要 >7)。
AVOID_V2_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=8.0,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-30.0,        # 5× 加重
    avoid_ttc_floor_norm=0.6,      # 提早觸發(<3s 就罰)
    avoid_collision_penalty=-500.0,  # 撞車變貴(<< env -100)
    pedal_jerk_weight=-0.3,
)

# avoid_v3(再推 RL:更強 + 更長 + 換感知)。搭配 5 扇區風險感知(risk_features v3,看得到
# 路口橫切來車,obs 327)。vs v2:TTC 懲罰再加重(-30→-50)、再提早(門檻 0.6→0.7,<3.5s 就罰)
# → 更早、更強地壓過速度誘因。撞車 -500、卡住 -10 維持。速度 target 8(使用者要 >7)。
AVOID_V3_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=8.0,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,        # 更強
    avoid_ttc_floor_norm=0.7,      # 更早(<3.5s 就罰)
    avoid_collision_penalty=-500.0,
    pedal_jerk_weight=-0.3,
)

# avoid_v4(根因修法:距離缺口獎勵)。= avoid_v3(保留 env 速度項 + progress + 卡住懲罰 + TTC
# 懲罰 + 撞車 -500)再「加上」距離缺口懲罰——這是診斷出的關鍵:碰撞 94% 是低速潛行頂前車,TTC 型
# 訊號近距低速盲,故補一個「距離型」密集梯度逼 RL 學會在前車後停住並 hold。TTC 懲罰保留(抓較高速
# 接近),兩者互補(對應「距離護盾 0.28 勝過 TTC 護盾 0.40」的實測)。搭配 5 扇區風險感知 obs 327、
# env desired_speed=8、Pure-Pursuit 轉向、推論期可再疊薄距離護盾當安全網。
AVOID_V4_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=8.0,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,        # 保留 avoid_v3 的 TTC 懲罰(抓較高速接近)
    avoid_ttc_floor_norm=0.7,
    avoid_collision_penalty=-500.0,
    avoid_gap_weight=-8.0,         # 新增:距離缺口懲罰(根因修法,逼近前車的密集減速梯度)
    avoid_gap_floor=7.0,
    pedal_jerk_weight=-0.3,
)

# avoid_v5(拉速度):= avoid_v4(根因距離缺口獎勵 + TTC + 撞車/卡住懲罰)再「加大 progress 權重」
# 把自由車速拉高。關鍵洞察:開闊路(無前車)時 gap/TTC 懲罰皆為 0,唯一限速的是 progress 斜率太弱,
# 故策略只開到 ~5.9。加大 progress 權重(1→4)直接在「安全的開闊路」拉高車速,而在車流中安全懲罰
# 仍在 → 不會變魯莽。搭配訓練 env desired_speed=10(放寬速度 tent 上限)。推論期掛 DSAFE+護盾 backstop。
AVOID_V5_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=10.0,           # 封頂拉到 10(原 8)→ 鼓勵到 10 才飽和
    survive_progress_weight=4.0,     # 拉速度的主槓桿(原 1.0)
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,          # 保留 avoid_v4 全部安全訊號(車流中仍謹慎)
    avoid_ttc_floor_norm=0.7,
    avoid_collision_penalty=-500.0,
    avoid_gap_weight=-8.0,
    avoid_gap_floor=7.0,
    pedal_jerk_weight=-0.3,
)

# avoid_light_v1:在 avoid_v5(已驗證的避撞/速度配置)之上加「紅綠燈反應」。必須以 traffic='on'
# 訓練,紅/黃/綠 one-hot 才會變動、紅燈懲罰才會觸發(v5 全程凍結綠燈 → 從未學過停紅燈)。
#  - red_light_creep_weight=-5.0:紅燈觸發區內每步按速度密集扣分(主梯度,逼策略煞停)。
#  - red_light_penalty=-100:闖過紅燈線的一次性事件懲罰(加重至 -100)。
# 其餘安全/速度項全沿用 v5;對向誤煞已在 risk_features/path_aware_lead 加朝向閘門根除。
AVOID_LIGHT_V1_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=10.0,
    survive_progress_weight=4.0,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,
    avoid_ttc_floor_norm=0.7,
    avoid_collision_penalty=-500.0,
    avoid_gap_weight=-8.0,
    avoid_gap_floor=7.0,
    pedal_jerk_weight=-0.3,
    red_light_penalty=-100.0,
    red_light_creep_weight=-5.0,
)

# avoid_light_v2:紅燈 RL「完整重做」。前提是燈號觀測已改成「前瞻」(traffic_light.py:看得到前方
# 紅燈),否則學不起來。設計:
#  - red_light_terminal=True + red_light_penalty=-300:闖紅燈 = 終止回合 + 大懲罰(撞車同級),
#    堵住「闖過去繼續賺速度獎勵」的漏洞(前兩次失敗的根因)。
#  - red_light_creep_weight=-5.0:前方有紅燈仍前進 → 密集減速懲罰(配合前瞻 → 遠遠就減速)。
#  - 其餘沿用 avoid_v5 的避撞/速度項;對向誤煞已由 risk/path_aware 朝向閘門根除。
AVOID_LIGHT_V2_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=10.0,
    survive_progress_weight=4.0,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,
    avoid_ttc_floor_norm=0.7,
    avoid_collision_penalty=-500.0,
    avoid_gap_weight=-8.0,
    avoid_gap_floor=7.0,
    pedal_jerk_weight=-0.3,
    red_light_penalty=-300.0,
    red_light_creep_weight=-5.0,
    red_light_terminal=True,
)

# avoid_light_v3:在 avoid_light_v2(紅燈終止 + 前瞻 + 避撞,純策略已零違規/零碰撞但過度保守 ~2 m/s)
# 之上「把自由路速度推上 8+」。只在「淨空路段」(無前車 gap<=0、高 TTC、遠離路口/紅燈 jd>=0.6)給
# 加速獎勵,因此不會誘導衝紅燈;紅燈仍由終止懲罰強制停。提高 v_target 與 free-speed target 到 10/11。
AVOID_LIGHT_V3_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=11.0,            # progress 項封頂拉到 11(原 10)→ 鼓勵更快
    survive_progress_weight=4.0,
    survive_v_floor=5.0,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,
    avoid_ttc_floor_norm=0.7,
    avoid_collision_penalty=-500.0,
    avoid_gap_weight=-8.0,
    avoid_gap_floor=7.0,
    pedal_jerk_weight=-0.3,
    red_light_penalty=-300.0,
    red_light_creep_weight=-5.0,
    red_light_terminal=True,
    # 淨空路段加速獎勵(推自由速度):floor 6 → target 10,只在 clear 時給。
    avoid_free_speed_weight=4.0,
    avoid_free_speed_floor=6.0,
    avoid_free_speed_target=10.0,
    avoid_free_speed_ttc_floor_norm=0.85,
    avoid_free_speed_junction_clear=0.6,
)

# avoid_light_v4:修兩個問題 —— (1) 紅燈停太早(離停止線很遠):creep 懲罰加距離閘門
# red_light_creep_dist_norm=0.24(=12m),只有接近停止線才罰移動,遠處照常開;
# (2) 自由速度平均要 ≥6:free-speed 獎勵 floor 降到 4(更早介入)、啟用淨空路爬行懲罰
# survive_crawl_penalty=-2(慢於 5 m/s 扣分)、加大 progress/free-speed 權重。其餘沿用 v3。
AVOID_LIGHT_V4_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=11.0,
    survive_progress_weight=5.0,         # 4→5 加強速度
    survive_v_floor=5.0,
    survive_crawl_penalty=-2.0,          # 淨空路慢於 5 m/s 扣分(推平均速度)
    survive_junction_clear=0.4,
    survive_alive_bonus=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,
    avoid_ttc_floor_norm=0.7,
    avoid_collision_penalty=-500.0,
    avoid_gap_weight=-8.0,
    avoid_gap_floor=7.0,
    pedal_jerk_weight=-0.3,
    red_light_penalty=-300.0,
    red_light_creep_weight=-5.0,
    red_light_terminal=True,
    red_light_creep_dist_norm=0.24,      # 只在 ~12m 內才罰移動 → 不再 40m 外超早停
    avoid_free_speed_weight=5.0,         # 4→5
    avoid_free_speed_floor=4.0,          # 6→4:從 4 起拉,推平均自由速度 ≥6
    avoid_free_speed_target=10.0,
    avoid_free_speed_ttc_floor_norm=0.85,
    avoid_free_speed_junction_clear=0.6,
)

# avoid_v6: speed-up without raising collision risk. Keep the avoid_v5 safety
# terms, reduce broad progress, and add speed reward only when the road is
# clear: no path-aware lead, min_ttc is high, and not near a junction.
AVOID_V6_REWARD_CONFIG = RewardConfig(
    steer_jerk_weight=0.0,
    replace_speed_term=False,
    survive=True,
    survive_v_target=10.0,
    survive_progress_weight=2.5,
    avoid_free_speed_weight=3.0,
    avoid_free_speed_floor=6.0,
    avoid_free_speed_target=9.0,
    avoid_free_speed_ttc_floor_norm=0.85,
    avoid_free_speed_junction_clear=0.6,
    survive_alive_bonus=0.0,
    survive_crawl_penalty=0.0,
    survive_stuck_penalty=-10.0,
    avoid_ttc_weight=-50.0,
    avoid_ttc_floor_norm=0.7,
    avoid_collision_penalty=-500.0,
    avoid_gap_weight=-8.0,
    avoid_gap_floor=7.0,
    pedal_jerk_weight=-0.3,
)

REWARD_PRESETS = {
    'default': DEFAULT_REWARD_CONFIG,
    'cruise': CRUISE_REWARD_CONFIG,
    'cruise2': CRUISE2_REWARD_CONFIG,
    'cruise3': CRUISE3_REWARD_CONFIG,
    'cruise4': CRUISE4_REWARD_CONFIG,
    'cruise5': CRUISE5_REWARD_CONFIG,
    'cruise_v2': CRUISE_V2_REWARD_CONFIG,
    'cruise_v2b': CRUISE_V2B_REWARD_CONFIG,
    'cruise_fast': CRUISE_FAST_REWARD_CONFIG,
    'cruise_safe': CRUISE_SAFE_REWARD_CONFIG,
    'cruise_v7': CRUISE_V7_REWARD_CONFIG,
    'cruise_v8': CRUISE_V8_REWARD_CONFIG,
    'cruise_v9': CRUISE_V9_REWARD_CONFIG,
    'cruise_v12': CRUISE_V12_REWARD_CONFIG,
    'survive_v1': SURVIVE_V1_REWARD_CONFIG,
    'avoid_v1': AVOID_V1_REWARD_CONFIG,
    'avoid_v2': AVOID_V2_REWARD_CONFIG,
    'avoid_v3': AVOID_V3_REWARD_CONFIG,
    'avoid_v4': AVOID_V4_REWARD_CONFIG,
    'avoid_v5': AVOID_V5_REWARD_CONFIG,
    'avoid_light_v1': AVOID_LIGHT_V1_REWARD_CONFIG,
    'avoid_light_v2': AVOID_LIGHT_V2_REWARD_CONFIG,
    'avoid_light_v3': AVOID_LIGHT_V3_REWARD_CONFIG,
    'avoid_light_v4': AVOID_LIGHT_V4_REWARD_CONFIG,
    'avoid_v6': AVOID_V6_REWARD_CONFIG,
}
