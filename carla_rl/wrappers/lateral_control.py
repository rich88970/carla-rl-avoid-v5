"""Geometric lateral controller — the hierarchical-control fix for steering weave.

Four iterations proved reward shaping / behavior anchoring / EMA filtering
cannot stop a neural net that outputs steering directly from weaving. The
project plan's reserved answer: hand steering to a geometric controller and
let RL keep only the longitudinal (throttle/brake) decision.

PurePursuitController computes a steering command from the reference path that
is already in the observation (the 12 lookahead waypoints, obs[271:307], in
ego-frame local x/y). A geometric controller is smooth by construction — there
is no per-step neural jitter to suppress, so body weave drops to the
controller's (autopilot-class) level.

HybridSteerWrapper exposes the same 3-D action space (so the existing BC/SAC
actors load unchanged) but OVERWRITES the steering channel with the controller
and reports the executed action in info['applied_action'] (so the replay
buffer and reward see what actually drove the car).
"""

import os

import gym
import numpy as np

from carla_rl.wrappers.risk_features import N_RISK_FEATURES

WAYPOINT_SLICE = slice(271, 307)   # 12 x (local_x, local_y, rel_yaw)
WHEELBASE = 2.875                  # Tesla Model 3, m


def _envf(name, default):
    """Inference-time safety knobs are env-overridable (CARLA_*) so the safety
    aggressiveness can be swept at eval without editing/retraining anything."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class PurePursuitController:
    # ld = max(8, 1.0 * speed_smoothed). Constant-speed verify: yaw_jerk 0.712,
    # below autopilot's 1.722. The lookahead uses a SMOOTHED speed and the
    # output is low-passed, so RL longitudinal jitter (pedal surge -> speed
    # wobble) cannot bleed into the steering (the cornering-weave root cause).
    def __init__(self, lookahead_min=8.0, lookahead_k=1.0, gain=1.0,
                 max_steer_rad=1.0, speed_ema=0.7, steer_ema=0.5):
        self.ld_min = lookahead_min
        self.ld_k = lookahead_k
        self.gain = gain
        self.max_steer_rad = max_steer_rad
        self.speed_ema = speed_ema
        self.steer_ema = steer_ema
        self._v_smooth = None
        self._prev_steer = 0.0

    def reset(self):
        self._v_smooth = None
        self._prev_steer = 0.0

    def steer(self, obs):
        wps = np.asarray(obs[WAYPOINT_SLICE], dtype=np.float64).reshape(12, 3)
        speed = float(obs[3])
        self._v_smooth = (speed if self._v_smooth is None
                          else (1 - self.speed_ema) * speed + self.speed_ema * self._v_smooth)
        ld = max(self.ld_min, self.ld_k * self._v_smooth)
        dists = np.hypot(wps[:, 0], wps[:, 1])
        ahead = np.flatnonzero((dists >= ld) & (wps[:, 0] > 0))
        idx = int(ahead[0]) if ahead.size else 11
        tx, ty = wps[idx, 0], wps[idx, 1]
        l = max(float(np.hypot(tx, ty)), 0.5)
        if l < 0.6 or tx <= 0:          # no usable waypoint -> hold straight
            return 0.0
        alpha = np.arctan2(ty, tx)      # bearing to the lookahead point
        delta = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), l)
        raw = float(np.clip(self.gain * delta / self.max_steer_rad, -1.0, 1.0))
        out = (1.0 - self.steer_ema) * raw + self.steer_ema * self._prev_steer
        self._prev_steer = out
        return out


class StanleyController:
    """Stanley lateral control: steer = heading_error + atan(k * cross_track /
    (v + eps)). Tracks the nearest path point's heading and lateral offset
    directly, so it doesn't 'cut' corners the way pure-pursuit's lookahead can.
    Reads the nearest waypoint from the obs: rel_yaw = heading error, local_y =
    cross-track error. A small steer EMA caps high-frequency output."""

    def __init__(self, k=0.5, max_steer_rad=1.0, smooth=0.3, eps=1.0):
        self.k = k
        self.max_steer_rad = max_steer_rad
        self.smooth = smooth
        self.eps = eps
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def steer(self, obs):
        wps = np.asarray(obs[WAYPOINT_SLICE], dtype=np.float64).reshape(12, 3)
        speed = float(obs[3])
        # nearest waypoint ahead with real data
        idx = 0
        for i in range(12):
            if wps[i, 0] > 0 or abs(wps[i, 1]) > 1e-6:
                idx = i
                break
        heading_err = float(wps[idx, 2])         # rel_yaw of the path
        cross_track = float(wps[idx, 1])          # ego-frame lateral offset
        delta = heading_err + np.arctan2(self.k * cross_track, speed + self.eps)
        raw = float(np.clip(delta / self.max_steer_rad, -1.0, 1.0))
        out = (1.0 - self.smooth) * raw + self.smooth * self._prev
        self._prev = out
        return out


# Longitudinal safety override (V7): the diagnosis showed the policy throttles
# INTO the car ahead at junctions (rear-ends, mean throttle 0.33 > brake 0.19).
# Like pure-pursuit guarantees no steering weave, this guarantees no rear-end:
# when a detected lead is critically close OR closing within the TTC window,
# force the brake regardless of what the policy commanded. Only fires with a
# real on-path lead (path_aware_lead, gap>0), so it never brakes on open road.
SAFETY_CRITICAL_GAP = _envf('CARLA_SAFETY_GAP', 7.0)   # m center-to-center (~2.3 m clear)
SAFETY_TTC = _envf('CARLA_SAFETY_TTC', 2.5)            # s — brake if a closing lead hits within this
SAFETY_CLOSING_MIN = 0.3    # m/s — minimum closing speed to compute TTC
SAFETY_BRAKE = 0.7

# 統一安全控制器(把 V7 的 safety_brake「固定 7m + 二元硬煞」與臨時 TTC 護盾重寫成一個乾淨控制器,
# 同時做三件事:動態安全距離 + 分級煞車 + TTC):
#   - 動態安全距離 d_safe = d0 + τ·v + v²/(2a)(反應距離 + 物理煞停距離,隨速縮放)。低速 d_safe 小
#     → 不對「停住的前車」過早誤煞 → 降低卡住;高速 d_safe 大 → 提早反應 → 降低碰撞。
#   - 分級煞車:力度 ∝ 侵入安全距離的深度,並同步收油 → 平滑減速、貼到 d_safe 後 hold 住(非二元
#     剎死)→ 再降低卡住。
#   - TTC:逼近速度快(closing TTC 低)時直接全力煞 → 補抓高速接近(診斷出的少數較高速前撞)。
# 註:前車距為中心對中心,接觸約 4.7m,故 d0 須 > 4.7(留淨空)。env 變數 CARLA_DSAFE 開啟。
DSAFE_D0 = _envf('CARLA_DSAFE_D0', 5.5)      # m,靜止最小間距(>接觸距 4.7,留 ~0.8m 淨空)
DSAFE_TAU = _envf('CARLA_DSAFE_TAU', 0.6)    # s,反應時間
DSAFE_AMAX = _envf('CARLA_DSAFE_AMAX', 3.5)  # m/s²,假設可用減速度(算物理煞停距離)
DSAFE_TTC = _envf('CARLA_DSAFE_TTC', 2.0)    # s,逼近 TTC 低於此 → 全力煞(高速接近硬底線)


# 側向緊急閃避(AES):backstop 只會煞車,但有些迫近碰撞煞車已來不及。此函式在「正前方迫近且
# 煞不住」時,往「確認非常空曠」的一側加一個轉向偏移閃開——唯一能「不減速也降碰撞」的槓桿。
# 嚴格門檻(只在真的要撞且某側全空時才動),避免平時抖動或衝出路面。風險:密集車流側向閃避易撞
# 鄰道/出界,故 eval 後若 collision/off_road 上升即代表此槓桿無效或方向錯,須還原。
# 觀測 risk 區塊(ego frame,y=左、+bearing=左,與 waypoint 同框)10 維:
#   [ttc_f,dist_f, ttc_lf,dist_lf, ttc_rf,dist_rf, ttc_l,dist_l, ttc_r,dist_r](皆正規化 [0,1])
EVADE_TTC = _envf('CARLA_EVADE_TTC', 0.15)        # 正前方 ttc_norm 低於此(≈0.75s)才考慮閃
EVADE_DIST = _envf('CARLA_EVADE_DIST', 0.18)      # 正前方 dist_norm 低於此(≈7m)才考慮閃
EVADE_SIDE_CLEAR = _envf('CARLA_EVADE_SIDE_CLEAR', 0.5)   # 目標側 dist_norm 須 > 此(≈20m)才敢閃
EVADE_OFFSET = _envf('CARLA_EVADE_OFFSET', 0.4)   # 轉向偏移量(+ = 左,與 Pure-Pursuit 同號)


def evasion_steer(rf, base_steer, ttc_trig=EVADE_TTC, dist_trig=EVADE_DIST,
                  side_clear=EVADE_SIDE_CLEAR, offset=EVADE_OFFSET):
    """側向緊急閃避。rf = 10 維 risk 區塊。回傳 (steer, fired)。
    僅當正前方迫近(ttc_f<ttc_trig 且 dist_f<dist_trig,煞車已來不及),且某一側「非常空曠」
    (該側 dist>side_clear 且 ttc 安全>0.5)時,往該側加 offset(+左/-右,與 Pure-Pursuit 同號);
    兩側皆不夠空則不閃(只靠煞車)。門檻嚴格 → 平時不動,僅真要撞才介入。"""
    rf = [float(x) for x in rf]
    ttc_f, dist_f = rf[0], rf[1]
    if not (ttc_f < ttc_trig and dist_f < dist_trig):
        return base_steer, False
    left_clear = min(rf[3], rf[7])    # dist_lf, dist_l
    left_safe = min(rf[2], rf[6])     # ttc_lf, ttc_l
    right_clear = min(rf[5], rf[9])   # dist_rf, dist_r
    right_safe = min(rf[4], rf[8])    # ttc_rf, ttc_r
    if left_clear >= right_clear and left_clear > side_clear and left_safe > 0.5:
        return float(np.clip(base_steer + offset, -1.0, 1.0)), True    # 往左(+)
    if right_clear > left_clear and right_clear > side_clear and right_safe > 0.5:
        return float(np.clip(base_steer - offset, -1.0, 1.0)), True    # 往右(-)
    return base_steer, False


def dynamic_safe_gap(v, d0=DSAFE_D0, tau=DSAFE_TAU, a=DSAFE_AMAX):
    """動態安全距離(m)= 最小間距 d0 + 反應距離 τ·v + 物理煞停距離 v²/(2a)。隨速單調遞增。"""
    v = max(0.0, float(v))
    return d0 + tau * v + v * v / (2.0 * max(a, 0.1))


def graduated_brake(gap, v, closing, cmd_throttle, cmd_brake,
                    d0=DSAFE_D0, tau=DSAFE_TAU, a=DSAFE_AMAX, ttc_floor=DSAFE_TTC):
    """統一安全控制器的純函式(兩區式 AEB:緊急內圈 + 舒適外圈)。回傳 (throttle, brake)。
    gap<=0(前方無 on-path 車)或 gap>=d_safe(安全距離外)→ 不介入,RL 指令原樣通過。
    gap<=d0(緊急內圈)→ 全力煞、收油(決斷止住低速潛行,94% 失效)。
    d0<gap<d_safe(舒適外圈)→ 分級:deficit=(d_safe-gap)/(d_safe-d0) ∈ (0,1),brake∝deficit、
    throttle 隨 deficit 收油(高速提早平滑減速、降卡住);逼近 TTC<ttc_floor → 全力煞(高速接近硬底線)。"""
    t, b = float(cmd_throttle), float(cmd_brake)
    if gap <= 0.0:
        return t, b
    # 內圈(緊急區):gap 已進到最小安全距 d0 內 → 果斷全力煞、收油。這是止住「低速潛行頂前車」
    # (佔 94% 失效)的關鍵——分級的溫和煞車太弱止不住潛行,必須像 AEB 的緊急區一樣決斷。
    if gap <= d0:
        return 0.0, max(b, 1.0)
    d_safe = dynamic_safe_gap(v, d0, tau, a)
    if gap >= d_safe:
        return t, b
    # 外圈(舒適/預期區):d0..d_safe 之間分級——高速時提早、平滑減速(降卡住),貼近 d0 才接近全力。
    deficit = min(1.0, max(0.0, (d_safe - gap) / max(d_safe - d0, 1e-6)))
    b = max(b, deficit)               # 分級:越深入越大力(gap→d0 時 →1,與內圈連續)
    t = t * (1.0 - deficit)           # 收油:越深入越收
    if closing > SAFETY_CLOSING_MIN and (gap / closing) < ttc_floor:
        b = max(b, 0.9)               # TTC 硬底線:高速接近 → 全力煞
        t = 0.0
    return t, b


def project_3d_pedals(throttle, brake):
    """把舊三維 SAC 的獨立油門/煞車投影成單一可執行踏板,規則與 CarlaGymEnv.step() 一致:
      throttle >= brake → 保留油門、清零煞車;  brake > throttle → 保留煞車、清零油門。
    平手刻意偏向油門(沿用底層環境介面與舊 checkpoint 語意)。
    取代舊的「brake>0 即清零油門」——Gaussian actor 的 brake 幾乎不會剛好 0,微量煞車會被放大成
    全斷油而僵住(50/100/150 車零移動的根因,屬控制介面回歸 bug,非策略本身或安全層)。"""
    throttle = float(np.clip(throttle, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    if throttle >= brake:
        return throttle, 0.0
    return 0.0, brake


# Predictive safety planner (V9): unlike the reactive brake above, this predicts
# ALL nearby vehicles a few seconds ahead and caps the ego speed to the highest
# value that keeps PLAN_MARGIN from every predicted trajectory over the horizon.
# Because it predicts, it slows EARLY -> effective at high speed, omnidirectional.
PLAN_HORIZON = _envf('CARLA_PLAN_HORIZON', 3.0)     # s
PLAN_MARGIN = _envf('CARLA_PLAN_MARGIN', 3.0)       # m keep-out radius (ego + other half-widths + buffer)
PLAN_VMAX = 12.0       # m/s, top of the candidate-speed search
_PLAN_DT = 0.2
# slows-not-freeze:planner 為「預測中」的衝突把速度上限降到 0 會讓車在「已淨空的路」上
# 凍結(路口側向來車煞停後無法重新起步 → stuck 終止,長程評估的主要失效)。除非有車「當下」
# 已迫近(在 keep-out 半徑內),否則把速度上限墊在一個低 creep 速度,讓車緩行通過、不凍結。
PLAN_MIN_CREEP = _envf('CARLA_PLAN_MIN_CREEP', 1.5)  # m/s;0 = 關閉(回到會完全煞停的舊行為)


def safe_speed_cap(vehicles, ego_speed, v_max=PLAN_VMAX):
    """Highest ego speed (m/s, along +x in the ego frame) that keeps PLAN_MARGIN
    from every vehicle's constant-velocity prediction over PLAN_HORIZON. Searches
    candidate speeds top-down and checks the FULL ego-vs-vehicle distance over the
    horizon (so vehicles behind / being passed correctly impose no cap). Returns
    a large value (no cap) when there are no vehicles."""
    veh = np.asarray(vehicles, dtype=np.float64).reshape(-1, 4)
    if veh.shape[0] == 0:
        return 1e9
    ts = np.arange(1, int(PLAN_HORIZON / _PLAN_DT) + 1) * _PLAN_DT      # (T,)
    vx = veh[:, 3] * np.cos(veh[:, 2])
    vy = veh[:, 3] * np.sin(veh[:, 2])
    px = veh[:, 0:1] + vx[:, None] * ts[None, :]                        # (M,T)
    py = veh[:, 1:2] + vy[:, None] * ts[None, :]                        # (M,T)
    for v in np.arange(v_max, -1e-9, -0.5):
        d = np.hypot(v * ts[None, :] - px, py)                         # (M,T)
        if np.all(d >= PLAN_MARGIN):
            return float(v)
    return 0.0


def _imminent(vehicles):
    """是否有車「當下」就在 keep-out 半徑內、且不在正後方(x>=-1,即前方或旁邊)。
    用來判斷是否該允許完全煞停;若只是『預測中』的遠處衝突,則改為緩行不凍結。"""
    veh = np.asarray(vehicles, dtype=np.float64).reshape(-1, 4)
    if veh.shape[0] == 0:
        return False
    ahead = veh[veh[:, 0] >= -1.0]           # 排除正後方(被超過)的車
    if ahead.shape[0] == 0:
        return False
    return bool(np.any(np.hypot(ahead[:, 0], ahead[:, 1]) < PLAN_MARGIN))


class SafetyPlanner:
    """Caps throttle to the predictive safe speed; full-ish brake when over it.
    slows-not-freeze:除非有車當下迫近,否則速度上限不低於 PLAN_MIN_CREEP(緩行而非凍結)。"""

    def cap_action(self, throttle, brake, ego_speed, vehicles):
        cap = safe_speed_cap(vehicles, ego_speed)
        # 只在「當下」就有車迫近時才允許完全停;否則墊高到 creep 速度,讓車緩行通過、
        # 不會在路口為遠處側向來車凍結(長程 stuck 的主因)。
        if PLAN_MIN_CREEP > 0.0 and cap < PLAN_MIN_CREEP and not _imminent(vehicles):
            cap = PLAN_MIN_CREEP
        if ego_speed > cap + 0.5:
            return 0.0, max(brake, 0.6)
        if ego_speed > cap:
            return 0.0, max(brake, 0.3)
        return throttle, brake


class HybridSteerWrapper(gym.Wrapper):
    def __init__(self, env, controller=None, throttle_ema=0.0, safety_brake=False,
                 safety_planner=False, signed_pedal=False):
        super().__init__(env)
        self.controller = controller or PurePursuitController()
        self.throttle_ema = throttle_ema
        self.safety_brake = safety_brake
        self.safety_planner = safety_planner
        self.planner = SafetyPlanner() if safety_planner else None
        self._last_obs = None
        self._prev_throttle = 0.0
        # signed-pedal 模式:SAC 動作改為一維 u∈[-1,1](u≥0→油門 u、u<0→煞車 -u);方向盤仍
        # 由 Pure Pursuit。消除「油門/煞車雙輸出 + 無效方向盤梯度」與「動作/執行/buffer 不一致」。
        self.signed_pedal = signed_pedal
        if signed_pedal:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,),
                                               dtype=np.float32)
        # 輕量 TTC 護盾(post-shield):診斷顯示 RL「看到危險卻不煞」是主因,而 min_ttc 就在
        # 觀測的 risk 區塊裡(策略看得到、只是沒照做)。此護盾讀那個 min_ttc,< 門檻就強制煞車
        # ——用 RL 自己的感知做薄 backstop,精準補「看到卻不煞」。env 變數 CARLA_TTC_SHIELD 開啟
        # (預設 0 = 關)。只在 risk_obs(obs 帶 risk 區塊)時生效。
        self.ttc_shield = _envf('CARLA_TTC_SHIELD', 0.0)
        self.ttc_shield_brake = _envf('CARLA_TTC_SHIELD_BRAKE', 0.8)
        # 統一安全控制器(動態安全距離 + 分級煞車 + TTC),env 變數 CARLA_DSAFE 開啟(預設關)。
        self.dynamic_safety = _envf('CARLA_DSAFE', 0.0) > 0.0
        # 側向緊急閃避(AES),env 變數 CARLA_EVADE 開啟(預設關)。
        self.evade = _envf('CARLA_EVADE', 0.0) > 0.0

    def _world_vehicles_ego_frame(self):
        """All other vehicles within 40 m as ego-frame [x, y, rel_yaw, speed]."""
        tr = self.env.ego.get_transform()
        eyaw = np.deg2rad(tr.rotation.yaw)
        c, s = np.cos(-eyaw), np.sin(-eyaw)
        ex, ey = tr.location.x, tr.location.y
        rows = []
        for v in self.env.world.get_actors().filter('vehicle.*'):
            if v.id == self.env.ego.id:
                continue
            loc = v.get_transform().location
            dx, dy = loc.x - ex, loc.y - ey
            if dx * dx + dy * dy > 1600.0:
                continue
            lx = c * dx - s * dy
            ly = s * dx + c * dy
            vel = v.get_velocity()
            rows.append([lx, ly, np.deg2rad(v.get_transform().rotation.yaw) - eyaw,
                         float(np.hypot(vel.x, vel.y))])
        return np.asarray(rows, dtype=np.float64).reshape(-1, 4)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_obs = obs
        self._prev_throttle = 0.0
        if hasattr(self.controller, 'reset'):
            self.controller.reset()
        return obs

    def step(self, action):
        steer = self.controller.steer(self._last_obs)
        # 側向緊急閃避:正前方迫近且煞不住時,往確認空曠的一側偏轉(讀觀測 risk 區塊)。
        if self.evade:
            lo = np.asarray(self._last_obs, dtype=np.float64)
            if lo.size >= N_RISK_FEATURES:
                steer, _ = evasion_steer(lo[-N_RISK_FEATURES:], steer)
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.signed_pedal:
            # 一維 signed pedal u∈[-1,1] 本就互斥
            u = float(np.clip(raw_action[0], -1.0, 1.0))
            raw_policy_throttle = max(u, 0.0)
            raw_policy_brake = max(-u, 0.0)
            throttle, brake = raw_policy_throttle, raw_policy_brake
        else:
            # 舊 3D checkpoint [throttle, steer, brake]:先依底層 env 規則「依大小互斥」投影成單一踏板
            raw_policy_throttle = float(np.clip(raw_action[0], 0.0, 1.0))
            raw_policy_brake = float(np.clip(raw_action[2], 0.0, 1.0))
            throttle, brake = project_3d_pedals(raw_policy_throttle, raw_policy_brake)
        # 煞車立即生效;只有油門用 EMA 平滑(投影後 brake>0 代表「真的選擇煞車」,才清零油門)
        if brake > 0.0:
            throttle = 0.0
            self._prev_throttle = 0.0
        elif self.throttle_ema > 0.0:
            throttle = ((1.0 - self.throttle_ema) * throttle
                        + self.throttle_ema * self._prev_throttle)
            throttle = float(np.clip(throttle, 0.0, 1.0))
            self._prev_throttle = throttle
        else:
            self._prev_throttle = throttle
        # policy_*:踏板投影 + EMA 後、尚未進安全層的控制(供 trace 區分「策略」vs「安全層覆寫」)
        policy_throttle = float(throttle)
        policy_brake = float(brake)
        ttc_shield_fired = False
        dsafe_active = False
        shield_ttc = float('nan')
        dsafe_d = float('nan')
        dsafe_gap = float('nan')
        if self.safety_brake:
            from carla_rl.wrappers.lead_vehicle import path_aware_lead
            ego_speed = float(self._last_obs[3])
            # Two lead sources, braked on whichever sees a closer/sooner lead:
            #  - path_aware_lead: on-route, but only the 5 nearest vehicles
            #  - obs[7]/obs[8]: the env's forward box over ALL vehicles (catches
            #    the lead the 5-nearest list misses at busy junctions; phantom-
            #    prone in turns, acceptable for an emergency brake)
            gp, ls = path_aware_lead(np.asarray(self._last_obs, dtype=np.float64)[:307])
            leads = []
            if gp > 0.0:
                leads.append((gp, ego_speed - ls))
            ge, re = float(self._last_obs[7]), float(self._last_obs[8])
            if ge > 0.0:
                leads.append((ge, re))
            for gap, closing in leads:
                ttc = gap / closing if closing > SAFETY_CLOSING_MIN else 1e9
                if gap < SAFETY_CRITICAL_GAP or ttc < SAFETY_TTC:
                    throttle = 0.0
                    brake = max(brake, SAFETY_BRAKE)
                    break
        if self.safety_planner:
            throttle, brake = self.planner.cap_action(
                throttle, brake, float(self._last_obs[3]),
                self._world_vehicles_ego_frame(),
            )
        # 輕量 TTC 護盾:從觀測的 risk 區塊(最後 N_RISK_FEATURES 維,5 扇區 × [ttc,dist])取
        # 最小 ttc(偶數索引),< 門檻就強制煞車。直接補診斷出的「看到卻不煞」。
        if self.ttc_shield > 0.0:
            lo = np.asarray(self._last_obs, dtype=np.float64)
            if lo.size >= N_RISK_FEATURES:
                risk_ttc = float(lo[-N_RISK_FEATURES::2].min())   # 5 扇區 ttc 的最小值
                shield_ttc = risk_ttc
                if risk_ttc < self.ttc_shield:
                    throttle = 0.0
                    brake = max(brake, self.ttc_shield_brake)
                    ttc_shield_fired = True
        # 統一安全控制器(動態安全距離 + 分級煞車 + TTC,取代固定 7m 二元 safety_brake)。
        # 兩個前車來源,取最近者套用分級煞車:
        #  - path_aware_lead:沿(可能彎曲的)路徑走廊,僅 5 最近車。
        #  - world_forward_lead:掃 40m 內「全部」車輛、同向同車道過濾(補 5 最近漏掉的第 6+ 近前車)。
        # 原本第二來源是 env 的 obs[7]/[8] 前向探測箱(±2.5m 直箱、無朝向過濾),轉彎時會把對向車道
        # 來車當前車誤煞、無謂壓低車速(lead_vehicle.py:轉彎 40% 幻影)。改用有朝向閘門的
        # world_forward_lead → 保留「同向前車提早分級煞車」的防撞力,同時根除對向誤煞。
        if self.dynamic_safety:
            from carla_rl.wrappers.lead_vehicle import path_aware_lead, world_forward_lead
            ego_speed = float(self._last_obs[3])
            leads = []
            gp, ls = path_aware_lead(np.asarray(self._last_obs, dtype=np.float64)[:307])
            if gp > 0.0:
                leads.append((gp, ego_speed - ls))
            gw, lw = world_forward_lead(self._world_vehicles_ego_frame())
            if gw > 0.0:
                leads.append((gw, ego_speed - lw))
            if leads:
                gap, closing = min(leads, key=lambda t: t[0])   # 最近的前車
                dsafe_gap = gap
                dsafe_d = dynamic_safe_gap(ego_speed)            # 當前動態安全距離 d_safe
                before_throttle, before_brake = float(throttle), float(brake)
                throttle, brake = graduated_brake(gap, ego_speed, closing, throttle, brake)
                dsafe_active = (abs(float(throttle) - before_throttle) > 1e-6
                                or abs(float(brake) - before_brake) > 1e-6)
        # 最終控制只能有一個踏板:安全層(DSAFE/護盾/planner)要求煞車時,煞車優先、清零油門。
        if brake > 0.0:
            throttle = 0.0
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))
        full = np.array([throttle, steer, brake], dtype=np.float32)
        obs, reward, done, info = self.env.step(full)
        self._last_obs = obs
        # 用「底層真正執行」的動作(env 互斥投影後)算 applied_pedal/steer,確保 buffer 標籤
        # = 真正造成 next_obs 的動作(env_applied_action 由 CarlaGymEnv 放入 info)。
        actual = np.asarray(info.get('env_applied_action', full), dtype=np.float32).reshape(-1)
        applied_pedal = float(actual[0] - actual[2])
        applied_steer = float(actual[1])
        info['applied_pedal'] = applied_pedal
        info['applied_steer'] = applied_steer
        # 動作鏈診斷(供逐步 trace 區分:策略原始 → 投影+EMA → 安全層覆寫後實際執行):
        info['raw_policy_throttle'] = float(raw_policy_throttle)  # SAC actor 原始輸出
        info['raw_policy_brake'] = float(raw_policy_brake)
        info['policy_throttle'] = float(policy_throttle)          # 踏板投影+EMA 後、未進安全層
        info['policy_brake'] = float(policy_brake)
        info['final_throttle'] = float(actual[0])                 # 安全層+env 互斥後真正送入 CARLA
        info['final_brake'] = float(actual[2])
        info['ttc_shield_fired'] = bool(ttc_shield_fired)
        info['dsafe_active'] = bool(dsafe_active)
        # 額外(trace 用):護盾看到的風險 TTC、DSAFE 的 d_safe/gap;shield_fired 為相容別名
        info['shield_fired'] = bool(ttc_shield_fired)
        info['shield_ttc'] = float(shield_ttc)
        info['dsafe_d'] = float(dsafe_d)
        info['dsafe_gap'] = float(dsafe_gap)
        if self.signed_pedal:
            # 一維模式:buffer 只存「實際 applied_pedal」(經安全層+env 互斥後),不存三維、不存 raw
            info['applied_action'] = np.array([applied_pedal], dtype=np.float32)
        else:
            info['applied_action'] = actual.copy()
        info['pp_steer'] = steer
        return obs, reward, done, info
