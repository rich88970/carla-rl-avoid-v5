# SAC 避撞策略 `sac_avoid_v5` + 對向誤煞修正 — 完整過程(預訓練 → 最終結果)

> 本文件只涵蓋「避撞」這條主線(BC 預訓練 → SAC avoid_v1–v5 → DSAFE 安全控制器 → 對向誤煞修正
> → 多地圖泛化)。不含其他分支(cruise/V1–V14 早期實驗、紅綠燈 / signed-pedal 等)。
> 環境:CARLA 0.9.15(源碼建置)、Town03、50 車、`number_of_walkers=0`、lights off、dt=0.1。

---

## 0. 架構總覽
分層控制:**方向盤由 Pure-Pursuit 幾何控制器**(沿規劃 waypoints),**SAC 只學縱向**(油門/煞車)。
觀測 327 維(307 資料集相容 + 5 紅綠燈/路口 + 5 預測 + 6 風險特徵)。SAC: actor/critic 隱藏層
(512,512),**critic 加 LayerNorm + reward_scale 0.2**(穩住 Q,避免發散)。

---

## 1. 預訓練(BC warm start)
- 在官方離線資料集(約 1.1M steps、7,008 條軌跡、Town03)上做行為克隆,得 `bc_actor.pth`。
- 目的:給策略一個「會沿路開、會跟車」的起點,SAC 不必從零探索。
- SAC 微調時以 `expand_first_layer` 把 307 維 BC actor 零填充到 327 維,並沿用 BC 的觀測正規化器。

## 2. SAC 避撞微調(avoid_v1 → avoid_v5)
獎勵逐版演進(`configs/reward_config.py`,純函式 `survive_shaping` / `gap_deficit_penalty`):

| 版本 | 關鍵獎勵 | 動機 |
|---|---|---|
| avoid_v1/v2 | TTC 接近懲罰(min_ttc 低於門檻線性扣分) | 讓 RL 自學「逼近就減速」的梯度 |
| avoid_v3 | 5 扇區風險感知 + 加重 TTC | 看得到路口橫切來車 |
| **avoid_v4** | **距離缺口懲罰**(前車距 < 7m 按缺口扣分,隨速度淡出) | **根因修法:診斷出 94% 碰撞是「低速潛行頂上前車」,而 TTC 在近距低速會盲;改用距離型密集梯度逼策略「在前車後停住」** |
| **avoid_v5** | survive 存活獎勵(progress 至 v_target 10)+ 保留 v4 全部安全項 | 最終訓練策略;`sac_avoid_v5.pth` |

訓練設定:`--lateral-control --throttle-ema 0.4 --critic-layernorm --reward-scale 0.2`,
50 車、Town03、lights off。伺服器在 reset 會間歇性 SkeletalMesh abort → 用 `restart_server` +
evaluator 重試迴圈容錯。

### 2.1 獎勵組成:基底來自 easycarla,避撞 shaping 是本專案
RL 最佳化的 reward 是**兩層相加**,在 `wrappers/reward_shaping.py` 裡 `shaped = reward` 一行就是接力點
(先拿環境的基底 reward,再往上疊本專案的避撞項):

| 層 | 內容 | 來源 |
|---|---|---|
| **基底 reward** | 速度追蹤(往 `desired_speed`)、車道偏移、橫向加速度平順度、靜止不動懲罰、碰撞/出界 −100 | **easycarla**(`EasyCarla-RL/easycarla/envs/carla_env.py`,`avoid_v5` 以 `replace_speed_term=False` 完全不動它) |
| **避撞 shaping** | TTC 接近懲罰、**距離缺口懲罰**、survive 存活/前進、pedal jerk、steer jerk | **本專案**(`configs/reward_config.py` + `wrappers/reward_shaping.py`) |

> 關鍵:easycarla 的基底只獎勵「開快、別偏車道、別撞」,**不會**教車「逼近前車就先收油停住」。真正讓避撞學
> 起來的密集梯度(TTC 懲罰、距離缺口懲罰)是本專案這層加的——尤其 `avoid_gap` 的低速潛行根因修法不在
> easycarla 內。即:**reward 的地基是 easycarla 的,但避撞這條線之所以成立的那部分 reward 是本專案設計的。**

## 3. DSAFE 安全控制器(評估期、env 變數開啟)
純策略雖會避撞,但仍有殘留碰撞(低速潛行)與卡住。加上「統一安全控制器」(`lateral_control.py`)
作部署安全層,三機制:
1. **動態安全距離** d_safe = d0 + τ·v + v²/(2a):低速 d_safe 小(不對停住前車過早誤煞→降卡住),
   高速 d_safe 大(提早反應→降碰撞)。
2. **分級煞車**:侵入安全距離越深、煞越大(緊急內圈全力煞、舒適外圈分級),貼到 d_safe 後 hold。
3. **TTC 護盾**:讀 5 扇區風險最小 TTC,逼近過快直接全力煞(補抓較高速接近)。

開啟:`CARLA_DSAFE=1 CARLA_DSAFE_D0=7 CARLA_TTC_SHIELD=0.4`。
成效:碰撞 **0.64 → 0.12**、成功 **0.32 → 0.80**(25×3000 實測)。

## 4. 對向誤煞修正(最終關鍵修正)
**問題**(看影片發現):車會把**對向車道的來車**當危險而無謂煞車,壓低車速、甚至卡住。

**根因**:DSAFE 的前車來源之一是 env 的 `obs[7]/[8]` 前向探測箱(±2.5m 直箱、**無朝向過濾**),
轉彎時會掃進對向車道把對向車當前車(轉彎 40% 幻影)。

**修法**(`lead_vehicle.py` / `risk_features.py` / `lateral_control.py`):
- 新增 `world_forward_lead`:掃 40m 內**全部**車輛,加**朝向閘門**(cos(rel_yaw)>0.25 排除對向/橫切)
  + 車道閘門(|y|<1.8m),回最近**同向**前車 → 取代 obs[7]/[8]。
- 並把朝向閘門補進 `risk_features`(TTC 護盾用)與 `path_aware_lead`(獎勵用),根除對向污染。
- 8 項單元測試 `test_world_forward_lead.py` 全 PASS。

## 5. 最終結果(Town03, 50 車, lights off, 12×2000)
| 指標 | baseline(舊 DSAFE) | **+ 對向修正(最終)** |
|---|---|---|
| 碰撞 collision | 0.12 | **0.083** ↓ |
| 成功 success | 0.80 | **0.92** ↑ |
| 卡住 stuck | 0.08 | **0.00** ↓ |
| 自由速度 free speed | 6.1 m/s | **6.30** ↑ |
| 車流中均速 avg speed | ~4.2 | **6.23** ↑ |
| 出界 off-road | — | **0.00** |

**每項皆優於 baseline**:對向誤煞會卡住車流且 stop-go 反增風險,移除後車流順暢、防撞力由
world_forward_lead(同向前車仍提早分級煞車)+ TTC 護盾維持。數據:`eval_oncomingfix2_town03`。

## 6. 多地圖泛化(只在 Town03 訓練)
| 城鎮 | 碰撞 | 成功 | 備註 |
|---|---|---|---|
| Town05(多線高速) | 0.00 | 1.00 | 泛化極佳 |
| Town10HD(密集都市) | 0.00 | 0.83 | 安全但窄路口偏慢 |
| Town04(環狀高速) | 0.25 | 0.75 | 高速順、唯一碰撞為出生即撞 |
| Town02(窄街市郊) | 0.50 | 0.50 | 街道比訓練窄,最差 |
出界全為 0(Pure-Pursuit 轉向與地圖無關);失敗皆為與車輛相撞。

## 6.5 車流密度泛化(100 / 150 車)— 踏板投影 bug 的診斷與修正
初次把最終配置放到 Town03 **100 / 150 車**(headline 是較寬鬆的 50 車),結果車**原地僵住、0 m/s**。
逐步動作鏈追蹤(`diag_pedal_trace`,16 欄:raw_policy → policy → final + 護盾/DSAFE 狀態)精準定位根因。

**根因 = 3D 踏板投影介面 bug,不是策略能力、也不是安全層。**
- 策略其實**狂踩油門**(`raw_policy_throttle` ~0.9),但 Gaussian actor 同時輸出**微量煞車**
  (~0.05–0.3,幾乎不會剛好等於 0)。
- 舊 `HybridSteerWrapper` 規則 `if brake > 0: throttle = 0` 把這個微量煞車**放大成全斷油**
  → `final_throttle = 0` → 車不動 → ~12 秒後被判 stuck。
- 三條件對照(100 車、同出生點、headless)排除其他可能:**autopilot 同出生點能跑 5.63 m/s**
  (排除環境/spawn);**DSAFE 關掉仍不動、護盾/DSAFE 介入比例 `frac=0.000`**(排除安全層)。

**修正**:新增純函式 `project_3d_pedals()`——依大小互斥(throttle ≥ brake 保留油門、平手偏油門),與
`CarlaGymEnv.step()` 底層規則一致,取代「brake>0 即斷油」。signed-pedal 一維模式不受影響。
回歸測試 `scripts/test_pedal_projection.py`;smoke(DSAFE off、50 車)直接驗證 `final_throttle` 真正送出、
車 **0 → 6.27 m/s**。

**修正後(全密度能開、零卡住):**

| Town03 密度 | 成功 | 碰撞 | 卡住 | 均速 |
|---|---|---|---|---|
| 50 車 | 0.75 | 0.25 | 0.00 | 5.95 m/s |
| 100 車 | 0.83 | 0.17 | 0.00 | 4.23 m/s |
| 150 車 | 0.83 | 0.17 | 0.00 | 3.10 m/s |

速度隨密度優雅遞減(6.0→4.2→3.1),碰撞穩定 ~0.17–0.25。

> ⚠️ **與第 5 節 headline 的關係**:0.083 / 0.92 @ 50 車 是**修正前**的成績——那條 `brake>0→斷油`
> 規則等於**過度煞車**(50 車更安全 0.083,但在 100+ 車直接癱瘓)。修正成正確的踏板投影後,50 車碰撞
> 升為 0.25,但換得**全密度可行駛、零卡住**。**舊的「100/150 原地僵住」是控制介面 bug,不代表模型
> 能力**(spec §十)。
> 方法註:高密度逐回合 churn 會觸發源碼版伺服器 SkeletalMesh 當機,密度測試一律走 `-nullrhi` headless。
> 數據:`data/eval_fix_{50,100,150}`(修正後)、`eval_dens100/150`(修正前僵住)、`eval_auto100`
> (autopilot 對照)、`data/pedal_trace_fixed/*.csv`(逐步動作鏈)。

## 7. 速度天花板(結構性)
開闊路自由速度 ~6 m/s 為**策略結構性天花板**:用更緊的 DSAFE 實驗,碰撞 0.12→0.20(+67%)卻只換
+0.2 速度;加大 progress 獎勵 4× 也只 +0.2。→ 安全與速度在此設定下的取捨已達實務極限。

## 8. 影片(`videos/`)
**預訓練(BC)起點 —— 兩種失敗模式(說明為何需要 Pure-Pursuit + SAC):**
- `pretrain_BC_raw_offroad_a.mp4` / `_b.mp4` — **原始 BC**(自己控方向):方向盤抖動,2–5 秒就衝出路面。
- `pretrain_BC_purepursuit_stuck_a.mp4` / `_b.mp4` — **BC 縱向 + Pure-Pursuit 轉向**:方向穩了,但 BC 縱向
  幾乎不給油 → 車不動、12 秒卡住終止。
- 結論:**BC 只給權重初始化,本身不會開車**;會開是 SAC 微調 + 避撞獎勵教出來的。

**過渡(context):SAC 自控方向 → 方向震盪(為何改 Pure-Pursuit):**
- `context_SAC_ownsteer_weave_a.mp4` / `_b.mp4` — 早期 SAC 控 3D(含方向),方向盤明顯 weave 震盪。
  這是引入 Pure-Pursuit 幾何轉向(讓 SAC 只管縱向)的動機。(屬早期分支,僅作對照。)

**最終策略:**
- `final_oncomingfix_ep0.mp4` / `_ep1.mp4` — **avoid_v5 + 對向修正 + DSAFE**,collision 0.083 的實際行駛。
- `intermediate_naive_removal_ep0.mp4` / `_ep1.mp4` — 中間版(直接拿掉 obs[7]/[8] 但未加
  world_forward_lead),碰撞反升,說明為何需要「外科式」修正。

**基準參考:**
- `baseline_autopilot_ep0.mp4` — autopilot 上限參考。

> 註:預訓練 BC 在此環境兩種模式都無法正常行駛(原始→出界、+幾何轉向→不動),屬誠實呈現;
> 這正是本專案要做 SAC 縱向微調 + DSAFE 的動機。

## 重現指令
```
set "CARLA_DSAFE=1" & set "CARLA_DSAFE_D0=7" & set "CARLA_TTC_SHIELD=0.4"
.\.venv\Scripts\python.exe -u -m carla_rl.scripts.run_eval --policy sac ^
  --checkpoint carla_rl/checkpoints/sac_avoid_v5.pth --reward-preset avoid_v5 ^
  --episodes 12 --vehicles 50 --max-steps 2000 --town Town03 --traffic off ^
  --lateral-control --video 2 --out carla_rl/logs/eval_avoid
```
