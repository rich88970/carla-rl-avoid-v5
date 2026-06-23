# CHANGELOG —— SAC 避撞 `sac_avoid_v5` + 對向誤煞修正(本線專屬)

> 從完整 CHANGELOG 擷取、只保留這條主線(BC 預訓練 → SAC 自控方向/weave → Pure-Pursuit →
> avoid_v1–v5 → DSAFE → 對向修正 → 多地圖)。其他分支(cruise/紅綠燈/signed-pedal 等)不收。
> 時間由舊到新。

## 預訓練(BC warm start)
- 在官方離線資料集(約 1.1M steps、7,008 條軌跡、Town03)做行為克隆,得 `bc_actor.pth`,作為 SAC
  的暖啟初始化(SAC 微調時以 `expand_first_layer` 把 307 維零填充到 327 維,沿用 BC 觀測正規化器)。
- 基準對照:random / autopilot / 預訓練 DQL 跑過評估,確立指標上下限。

## SAC 自控方向 → 方向震盪(weave)→ 引入 Pure-Pursuit
- 早期 SAC 控完整三維 `[throttle, steer, brake]`(含方向)。**方向盤明顯 weave 震盪**(高頻左右擺),
  純 SAC 微調把平順的 BC 策略開成抖動。
- 抗震盪嘗試(動作平滑 / 低熵 / steer EMA)只能部分緩解。
- **定案:改分層架構 —— 方向盤交給幾何式 Pure-Pursuit 控制器(沿規劃 waypoints),SAC 只學縱向。**
  方向震盪歸零(零 weave),這成為之後所有避撞版本的固定底座。

## RL 避撞 avoid_v1 → avoid_v5(獎勵演進)
- **avoid_v1**:獎勵加 TTC 接近懲罰(`avoid_ttc_weight`),讓 RL 自學「逼近就減速」。長訓 ~262k
  收斂;但裸策略(無安全層)長程碰撞 **0.96** → 純 RL 避撞「失敗」。
- **avoid_v2/v3**:加重避撞 + 5 扇區風險感知 + 橫向閘門(修直線把對向車道誤判的假陽性);裸策略
  碰撞 0.96→0.56,仍未突破。
- **根因診斷**:把碰撞幾何儀器化 → **94% 是「低速潛行頂上前車」,非高速衝撞;根因 = TTC 型感知對
  近距低速盲**。
- **avoid_v4(根因修法)**:獎勵加「距離缺口懲罰」(`avoid_gap_weight=-8`,前車距 < 7m 按缺口扣分、
  隨速度淡出),逼 RL 在前車後停住而非潛行。
- **avoid_v5(最終策略)**:survive 存活獎勵(progress 至 `survive_v_target` 10)+ 保留 v4 全部安全項。
  critic 加 LayerNorm + `reward_scale 0.2`(壓低 progress、Q 有界無發散)。`sac_avoid_v5.pth`。

## DSAFE 統一安全控制器(評估期、env 變數開啟)
- 純策略仍有殘留碰撞(低速潛行)與卡住 → 加「統一安全控制器」作部署安全層,三機制:
  1. **動態安全距離** d_safe = d0 + τ·v + v²/(2a)(低速小→降卡住、高速大→降碰撞)。
  2. **分級煞車**(緊急內圈全力煞 + 舒適外圈分級,貼 d_safe 後 hold)。
  3. **TTC 護盾**(5 扇區風險最小 TTC,逼近過快全力煞)。
- 逐步調參(min_ttc 護盾 0.64→0.40;d0=7;三機制完整實現)→ **avoid_v5 + DSAFE(d0=7,τ=0.6)+ TTC
  護盾 = 碰撞 0.12、成功 0.80、卡住 0.08、零出界、零震盪**(env:`CARLA_DSAFE=1 CARLA_DSAFE_D0=7
  CARLA_TTC_SHIELD=0.4`)。碰撞 0.64→0.12(5.3×)、成功 0.32→0.80。
- 註:側向緊急閃避(AES)實測反使碰撞 0.12→0.28 → 還原。

## 速度天花板(結構性)
- avoid_v5 progress 獎勵 ×4 + target 10 重訓:free speed 僅 5.88→6.09(+0.2);更緊 DSAFE 換 +0.2 速度
  卻 +67% 碰撞。**→ 開闊路 ~6 m/s 為結構性天花板**(5 扇區風險對遠車保守 + Pure-Pursuit 彎道減速 +
  reward_scale 0.2),非調獎勵可破。速度探索定案。

## 對向誤煞修正(★ 最終,全面優於 baseline)
- 看影片發現:車把**對向車道來車**當危險而無謂煞車(壓速、卡住)。根因 = DSAFE 用 env `obs[7]/[8]`
  前向探測箱(±2.5m 直箱、無朝向過濾),轉彎掃進對向車道(轉彎 40% 幻影)。
- 修法:新增 `world_forward_lead`(掃 40m 全車輛 + 朝向閘門 cos>0.25 排除對向/橫切 + 車道閘門 |y|<1.8m,
  回最近同向前車)取代 obs[7]/[8];並把朝向閘門補進 `risk_features` 與 `path_aware_lead`。8 項單元測試 PASS。
- **結果(Town03, 50 車, lights off):碰撞 0.12→0.083、成功 0.80→0.92、卡住 0.08→0.00、free 6.1→6.30、
  avg →6.23、出界 0** —— 每項皆改善。

## 多地圖泛化(只在 Town03 訓練)
- Town05(多線高速)碰撞 0.00 / 成功 1.00;Town10HD(密集都市)0.00 / 0.83(窄路口偏慢);
  Town04(環狀高速)0.25 / 0.75(唯一碰撞為出生即撞);Town02(窄街市郊)0.50 / 0.50(最差)。
- 出界全為 0(Pure-Pursuit 與地圖無關);失敗皆為與車輛相撞。
