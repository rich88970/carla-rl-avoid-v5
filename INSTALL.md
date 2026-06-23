# 安裝與重現指南(INSTALL)

本文件補齊 clean-clone 後實際跑起來所需的步驟。整體相依鏈:
**CARLA 0.9.15 伺服器 → carla python wheel(版本須一致)→ EasyCarla-RL(base env)→ 本專案 `carla_rl/`**。

> ⚠️ 範圍提醒:最終成果為 **Town03 / 50 車 / 紅綠燈固定綠燈(traffic off)/ 無行人(walkers=0)** 下的
> 「SAC + Pure-Pursuit + DSAFE」**整體系統**成績,success 指「安全存活到時間上限」而非到達導航終點
> (route_completion = -1)。細節見 `README.md` 與 `docs/PROCESS.md`。

---

## 0. 前置需求
- Windows 10/11、NVIDIA GPU(本機在 driver 596.36 上以**源碼建置**的 CARLA 0.9.15 運行;官方預編譯包在此機 shader 編譯致命錯誤,故用源碼建置)。
- **CARLA 0.9.15 伺服器**(本機為源碼建置,引擎 `UE4_Carla`、專案 `carla-src/carla-0.9.15`)。
- Python 3.10、Git、[Git LFS](https://git-lfs.com)。

## 1. 取得儲存庫 + Git LFS(checkpoint 是 LFS 物件)
```bash
git clone https://github.com/rich88970/carla-rl-avoid-v5
cd carla-rl-avoid-v5
git lfs install
git lfs pull        # 拉下 checkpoints/*.pth 真實內容
git lfs fsck        # 驗證物件完整;若失敗則只拿到數十位元組的 pointer
```

## 2. 建立 venv 並安裝相依
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# PyTorch 需用 CUDA 索引(requirements.txt 的 torch 行有說明):
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
> `carla==0.9.15` wheel 必須與**執行中的伺服器版本一致**,否則連線會失敗。

## 3. 安裝 EasyCarla-RL(base 環境,提供 307 維觀測與內建 reward)
`carla_rl/wrappers/gym_compat.py` 以 `from easycarla.envs import CarlaEnv` 取得基礎環境,故必須先裝它。
固定在本專案驗證過的 commit,並套用兩個本機修補(否則本機伺服器會間歇當機):
```bash
git clone https://github.com/silverwingsbot/EasyCarla-RL
cd EasyCarla-RL
git checkout fc1bcfe6d63c9d999837c8e1b7c2cfa092e1c640
```
套用本儲存庫提供的修補檔(免手動改檔):
```bash
git apply ..\carla-rl-avoid-v5\patches\easycarla_local.patch
git diff --check     # 確認 patch 無格式錯誤
```
patch 內容:(1) client timeout 10s→120s(源碼建置首次載入地圖較慢);(2) 重用已載入的 world、
改掉 `load_world`(該 reload 在源碼建置會間歇性 SkeletalMesh 當機)。接著 editable 安裝:
```bash
pip install -e .
cd ..
```

## 4. 設定伺服器路徑(環境變數,**必填**)
`carla_rl/utils/server.py`、`carla_rl/scripts/start_carla.ps1`、`carla_rl/wrappers/route_planner.py`
**不保留任何本機絕對路徑**,一律從環境變數讀取(缺少會明確報錯):
```powershell
setx CARLA_UE4_EDITOR "C:\path\to\Engine\Binaries\Win64\UE4Editor.exe"
setx CARLA_UPROJECT   "C:\path\to\carla-0.9.15\Unreal\CarlaUE4\CarlaUE4.uproject"
# 只有用到全域路徑規劃(--use-route,非最終避撞所需)才需要:
setx CARLA_PYTHONAPI  "C:\path\to\carla-0.9.15\PythonAPI\carla"
```

## 5. 不需伺服器的離線驗收(import / 單元測試 / checkpoint)
```bash
python -m compileall carla_rl
python -c "import carla_rl"
python -c "from carla_rl.wrappers import CarlaGymEnv"
python -c "from carla_rl.evaluation import evaluate"
python -c "from carla_rl.agents.sac import SAC"
python -m carla_rl.scripts.test_world_forward_lead
python -c "import torch; x=torch.load('checkpoints/sac_avoid_v5.pth', map_location='cpu', weights_only=False); print(x['obs_dim'], x['hidden'])"
```
預期:皆無錯誤,單元測試印出 `... PASS`,checkpoint 印出 `327 [512, 512]`。
(以上在本專案開發機已實測通過。)

## 6. 啟動伺服器 + 跑最終避撞評估
```powershell
.\carla_rl\scripts\start_carla.ps1     # 啟動 CARLA(3–5 分鐘)
$env:CARLA_DSAFE = "1"
$env:CARLA_DSAFE_D0 = "7"
$env:CARLA_TTC_SHIELD = "0.4"
python -m carla_rl.scripts.run_eval --policy sac `
  --checkpoint checkpoints/sac_avoid_v5.pth --reward-preset avoid_v5 `
  --episodes 12 --vehicles 50 --max-steps 2000 --town Town03 --traffic off `
  --lateral-control --video 2 --out carla_rl/logs/eval_avoid
```

## 7. 訓練資料(只有從頭重訓才需要;最終評估不需要)
BC 暖啟與 SAC replay prefill 預期的資料位置:
```
data/easycarla_offline_dataset.hdf5
```
此檔**不隨 Git 儲存庫發布**。未取得資料集仍可載入 `checkpoints/sac_avoid_v5.pth` 執行最終評估
(第 6 節),但無法從頭重建 BC 暖啟與 SAC 訓練。
- 來源:EasyCarla-RL 官方離線資料集(下載連結見上游 repo README)。
- 原始專案/授權:EasyCarla-RL,Apache-2.0(見 `THIRD_PARTY_NOTICES.md`)。
- 預期檔名 / 位置:`easycarla_offline_dataset.hdf5`,放於 `data/`。
- 大小:2,762,730,173 bytes(約 2.76 GB)。
- SHA-256:`eef2fcab3b377872315bb5e5375b453e5daf6da9f21d474ba31d56826ed77651`

## 已知限制(重現相關)
- 12 回合樣本偏少;baseline 與 final 的回合/步數不完全一致,非統計顯著消融。
- 最終 checkpoint 仍是三維 actor,執行時 steer 被 Pure-Pursuit 覆寫(動作分布不一致,屬技術限制)。
- 紅綠燈與行人未正式評估(traffic off、walkers=0)。
- 行人生成在本機源碼版會 SkeletalMesh abort,故 `number_of_walkers` 固定為 0。
