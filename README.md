# CARLA RL 避撞專案 (`sac_avoid_v5`)

這是 CARLA 0.9.15 自動駕駛避撞實驗的可執行交付版。專案基於
[EasyCarla-RL](https://github.com/silverwingsbot/EasyCarla-RL)，最終架構是:

- SAC 只負責縱向控制，也就是油門與煞車。
- 方向盤由 Pure-Pursuit 幾何控制器接管，避免 SAC 方向震盪。
- DSAFE 安全層在評估時用動態安全距離與 TTC 護盾補最後防線。
- 最終成果不包含紅綠燈與行人，正式設定是 Town03、50 台車、traffic off、walkers=0。

本 repo 是完整實驗目錄整理後的公開交付版。執行不依賴原本的本機絕對路徑；CARLA 與 UE4 的路徑一律用環境變數設定。

## 快速結果

最終 checkpoint: `checkpoints/sac_avoid_v5.pth`

正式評估設定: `Town03`、50 vehicles、lights off、12 episodes、`SAC + Pure-Pursuit + DSAFE`

| 指標 | 結果 |
|---|---:|
| collision | 0.083 |
| success | 0.92 |
| stuck | 0.00 |
| free speed | 6.30 m/s |
| avg speed | 6.23 m/s |
| off-road | 0.00 |

完整研發線請看 [`CHANGELOG_avoid_v5.md`](CHANGELOG_avoid_v5.md)，逐步實驗過程與取捨請看 [`docs/PROCESS.md`](docs/PROCESS.md)。

## 目錄重點

```text
.
├── README.md
├── INSTALL.md
├── requirements.txt
├── CHANGELOG_avoid_v5.md
├── docs/PROCESS.md
├── carla_rl/
│   ├── agents/              SAC agent
│   ├── configs/             reward/env presets
│   ├── evaluation/          evaluator、metrics、影片錄製
│   ├── models/              actor/critic networks
│   ├── scripts/             run_eval、train_sac、test_*、診斷工具
│   ├── utils/               CARLA server 啟動與等待
│   └── wrappers/            EasyCarla wrapper、Pure-Pursuit、DSAFE、risk features
├── checkpoints/
│   └── sac_avoid_v5.pth     Git LFS 追蹤的最終 checkpoint
└── data/                    已整理的訓練/評估 CSV 與 JSON 結果
```

`videos/`、大型離線資料集、replay buffer 不隨 repo 發布。展示影片只保留在本機，不進 GitHub。

## 1. 安裝

先準備:

- Windows 10/11
- Python 3.10
- Git + Git LFS
- CARLA 0.9.15 server
- EasyCarla-RL base environment

clone 後先拉 Git LFS checkpoint:

```powershell
git clone https://github.com/rich88970/carla-rl-avoid-v5
cd carla-rl-avoid-v5
git lfs install
git lfs pull
```

建立 Python 環境:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

EasyCarla-RL 與 CARLA 伺服器的完整安裝步驟在 [`INSTALL.md`](INSTALL.md)。最重要的是三個環境變數:

```powershell
setx CARLA_UE4_EDITOR "C:\path\to\Engine\Binaries\Win64\UE4Editor.exe"
setx CARLA_UPROJECT   "C:\path\to\carla-0.9.15\Unreal\CarlaUE4\CarlaUE4.uproject"
setx CARLA_PYTHONAPI  "C:\path\to\carla-0.9.15\PythonAPI\carla"
```

`CARLA_PYTHONAPI` 只有使用全域 route planning 時才需要；最終避撞評估通常不需要。

## 2. 離線驗收

這些檢查不需要啟動 CARLA server:

```powershell
python -m compileall carla_rl
python -c "import carla_rl"
python -c "from carla_rl.wrappers import CarlaGymEnv"
python -c "from carla_rl.evaluation import evaluate"
python -c "from carla_rl.agents.sac import SAC"
python -m carla_rl.scripts.test_world_forward_lead
python -m carla_rl.scripts.test_avoid_free_speed_reward
python -c "import torch; x=torch.load('checkpoints/sac_avoid_v5.pth', map_location='cpu', weights_only=False); print(x['obs_dim'], x['hidden'])"
```

checkpoint 檢查預期輸出:

```text
327 [512, 512]
```

## 3. 啟動 CARLA

如果環境變數已設定好，可以從 repo 根目錄啟動:

```powershell
.\carla_rl\scripts\start_carla.ps1
```

等 CARLA 啟動完成後再跑評估。第一次啟動或切地圖可能需要數分鐘。

## 4. 啟動最終模型

最終模型不是常駐服務，啟動方式是用 `run_eval` 載入 checkpoint 並連到正在執行的 CARLA server。
先完成第 3 節啟動 CARLA，再從 repo 根目錄執行:

```powershell
$env:CARLA_DSAFE = "1"
$env:CARLA_DSAFE_D0 = "7"
$env:CARLA_TTC_SHIELD = "0.4"

python -m carla_rl.scripts.run_eval --policy sac `
  --checkpoint checkpoints/sac_avoid_v5.pth `
  --reward-preset avoid_v5 `
  --episodes 12 `
  --vehicles 50 `
  --max-steps 2000 `
  --town Town03 `
  --traffic off `
  --lateral-control `
  --video 2 `
  --out carla_rl/logs/eval_avoid
```

這行會載入 `checkpoints/sac_avoid_v5.pth`，以 SAC 最終策略負責油門/煞車，方向盤由 Pure-Pursuit 接管，
DSAFE 則由三個環境變數開啟。輸出會寫到 `carla_rl/logs/eval_avoid/`。
如果只要指標、不錄影片，把 `--video 2` 改成 `--video 0`。

## 5. 進階:重新訓練或續訓

公開 repo 可直接載入最終 checkpoint 做評估，但沒有包含從頭重訓需要的全部大型資料:

- `data/easycarla_offline_dataset.hdf5` 不隨 repo 發布。
- replay buffer 不隨 repo 發布。
- BC 暖啟 checkpoint 不在公開交付內容中。

若你已補齊資料與 checkpoint，可用 `train_sac.py` 續訓或重跑實驗。例如從既有 SAC checkpoint 短續訓:

```powershell
python -m carla_rl.scripts.train_sac `
  --resume checkpoints/sac_avoid_v5.pth `
  --reward-preset avoid_v5 `
  --vehicles 50 `
  --traffic off `
  --lateral-control `
  --critic-layernorm `
  --reward-scale 0.2 `
  --total-steps 10000 `
  --run-name sac_avoid_v5_resume
```

如果缺少 replay buffer，`train_sac.py` 會拒絕高風險續訓。這是刻意的，因為先前實驗中 bufferless resume 會造成策略發散或退化。只想重現最終結果時，請使用第 4 節的評估指令。

## 6. 既有結果資料

已整理的評估結果在 `data/`:

- `data/eval_oncomingfix2_town03/`:最終 Town03 50 車結果。
- `data/eval_fix_50/`、`data/eval_fix_100/`、`data/eval_fix_150/`:修正後密度泛化。
- `data/eval_multimap_*`:多地圖泛化。
- `data/sac_avoid_v5/train_log.csv`:avoid_v5 訓練日誌。

## 授權與第三方

本專案部分程式衍生自 EasyCarla-RL。授權見 [`LICENSE`](LICENSE)，第三方歸屬與本機修補見
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 與 [`patches/easycarla_local.patch`](patches/easycarla_local.patch)。

## 注意事項

- 最終成果是分層系統，不是純 SAC 裸策略。
- `success` 代表安全存活到 episode 上限，不代表完成導航終點。
- 正式成果不涵蓋紅綠燈與行人。
- `checkpoints/*.pth` 由 Git LFS 管理；clone 後需 `git lfs pull`。
- 大型資料集、展示影片、replay buffer 不進 GitHub。
