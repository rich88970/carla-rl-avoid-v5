"""訓練收斂偵測:對 ep_return 取最近 N 集滾動平均,判斷平台 / 下降(使用者要求的自動早停)。
平台 = 距離歷史最佳已過 PLATEAU_FRAC*total 步且無新高;下降 = 滾動平均自最佳後持續低於
最佳*(1-DECLINE_DROP) 達 DECLINE_FRAC*total 步。
CLI: 讀 train_log.csv,印 'STOP <原因>' 並 exit 1(該停),或 'OK ...' exit 0。
    .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.plateau <run_dir> <total_steps>
"""
import csv
import sys
from pathlib import Path

import numpy as np

WIN = 30                # 滾動平均集數(平滑 RL 回報雜訊)
PLATEAU_FRAC = 0.10     # 無新高達 10% 總步數 → 平台(使用者 2026-06-18 調緊:20%→10%)
DECLINE_FRAC = 0.05     # 低於最佳達 5% 總步數 → 下降(使用者調緊:15%→5%)
DECLINE_DROP = 0.10     # 「低於最佳」的相對幅度(10%)


def detect(steps, returns, total_steps):
    """回傳 (stop: bool, reason: str)。steps/returns = 各 episode 的 step 與 ep_return。"""
    if len(returns) < WIN + 2:
        return False, 'warmup'
    r = np.asarray(returns, dtype=np.float64)
    s = np.asarray(steps, dtype=np.float64)
    ma = np.convolve(r, np.ones(WIN) / WIN, mode='valid')   # 滾動平均
    ma_step = s[WIN - 1:]                                    # 對齊每個 ma 的 step
    best_i = int(np.argmax(ma))
    best_ma, best_step = float(ma[best_i]), float(ma_step[best_i])
    cur_step = float(ma_step[-1])
    # 平台:距離最佳已過 20% 總步數(最佳不在最後 → 無新高)
    if cur_step - best_step >= PLATEAU_FRAC * total_steps:
        return True, (f'PLATEAU: no new best for {cur_step - best_step:.0f} steps '
                      f'(>={PLATEAU_FRAC * total_steps:.0f}); best_ma={best_ma:.1f}@{best_step:.0f}')
    # 下降:自最佳之後,滾動平均「最早」低於 best*(1-drop) 的 step 起,持續達 15% 總步數
    after = ma[best_i:]
    after_step = ma_step[best_i:]
    thresh = best_ma * (1.0 - DECLINE_DROP) if best_ma > 0 else best_ma * (1.0 + DECLINE_DROP)
    below = after < thresh
    if below.any():
        first_below = float(after_step[int(np.argmax(below))])
        if cur_step - first_below >= DECLINE_FRAC * total_steps:
            return True, (f'DECLINE: below best*{1 - DECLINE_DROP:.2f} for '
                          f'{cur_step - first_below:.0f} steps (>={DECLINE_FRAC * total_steps:.0f}); '
                          f'best_ma={best_ma:.1f}, cur_ma={float(ma[-1]):.1f}')
    return False, 'ok'


def main():
    run_dir = Path(sys.argv[1])
    total = int(sys.argv[2])
    rows = list(csv.DictReader(open(run_dir / 'train_log.csv')))
    steps = [float(r['step']) for r in rows]
    rets = [float(r['ep_return']) for r in rows]
    stop, reason = detect(steps, rets, total)
    print(('STOP ' if stop else 'OK ') + reason)
    sys.exit(1 if stop else 0)


if __name__ == '__main__':
    main()
