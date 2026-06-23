"""讀 train_log.csv,對每個數值欄位各出一張 PNG(x=step)+ 一張總覽 dashboard。

RL 訓練曲線本身雜訊極大(單集回報會大幅上下震盪),所以每張圖都疊一條**移動平均**(粗線)
看趨勢,原始值畫成淡細線當背景。每個指標也標註「代表什麼」,讓非作者也看得懂。
跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.plot_training <run_dir>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # 無視窗後端(伺服器/背景可用)
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# 每個 train_log 欄位代表什麼(顯示在圖上 + dashboard,讓數據一目了然)
METRIC_DESC = {
    'ep_return': '單集總獎勵(含塑形);RL 雜訊大,看移動平均趨勢上升即健康',
    'ep_env_return': '單集環境原始獎勵(未經我們的塑形)',
    'ep_steps': '單集步數(越高=活越久;撞車/出界/卡住會提早結束,上限 1000)',
    'critic_loss': 'Critic(價值網路)損失;穩定有界=健康,持續暴增=發散',
    'q1_mean': 'Critic 對「狀態-動作價值」的平均估計;暴衝發散=critic 壞掉(V12 教訓,須監控)',
    'actor_loss': 'Actor(策略網路)損失(SAC 為 -Q+α·logπ,通常為負)',
    'alpha': 'SAC 熵溫度係數 α(自動調節探索 vs 利用)',
    'entropy': '策略熵(越高=越隨機探索;訓練後期應下降趨穩)',
    'restarts': 'CARLA server 崩潰自動重啟累計次數(0 = 全程穩定)',
    'wall_min': '已訓練的實際時間(分鐘)',
}


def rolling_mean(y, w):
    """簡單移動平均(視窗 w),邊緣用前綴平均補,長度與 y 相同。"""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    w = max(1, min(w, len(y)))
    out = np.empty_like(y)
    csum = np.cumsum(np.insert(y, 0, 0.0))
    for i in range(len(y)):
        a = max(0, i - w + 1)
        out[i] = (csum[i + 1] - csum[a]) / (i + 1 - a)
    return out


def load(run_dir):
    """讀 train_log.csv → (steps, {欄位: [值]});跳過非數值欄。"""
    rows = list(csv.DictReader(open(Path(run_dir) / 'train_log.csv')))
    steps = [float(r['step']) for r in rows]
    cols = {}
    for k in rows[0].keys():
        if k == 'step':
            continue
        try:
            cols[k] = [float(r[k]) for r in rows]
        except ValueError:
            pass
    return steps, cols


def _draw(ax, steps, v, k, win):
    ax.plot(steps, v, lw=0.8, color='#bbbbbb', label='原始值(每集)')
    ax.plot(steps, rolling_mean(v, win), lw=2.2, color='#d6604d',
            label=f'移動平均(視窗 {win} 集)')
    ax.set_title(k, fontsize=11, fontweight='bold')
    desc = METRIC_DESC.get(k)
    if desc:
        ax.set_xlabel(desc, fontsize=8, color='#444')   # 把「代表什麼」寫在 x 軸下方
    ax.grid(alpha=0.3)


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    out = run_dir / 'plots'
    out.mkdir(exist_ok=True)
    steps, cols = load(run_dir)
    win = max(3, min(15, len(steps) // 5))    # 視窗隨資料量自適應
    # 每個數值欄位一張圖(原始淡線 + 移動平均粗線)
    for k, v in cols.items():
        fig, ax = plt.subplots(figsize=(8, 4.4))
        _draw(ax, steps, v, k, win)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout(); fig.savefig(out / f'{k}.png', dpi=110); plt.close(fig)
    # 總覽 dashboard
    n = len(cols); ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.6 * nrow), squeeze=False)
    flat = list(axes.flat)
    for ax, (k, v) in zip(flat, cols.items()):
        _draw(ax, steps, v, k, win)
    flat[0].legend(fontsize=7, loc='best')
    for ax in flat[n:]:
        ax.axis('off')
    fig.suptitle(f'訓練指標總覽({run_dir.name}) — 粗線=移動平均(視窗 {win} 集)、淡線=每集原始值',
                 fontsize=12, y=1.005)
    fig.tight_layout(); fig.savefig(out / 'dashboard.png', dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {len(cols)} metric charts + dashboard to {out} (rolling-mean window={win})')


if __name__ == '__main__':
    main()
