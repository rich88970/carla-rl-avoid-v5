"""RL-避撞各配置對照長條圖(碰撞/成功/卡住)。資料為 2026-06-19 各 25×3000 eval 實測。
跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.plot_avoid_comparison"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (label, collision, success, stuck) — 25 集 × 3000 步,50 車,Town03,desired 8
ROWS = [
    ('avoid_v3\n裸策略', 0.64, 0.32, 0.04),
    ('avoid_v4\n裸策略', 0.56, 0.32, 0.12),
    ('v4+距離煞車', 0.28, 0.64, 0.08),
    ('v4+雙backstop', 0.16, 0.64, 0.20),
    ('v4+DSAFE\nd0=7', 0.28, 0.68, 0.04),
    ('v4+DSAFE+護盾\n(最終)', 0.12, 0.80, 0.08),
]
labels = [r[0] for r in ROWS]
collision = [r[1] for r in ROWS]
success = [r[2] for r in ROWS]
stuck = [r[3] for r in ROWS]

x = np.arange(len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - w, collision, w, label='碰撞率 (↓)', color='#d6604d')
b2 = ax.bar(x, success, w, label='成功率 (↑)', color='#4393c3')
b3 = ax.bar(x + w, stuck, w, label='卡住率 (↓)', color='#f4a261')
for bars in (b1, b2, b3):
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
ax.axhline(0.80, ls='--', c='#4393c3', alpha=0.5)
ax.text(len(labels) - 0.5, 0.81, 'autopilot 成功率天花板 0.80', ha='right', fontsize=8, color='#4393c3')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('比率'); ax.set_ylim(0, 1.0)
ax.set_title('RL-避撞:各配置對照(碰撞 0.64→0.12、成功 0.32→0.80)')
ax.legend(loc='upper center', ncol=3)
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass
plt.tight_layout()
out = 'carla_rl/logs/avoid_v4_config_comparison.png'
plt.savefig(out, dpi=130)
print(f'wrote {out}')
