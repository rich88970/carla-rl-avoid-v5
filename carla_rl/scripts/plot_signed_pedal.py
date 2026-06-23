"""畫 1D signed-pedal 訓練 0→80k 的數據圖表(sac_pedal_v3 0-40k + sac_pedal_v4 40-80k)。
輸出 carla_rl/logs/signed_pedal_0_80k.png。

Run: .\.venv\Scripts\python.exe -m carla_rl.scripts.plot_signed_pedal
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    return list(csv.DictReader(open(path)))


rows = []
for p in ['carla_rl/logs/sac_pedal_v3/train_log.csv',
          'carla_rl/logs/sac_pedal_v4/train_log.csv']:
    rows += load(p)
rows.sort(key=lambda x: int(x['step']))

step = np.array([int(x['step']) for x in rows])
def col(k, d=0.0):
    return np.array([float(x.get(k, d) or d) for x in rows])

mean_speed = col('mean_speed')
ep_steps = col('ep_steps')
red = col('red_light_violation_count')
coll = col('collision_count')
critic = col('critic_loss')
raw_p = col('raw_pedal_mean')
app_p = col('applied_pedal_mean')
ep_ret = col('ep_return')
ep_env = col('ep_env_return')

fig, ax = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle('1D signed-pedal SAC training 0→80k (sac_pedal_v3 0-40k + v4 40-80k)', fontsize=13)

# 0) reward(shaped vs env)
a = ax[0, 0]
a.plot(step, ep_ret, '-o', ms=3, color='#8c564b', label='ep_return (shaped)')
a.plot(step, ep_env, '-', color='#17becf', alpha=0.7, label='ep_env_return (env)')
a.axvline(40000, ls=':', color='gray', alpha=0.7)
a.set_title('Episode reward (shaped vs env)'); a.set_xlabel('env step'); a.set_ylabel('return'); a.legend(fontsize=8); a.grid(alpha=0.3)

# 1) 平均速度 + 自由速度目標線
a = ax[0, 1]
a.plot(step, mean_speed, '-o', ms=3, color='#1f77b4', label='episode mean speed')
a.axhline(6.0, ls='--', color='green', alpha=0.6, label='target avg 6 m/s')
a.axvline(40000, ls=':', color='gray', alpha=0.7)
a.set_title('Mean speed per episode'); a.set_xlabel('env step'); a.set_ylabel('m/s'); a.legend(fontsize=8); a.grid(alpha=0.3)

# 2) 紅燈違規 + 碰撞(每集次數)
a = ax[0, 2]
a.bar(step, red, width=600, color='#d62728', label='red-light violations')
a.bar(step, coll, width=600, bottom=red, color='#ff7f0e', label='collisions')
a.axvline(40000, ls=':', color='gray', alpha=0.7)
a.set_title('Red-light violations & collisions per episode')
a.set_xlabel('env step'); a.set_ylabel('count'); a.legend(fontsize=8); a.grid(alpha=0.3)

# 3) critic loss(訓練穩定度)
a = ax[1, 0]
a.plot(step, critic, '-', color='#9467bd')
a.axvline(40000, ls=':', color='gray', alpha=0.7)
a.set_title('Critic loss (1D critic — low = easy to fit)')
a.set_xlabel('env step'); a.set_ylabel('loss'); a.grid(alpha=0.3)

# 4) 回合長度 + raw vs applied pedal 一致性(雙軸)
a = ax[1, 1]
a.plot(step, ep_steps, '-o', ms=3, color='#2ca02c', label='episode length (steps)')
a.axvline(40000, ls=':', color='gray', alpha=0.7)
a.set_title('Episode length + raw==applied pedal'); a.set_xlabel('env step')
a.set_ylabel('steps', color='#2ca02c'); a.grid(alpha=0.3)
a2 = a.twinx()
a2.plot(step, raw_p, '-', color='#1f77b4', alpha=0.5, label='raw pedal')
a2.plot(step, app_p, '--', color='red', alpha=0.7, label='applied pedal')
a2.set_ylabel('pedal (raw≈applied)', color='#1f77b4')
lines = a.get_lines() + a2.get_lines()
a.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='upper right')

ax[1, 2].axis('off')   # 第 6 格留空
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = 'carla_rl/logs/signed_pedal_0_80k.png'
plt.savefig(out, dpi=110)
print('saved', out)

# 文字摘要
def seg(lo, hi):
    m = (step >= lo) & (step < hi)
    return (m.sum(), red[m].sum(), coll[m].sum(), mean_speed[m].mean())
for lo, hi, name in [(0, 40000, 'v3 0-40k'), (40000, 80000, 'v4 40-80k')]:
    n, r, c, sp = seg(lo, hi)
    print('%s: %d eps, red=%d, coll=%d, mean_speed=%.2f, max|raw-app|=%.4f'
          % (name, n, r, c, sp, np.max(np.abs(raw_p[(step >= lo) & (step < hi)]
                                              - app_p[(step >= lo) & (step < hi)]))))
