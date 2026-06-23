"""Smoke test for the 1D signed-pedal SAC refactor (no CARLA server needed).

Verifies (spec point 12 + 第三輪審查 E):
  - SAC act_dim=1: action shape (1,), target_entropy == -1, bounds [-1, 1].
  - critic input = obs+1; ReplayBuffer act_dim=1.
  - convert_3d_actor_to_1d on a REAL checkpoint(自動讀 obs_dim/hidden)。
  - u<0 -> brake 投影。
  - 紅 vs 綠探針:同一觀測切換 red/green,印 u_red、u_green。
    註:u_red < u_green 是「訓練後」應達成的目標;此處來源 checkpoint(light_v2)實測紅燈反應很弱,
    故為「報告」而非硬性 assert(避免對未學會守燈的 checkpoint 誤判失敗)。

Run:
    .\.venv\Scripts\python.exe -m carla_rl.scripts.test_signed_pedal
"""

import os
import sys

import numpy as np
import torch

from carla_rl.agents.sac import SAC, ReplayBuffer
from carla_rl.models import convert_3d_actor_to_1d

# signed-pedal 為範圍外分支:其原始 checkpoint(light_v2)未隨此 avoid_v5 交付包含。clean-clone 時
# 退而用交付的 avoid_v5(同為 3D actor)驗證 3D→1D 轉換;都找不到才 SKIP(離 0 退出,不算失敗)。
_CANDIDATES = [
    'carla_rl/checkpoints/sac_avoid_light_v2.pth',
    'checkpoints/sac_avoid_light_v2.pth',
    'checkpoints/sac_avoid_v5.pth',
    'carla_rl/checkpoints/sac_avoid_v5.pth',
]
SRC_CKPT = next((p for p in _CANDIDATES if os.path.exists(p)), None)
if SRC_CKPT is None:
    print('SKIP  test_signed_pedal: 找不到任何 3D 來源 checkpoint(signed-pedal 為範圍外分支)')
    sys.exit(0)

# 自動讀來源 checkpoint 的 obs_dim / hidden(審查 E:不要寫死)
_c = torch.load(SRC_CKPT, map_location='cpu', weights_only=False)
OBS_DIM = int(_c['obs_dim'])
HIDDEN = tuple(_c['hidden'])
print(f'source={SRC_CKPT}  obs_dim={OBS_DIM}  hidden={HIDDEN}')


def pedal_to_throttle_brake(u):
    return max(u, 0.0), max(-u, 0.0)


# 1) SAC act_dim=1
agent = SAC(OBS_DIM, act_dim=1, hidden=HIDDEN, action_low=[-1.0], action_high=[1.0])
assert abs(agent.target_entropy - (-1.0)) < 1e-9, f'target_entropy={agent.target_entropy}'
assert np.allclose(agent.actor.act_scale.cpu().numpy(), [1.0])
assert np.allclose(agent.actor.act_bias.cpu().numpy(), [0.0])
a = agent.act(np.zeros(OBS_DIM, dtype=np.float32))
assert a.shape == (1,) and -1.0 <= float(a[0]) <= 1.0
print('PASS  SAC act_dim=1: action shape (1,), target_entropy -1, bounds [-1,1]')

# 2) critic input = obs+1
q1, q2 = agent.critic(torch.zeros(2, OBS_DIM, device=agent.device),
                      torch.zeros(2, 1, device=agent.device))
assert q1.shape == (2, 1)
print('PASS  critic input obs_dim+1, output (B,1)')

# 3) ReplayBuffer act_dim=1
buf = ReplayBuffer(100, OBS_DIM, act_dim=1)
buf.add(np.zeros(OBS_DIM, np.float32), np.array([-0.7], np.float32), 0.0,
        np.zeros(OBS_DIM, np.float32), 0.0)
assert buf.act.shape == (100, 1) and abs(buf.act[0, 0] + 0.7) < 1e-6
print('PASS  ReplayBuffer act_dim=1: stored action shape (1,)')

# 4) convert a real checkpoint -> 1D actor
conv = convert_3d_actor_to_1d(_c['actor'], OBS_DIM, HIDDEN)
out = conv.mean_action(torch.zeros(1, OBS_DIM))
assert out.shape == (1, 1)
print('PASS  convert_3d_actor_to_1d: real checkpoint -> 1D actor, output (1,1)')

# 4b) 一維 checkpoint 存→載一致(審查 E:save/load roundtrip)
import tempfile, os
agent.actor.load_state_dict(conv.state_dict())
tmp = os.path.join(tempfile.gettempdir(), 'sp_roundtrip.pth')
agent.save(tmp, {'signed_pedal': True})
a1 = agent.act(np.zeros(OBS_DIM, np.float32), deterministic=True)
agent2 = SAC(OBS_DIM, act_dim=1, hidden=HIDDEN, action_low=[-1.0], action_high=[1.0])
agent2.load(tmp)
a2 = agent2.act(np.zeros(OBS_DIM, np.float32), deterministic=True)
assert np.allclose(a1, a2, atol=1e-6), f'roundtrip mismatch {a1} vs {a2}'
print('PASS  1D checkpoint save/load roundtrip: output 一致')

# 5) u<0 -> brake mapping
t_neg, b_neg = pedal_to_throttle_brake(-0.7)
t_pos, b_pos = pedal_to_throttle_brake(0.5)
assert t_neg == 0.0 and abs(b_neg - 0.7) < 1e-9
assert abs(t_pos - 0.5) < 1e-9 and b_pos == 0.0
print('PASS  signed-pedal projection: u<0 -> brake, u>0 -> throttle')

# 6) 紅 vs 綠探針(報告,非硬 assert)
def u_for_light(actor, color):
    o = np.zeros((1, OBS_DIM), dtype=np.float32)
    o[0, 307] = 1.0
    o[0, 311] = 0.1
    o[0, 308 if color == 'red' else 310] = 1.0
    with torch.no_grad():
        return float(actor.mean_action(torch.tensor(o))[0, 0])

u_red = u_for_light(conv, 'red')
u_green = u_for_light(conv, 'green')
verdict = 'OK(紅<綠,已偏煞)' if u_red < u_green else 'INFO(未學會守燈,訓練後應轉為紅<綠)'
print(f'[probe] u_red={u_red:+.4f}  u_green={u_green:+.4f}  delta={u_red - u_green:+.4f}  -> {verdict}')

print('\nALL SIGNED-PEDAL SMOKE TESTS PASS')
