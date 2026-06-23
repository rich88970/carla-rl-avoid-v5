"""Networks shared by BC and SAC.

Action bounds follow the env's action space: [throttle, steer, brake] in
[0,1] x [-1,1] x [0,1]. The actor outputs a squashed Gaussian rescaled into
those bounds; BC trains the mean, SAC trains mean + std.

Observations are normalized with dataset statistics (global x/y coordinates
are O(100), lidar is [0,1] — without normalization BC barely learns). The
normalizer covers the first `fit_dim` dims (307 dataset dims); any appended
extension features pass through untouched (they are already ~[0,1]).
"""

import numpy as np
import torch
import torch.nn as nn

ACTION_LOW = np.array([0.0, -1.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class ObsNormalizer:
    # Floor keeps near-constant dataset dims from exploding live inputs: lane
    # width has std 0.001 in the data (always ~3.5 m), so a live 3.0 m lane
    # would normalize to -500 and saturate the network.
    STD_FLOOR = 0.1

    def __init__(self, mean, std, fit_dim):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), self.STD_FLOOR)
        self.fit_dim = fit_dim

    @classmethod
    def fit(cls, obs_array):
        return cls(obs_array.mean(axis=0), obs_array.std(axis=0), obs_array.shape[1])

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32).copy()
        if obs.ndim == 1:
            obs[: self.fit_dim] = (obs[: self.fit_dim] - self.mean) / self.std
        else:
            obs[:, : self.fit_dim] = (obs[:, : self.fit_dim] - self.mean) / self.std
        return obs

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'fit_dim': self.fit_dim}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d['mean'], d['std'], int(d['fit_dim']))


def mlp(in_dim, hidden, out_dim, layernorm=False):
    # LayerNorm after each hidden Linear is the standard cure for SAC critic
    # Q-value divergence (the V6 critic_loss climbed 90 -> 2000+ without it).
    # Applied to the critic only — the actor is BC-warm-started and adding LN
    # there would break those weights.
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim=3, hidden=(256, 256), low=None, high=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.body = mlp(obs_dim, hidden, 2 * act_dim)
        # 預設用 3D 動作邊界 [throttle,steer,brake];1D signed-pedal 則傳 low=[-1] high=[1]。
        low = torch.as_tensor(ACTION_LOW if low is None else np.asarray(low, np.float32))
        high = torch.as_tensor(ACTION_HIGH if high is None else np.asarray(high, np.float32))
        self.register_buffer('act_scale', (high - low) / 2.0)
        self.register_buffer('act_bias', (high + low) / 2.0)

    def forward(self, obs):
        mu, log_std = self.body(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def _squash(self, u):
        return torch.tanh(u) * self.act_scale + self.act_bias

    def mean_action(self, obs):
        mu, _ = self(obs)
        return self._squash(mu)

    def sample(self, obs):
        mu, log_std = self(obs)
        dist = torch.distributions.Normal(mu, log_std.exp())
        u = dist.rsample()
        action = self._squash(u)
        log_prob = dist.log_prob(u) - torch.log(
            self.act_scale * (1.0 - torch.tanh(u) ** 2) + 1e-6
        )
        return action, log_prob.sum(-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim=3, hidden=(256, 256), layernorm=False):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden, 1, layernorm=layernorm)
        self.q2 = mlp(obs_dim + act_dim, hidden, 1, layernorm=layernorm)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


def convert_3d_actor_to_1d(actor_3d_state, obs_dim, hidden):
    """把舊三維 actor [throttle,steer,brake] 轉成一維 signed-pedal actor u∈[-1,1]。

    共享特徵層(body.0 / body.2)直接複製;輸出頭(body.4)由 6 維(3 mean + 3 log_std)
    壓成 2 維(1 mean + 1 log_std):
      - 新 mean   = 0.5 * (throttle_mean   - brake_mean)      → 油門↑/煞車↑ 折成 signed pedal
      - 新 log_std= 0.5 * (throttle_log_std + brake_log_std)  → 縱向探索噪音平均
    steer 那一列(index 1 / 4)丟棄(方向盤由 Pure Pursuit 控制)。回傳一個已載入權重的
    1D GaussianActor(邊界 [-1,1])。obs_dim/hidden 須與舊 actor 相同。
    """
    new_actor = GaussianActor(obs_dim, act_dim=1, hidden=tuple(hidden),
                              low=[-1.0], high=[1.0])
    new_state = new_actor.state_dict()
    # 共享特徵層逐一複製(除最後輸出層 body.{last})
    last = max(int(k.split('.')[1]) for k in new_state if k.startswith('body.'))
    for k in new_state:
        if k.startswith('body.') and int(k.split('.')[1]) != last:
            if k in actor_3d_state and actor_3d_state[k].shape == new_state[k].shape:
                new_state[k] = actor_3d_state[k].clone()
    ow = actor_3d_state[f'body.{last}.weight']   # (6, H):0-2 mean(t,s,b) 3-5 log_std(t,s,b)
    ob = actor_3d_state[f'body.{last}.bias']
    nw = torch.zeros_like(new_state[f'body.{last}.weight'])  # (2, H)
    nb = torch.zeros_like(new_state[f'body.{last}.bias'])    # (2,)
    nw[0] = 0.5 * (ow[0] - ow[2])      # pedal mean = 0.5(throttle - brake)
    nw[1] = 0.5 * (ow[3] + ow[5])      # pedal log_std = 0.5(thr + brk log_std)
    nb[0] = 0.5 * (ob[0] - ob[2])
    nb[1] = 0.5 * (ob[3] + ob[5])
    new_state[f'body.{last}.weight'] = nw
    new_state[f'body.{last}.bias'] = nb
    new_actor.load_state_dict(new_state)
    return new_actor


def expand_first_layer(state_dict, old_dim, new_dim, prefix='body.0'):
    """Widen an actor's input layer from old_dim to new_dim inputs, zero-filling
    the new columns — the network's behavior on the original inputs is
    unchanged until training grows the new weights. Used to carry a BC policy
    trained on the 307-dim dataset into the live 312-dim observation."""
    key = f'{prefix}.weight'
    w = state_dict[key]
    if w.shape[1] == new_dim:
        return state_dict
    assert w.shape[1] == old_dim, f'unexpected input dim {w.shape[1]}'
    expanded = torch.zeros(w.shape[0], new_dim, dtype=w.dtype)
    expanded[:, :old_dim] = w
    state_dict[key] = expanded
    return state_dict
