"""Soft Actor-Critic with automatic entropy tuning.

Built for fine-tuning a BC-initialized actor on the live 312-dim observation:
load_bc_checkpoint() widens the BC actor's 307-dim input layer with
zero-filled columns, so the policy starts exactly as the BC policy and learns
to use the traffic-light features through training.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from carla_rl.models import (
    GaussianActor,
    ObsNormalizer,
    TwinQ,
    convert_3d_actor_to_1d,
    expand_first_layer,
)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim=3):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.obs), to(self.act), to(self.rew).unsqueeze(-1), \
            to(self.next_obs), to(self.done).unsqueeze(-1)

    def save(self, path):
        """Persist the used portion (~1 GB at 400k x 315 dims, uncompressed for
        speed). Resuming WITHOUT the buffer was the root cause of three
        training collapses: the replay gets repopulated with dataset prefill
        only, which is off-distribution for an evolved policy."""
        np.savez(
            path,
            obs=self.obs[: self.size], act=self.act[: self.size],
            rew=self.rew[: self.size], next_obs=self.next_obs[: self.size],
            done=self.done[: self.size], ptr=self.ptr, size=self.size,
        )

    def load(self, path):
        data = np.load(path)
        n = int(data['size'])
        assert n <= self.capacity, f'buffer file ({n}) exceeds capacity ({self.capacity})'
        self.obs[:n] = data['obs']
        self.act[:n] = data['act']
        self.rew[:n] = data['rew']
        self.next_obs[:n] = data['next_obs']
        self.done[:n] = data['done']
        self.size = n
        self.ptr = int(data['ptr']) % self.capacity
        return n


class SAC:
    def __init__(self, obs_dim, act_dim=3, hidden=(256, 256), device=None,
                 gamma=0.99, tau=0.005, lr=3e-4, actor_lr=None,
                 critic_layernorm=False, action_low=None, action_high=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.hidden = tuple(hidden)
        self.critic_layernorm = bool(critic_layernorm)
        self.actor = GaussianActor(obs_dim, act_dim, hidden,
                                   low=action_low, high=action_high).to(self.device)
        self.critic = TwinQ(obs_dim, act_dim, hidden,
                            layernorm=self.critic_layernorm).to(self.device)
        self.critic_target = TwinQ(obs_dim, act_dim, hidden,
                                   layernorm=self.critic_layernorm).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr or lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        self.normalizer = None  # set when loading a BC checkpoint

        # behavior-anchor (TD3+BC style): a frozen BC actor the policy is
        # regularized toward, so SAC can't drift into the high-frequency
        # jitter the diagnosis pinned on unconstrained fine-tuning.
        self.bc_actor = None
        self.bc_anchor = 0.0   # RL/Q-term scale; smaller => policy hugs BC harder
        self.bc_dim = 307

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def normalize(self, obs):
        return self.normalizer(obs) if self.normalizer is not None else obs

    def set_bc_anchor(self, bc_checkpoint, weight):
        """Load a frozen BC actor as the regularization anchor. The buffer's
        obs[:, :bc_dim] is already BC-normalized (the SAC normalizer is the BC
        normalizer with fit_dim=bc_dim), so the BC actor consumes that slice
        directly. weight is the RL/Q-term scale (TD3+BC's alpha): smaller pulls
        the policy toward BC harder."""
        ckpt = torch.load(bc_checkpoint, map_location=self.device, weights_only=False)
        bc = GaussianActor(ckpt['obs_dim'], hidden=tuple(ckpt['hidden'])).to(self.device)
        bc.load_state_dict(ckpt['actor'])
        bc.eval()
        for p in bc.parameters():
            p.requires_grad_(False)
        self.bc_actor = bc
        self.bc_dim = int(ckpt['obs_dim'])
        self.bc_anchor = float(weight)

    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(
            self.normalize(obs), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.mean_action(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def update(self, buffer, batch_size=256, update_actor=True):
        obs, act, rew, next_obs, done = buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            q_target = rew + self.gamma * (1.0 - done) * (
                torch.min(q1_t, q2_t) - self.alpha * next_logp
            )

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        metrics = {'critic_loss': critic_loss.item(), 'q1_mean': q1.mean().item()}

        if update_actor:
            new_act, logp = self.actor.sample(obs)
            q1_pi, q2_pi = self.critic(obs, new_act)
            q_pi = torch.min(q1_pi, q2_pi)
            bc_mse_val = 0.0
            if self.bc_actor is not None and self.bc_anchor > 0.0:
                # TD3+BC: normalize the Q term to a fixed scale so the
                # (unweighted) BC anchor stays comparable as |Q| grows. Anchor
                # the DETERMINISTIC mean — the diagnosis showed the mean itself
                # jitters, not just exploration noise.
                lmbda = (self.bc_anchor / q_pi.abs().mean()).detach()
                with torch.no_grad():
                    a_bc = self.bc_actor.mean_action(obs[:, :self.bc_dim])
                bc_mse = F.mse_loss(self.actor.mean_action(obs), a_bc)
                actor_loss = (self.alpha.detach() * logp - lmbda * q_pi).mean() + bc_mse
                bc_mse_val = bc_mse.item()
            else:
                actor_loss = (self.alpha.detach() * logp - q_pi).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            metrics.update({
                'actor_loss': actor_loss.item(),
                'alpha': self.alpha.item(),
                'entropy': -logp.mean().item(),
                'bc_mse': bc_mse_val,
            })

        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.mul_(1.0 - self.tau).add_(self.tau * p)

        return metrics

    def load_bc_checkpoint(self, path):
        """Initialize the actor from a BC checkpoint trained on the 307-dim
        dataset obs; the input layer is widened with zero columns to obs_dim."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = expand_first_layer(ckpt['actor'], ckpt['obs_dim'], self.obs_dim)
        self.actor.load_state_dict(state)
        # BC trains only the mean head; the log_std half of the output layer
        # is whatever random init it had -> large state-dependent exploration
        # noise from step one. Calm it: state-independent sigma ~ 0.37.
        with torch.no_grad():
            head = self.actor.body[-1]
            act_dim = head.out_features // 2
            head.weight[act_dim:].zero_()
            head.bias[act_dim:].fill_(-1.0)
        norm = ObsNormalizer.from_state_dict(ckpt['normalizer'])
        self.normalizer = norm
        return ckpt

    def init_signed_pedal_from_3d(self, path):
        """從舊三維 checkpoint 初始化 1D signed-pedal 策略(spec 7/8):只轉移 actor 的
        共享特徵層 + 由 throttle/brake 列折出的新縱向頭,以及 observation normalizer;
        critic / target critic / alpha / replay buffer 全部維持「全新」(動作維度不同,
        舊三維 critic 不可載入)。本 SAC 須以 act_dim=1 建立。"""
        assert self.act_dim == 1, 'init_signed_pedal_from_3d requires act_dim=1'
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        conv = convert_3d_actor_to_1d(ckpt['actor'], self.obs_dim, self.hidden)
        self.actor.load_state_dict(conv.state_dict())
        self.actor.to(self.device)
        if ckpt.get('normalizer'):
            self.normalizer = ObsNormalizer.from_state_dict(ckpt['normalizer'])
        return ckpt

    def save(self, path, meta=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'obs_dim': self.obs_dim,
            'hidden': list(self.hidden),
            'critic_layernorm': self.critic_layernorm,
            'normalizer': self.normalizer.state_dict() if self.normalizer else None,
            'meta': meta or {},
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt['log_alpha'].to(self.device))
        if ckpt.get('normalizer'):
            self.normalizer = ObsNormalizer.from_state_dict(ckpt['normalizer'])
        return ckpt
