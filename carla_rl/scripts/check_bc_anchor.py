"""Verify the TD3+BC behavior-anchor wiring (offline, no CARLA server).

Checks:
  1. A BC-initialized SAC actor matches the frozen BC actor (anchor mse ~ 0) —
     confirms obs[:, :307] feeds the BC actor correctly and widening is inert.
  2. Perturbing the policy raises the anchor mse (the regularizer has signal).
  3. One real update step runs and reports bc_mse; the TD3+BC lambda
     normalizes the Q term to ~bc_anchor scale.
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from carla_rl.agents.sac import SAC, ReplayBuffer
from carla_rl.data.offline_dataset import DATASET_PATH
from carla_rl.wrappers.traffic_light import NEUTRAL_FEATURES

BC = r'carla_rl\checkpoints\bc_actor.pth'
DIM = 312  # v3 layout: 307 dataset + 5 traffic-light (no smooth wrapper)


def main():
    bc_hidden = tuple(torch.load(BC, map_location='cpu', weights_only=False)['hidden'])
    agent = SAC(DIM, hidden=bc_hidden)
    agent.load_bc_checkpoint(BC)            # widens 307 -> 312, sets normalizer
    agent.set_bc_anchor(BC, weight=1.0)

    with h5py.File(DATASET_PATH, 'r') as f:
        raw = f['observations'][1000:1256]
    obs = np.stack([agent.normalize(np.concatenate([o, NEUTRAL_FEATURES])) for o in raw])
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)

    with torch.no_grad():
        a_pi = agent.actor.mean_action(obs_t)
        a_bc = agent.bc_actor.mean_action(obs_t[:, :agent.bc_dim])
        init_mse = F.mse_loss(a_pi, a_bc).item()
    print(f'1. BC-init policy vs frozen BC: anchor_mse = {init_mse:.6e}  (expect ~0)')

    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        pert_mse = F.mse_loss(agent.actor.mean_action(obs_t), a_bc).item()
    print(f'2. perturbed policy vs frozen BC: anchor_mse = {pert_mse:.6e}  (expect >> init)')

    buf = ReplayBuffer(512, DIM)
    acts = np.random.uniform(-1, 1, (256, 3)).astype(np.float32)
    for i in range(256):
        buf.add(obs[i], acts[i], 1.0, obs[i], 0.0)
    m = agent.update(buf, batch_size=256, update_actor=True)
    print(f"3. update OK: bc_mse={m['bc_mse']:.5f}  actor_loss={m['actor_loss']:.3f}  "
          f"q1_mean={m['q1_mean']:.3f}")

    assert init_mse < 1e-3, 'BC-init anchor mse should be ~0'
    assert pert_mse > init_mse * 10, 'perturbation should clearly raise anchor mse'
    assert 'bc_mse' in m
    print('\nBC ANCHOR WIRING OK')


if __name__ == '__main__':
    main()
