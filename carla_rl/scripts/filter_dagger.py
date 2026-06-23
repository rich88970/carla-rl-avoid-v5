"""Disagreement-filter the recovery-DAgger labels (V11).

The fast TM's edge over the student isn't systematic extra braking — averaged
over a recovery window its pedals look like normal cruising. Its edge is a rare,
precisely-timed correction at the imminent moment. To stop those from washing
out, keep only labels where the expert is MEANINGFULLY MORE CAUTIOUS than the
student would be on the SAME observation (longitudinal aggression = throttle -
brake; keep where student_aggression - expert_aggression > margin).

Reports the disagreement distribution, then saves the filtered subset.

Usage: python -m carla_rl.scripts.filter_dagger <dagger.npz> <student.pth> <out.npz> [margin]
"""

import sys

import numpy as np


def main():
    dagger_path = sys.argv[1] if len(sys.argv) > 1 else r'data\dagger_iter1.npz'
    student_ckpt = sys.argv[2] if len(sys.argv) > 2 else r'carla_rl\checkpoints\bc_autopilot.pth'
    out = sys.argv[3] if len(sys.argv) > 3 else r'data\dagger_filtered_iter1.npz'
    margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.20

    import torch
    from carla_rl.models import GaussianActor, ObsNormalizer

    d = np.load(dagger_path)
    obs, exp = d['obs'].astype(np.float32), d['act'].astype(np.float32)

    ckpt = torch.load(student_ckpt, map_location='cpu', weights_only=False)
    actor = GaussianActor(ckpt['obs_dim'], hidden=tuple(ckpt['hidden']))
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    norm = ObsNormalizer.from_state_dict(ckpt['normalizer'])
    with torch.no_grad():
        stu = actor.mean_action(
            torch.as_tensor(norm(obs[:, : ckpt['obs_dim']]))).numpy()

    # longitudinal aggression = throttle - brake; positive disagreement means the
    # student would drive more aggressively than the expert here (a correction)
    aggr_e = exp[:, 0] - exp[:, 2]
    aggr_s = np.clip(stu[:, 0], 0, 1) - np.clip(stu[:, 2], 0, 1)
    disagree = aggr_s - aggr_e

    pct = np.percentile(disagree, [10, 50, 75, 90, 95, 99])
    print(f'disagreement (student-expert aggression) percentiles '
          f'[10,50,75,90,95,99]: {np.round(pct, 3)}')
    for m in (0.1, 0.2, 0.3, 0.4, 0.5):
        print(f'  margin {m}: keep {(disagree > m).sum()} / {len(obs)} '
              f'({100 * (disagree > m).mean():.1f}%)')

    keep = disagree > margin
    f_obs, f_exp = obs[keep], exp[keep]
    np.savez(out, obs=f_obs, act=f_exp)
    print(f'\nmargin {margin}: kept {len(f_obs)} corrective labels -> {out}')
    if len(f_obs):
        print(f'kept mean: expert throttle={f_exp[:, 0].mean():.3f} '
              f'brake={f_exp[:, 2].mean():.3f} | student-on-same throttle='
              f'{np.clip(stu[keep, 0], 0, 1).mean():.3f} '
              f'brake={np.clip(stu[keep, 2], 0, 1).mean():.3f}')


if __name__ == '__main__':
    main()
