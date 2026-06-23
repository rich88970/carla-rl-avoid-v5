# Third-Party Notices

This project is based in part on EasyCarla-RL.

- Project: EasyCarla-RL
- Repository: https://github.com/silverwingsbot/EasyCarla-RL
- Base commit: fc1bcfe6d63c9d999837c8e1b7c2cfa092e1c640
- License: Apache License 2.0

Local modifications to the upstream base environment
(`easycarla/envs/carla_env.py`) are provided as a patch in
`patches/easycarla_local.patch`:

- client timeout 10s → 120s (Town03 load exceeds 10s on the source build)
- reuse an already-loaded world instead of `load_world` (the reload
  intermittently crashes the source-built server with a SkeletalMesh assert)

Additions in this repository (the `carla_rl/` package) include:

- Behavior-cloning warm start integration
- SAC fine-tuning (LayerNorm twin-Q critic, reward scaling)
- Reward shaping for TTC and distance-gap risk
- Pure-Pursuit hierarchical lateral control
- DSAFE dynamic safety distance and graded braking
- Risk (5-sector) and predictive observation features
- Oncoming-vehicle filtering (`world_forward_lead`)
- Evaluation, logging, plotting, and video tools

Modified and added files are documented through this repository's Git history.
