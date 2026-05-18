# PianoMime Baseline Plan and Status

Last updated: 2026-05-19

## Course Requirement Extract

The Dexterous Piano Track asks us to:

- Review robot piano playing and introduce RoboPianist, PianoMime, and RP1M.
- Run the original PianoMime baseline.
- Select 3 training-set songs that sound good for the robot to play.
- Produce final performance videos.
- Visualize an F1 score training curve.
- Train a multi-song policy or use provided checkpoints.
- Show generalist policy performance videos on unseen/test songs.
- Later, improve both the single-song policy and the generalist policy.

## Baseline Reproduction Status

Baseline reproduction is complete for the PDF requirements.

Completed items:

- Environment and dependencies are available under
  `/home/gaoj/share4/_piano/.venv`.
- Official PianoMime datasets and checkpoints are restored under the project
  directory.
- RoboPianist can run in headless MuJoCo EGL mode.
- Three training-set single-song replay baselines were evaluated:
  `Stan_1`, `Petrunko_3`, and `NeverGonnaGiveYouUp_1`.
- `Petrunko_3` PPO residual training was rerun for 2000 iterations and produced
  `eval_metrics.csv`, `eval_f1_curve.png`, and a final rollout video.
- Seven unseen-song generalist diffusion checkpoint evaluations were completed:
  `Alone_1`, `Numb_1`, `NoTimeToDie_1`, `Forester_1`, `EyesClosed_1`,
  `Paradise_1`, and `SomewhereOnlyWeKnow_1`.
- Result documentation is now centralized in `docs/BASELINE_RESULTS.md`.
- Code modification documentation is now centralized in
  `docs/CODE_MODIFICATION_SUMMARY.md`.
- Paths, task lists, scheduler defaults, and core baseline hyperparameters are
  centralized in `configs/baseline.toml`; see `docs/CONFIGURATION.md`.

## Result Locations

Shared result root:

```text
/home/gaoj/share4/_piano/baseline_results
```

Main result files:

```text
single_song/metrics.csv
single_song/videos/
single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/
multisong/metrics.csv
multisong/videos/
multisong/logs/
```

## Current Codebase Status

The repository is ready for teammates to inspect and use for follow-up
experiments, with the following notes:

- The working tree is a Git repository and is clean after the current docs are
  committed.
- Large artifacts are intentionally ignored and restored through
  `scripts/setup_artifacts.sh`.
- Use `configs/baseline.toml` for reproduction; copy it for new methods so the
  baseline config stays untouched.
- Long runs should use `scripts/start_tmux_baseline.sh` and local scratch
  execution to reduce shared-filesystem stalls.
- The current server has verified A800 runs. A 4090 smoke script exists, but a
  physical 4090 run has not been verified on this host.

## Next Research Work

The remaining work is no longer baseline reproduction. It belongs to the
algorithm-improvement phase:

- Choose one single-song improvement target and compare against the recorded
  baseline with F1 curve plus video.
- Choose one generalist improvement idea and compare baseline vs improved F1 on
  at least five unseen songs.
- Keep all new metrics and videos in the same `baseline_results` style, or add
  a parallel `improvement_results` directory.

## Optional Maintenance Work

- Push to GitHub once the new empty remote repository is created.
- Run a fresh-clone smoke test after the remote exists.
- Install FluidSynth/PortAudio system libraries only if audio-bearing videos
  are required.
