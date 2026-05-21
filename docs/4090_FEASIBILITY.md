# RTX 4090 Feasibility

I cannot truthfully claim this server has physically run the baseline on an RTX
4090 unless a 4090 is visible through `nvidia-smi`. The current verified runs
used the available server GPUs.

That said, the baseline is expected to be feasible on a 24 GB 4090:

- The high-level and low-level diffusion checkpoints are hundreds of MB, not
  multi-billion-parameter models.
- Single-song PPO is mostly MuJoCo/vector-env CPU work plus a moderate MLP
  policy. GPU memory is not the limiting factor.
- CUDA 11.8 PyTorch wheels are appropriate for modern NVIDIA GPUs when the
  installed driver supports them.
- The main operational requirement is to run one job per 4090 and avoid
  concurrent jobs fighting for the same 24 GB card.

Recommended 4090 settings:

```bash
cd /home/gaoj/share4/_piano/pianomime
GPU_IDS="0" GPU_FREE_MEM_MB=3000 SESSION=pianomime_4090_smoke \
  bash scripts/start_tmux_baseline.sh
```

Before running the full suite:

```bash
bash scripts/check_4090_feasibility.sh
```

If a 4090 run fails, first check:

- `nvidia-smi` driver/CUDA compatibility
- `MUJOCO_GL=egl`
- whether another process is already using most of the GPU memory
- whether the run is executing from local scratch rather than the shared
  directory
