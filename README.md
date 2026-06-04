# PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations
[[Project page]](https://pianomime.github.io/)
[[Paper]](https://arxiv.org/pdf/2407.18178)
[[Arxiv]](https://arxiv.org/abs/2407.18178)
[[Colab]](https://colab.research.google.com/drive/1Rv1XGPA0a4x3a_M6yXc7uiwKnmmIu95o?usp=sharing)

## Course Baseline Fork

This fork keeps the original PianoMime codebase while adding a course-oriented
baseline workflow: reproducibility scripts, headless-server fixes, centralized
configuration, and tmux automation for long runs.

### What Is Maintained for the Course

- Single-song replay baseline: `scripts/test_trained_actions.sh`
- Single-song PPO baseline: `scripts/run_ppo.sh`
- Generalist checkpoint evaluation: `scripts/run_multisong_task.sh`
- Automated multi-job runner: `scripts/start_tmux_baseline.sh`
- Central config for paths, tasks, and hyperparameters: `configs/baseline.toml`

The reproduced baseline currently includes training-set single-song replay
videos, PPO F1 training curves for the aligned single-song set, and seven
unseen-song generalist evaluation videos. Downstream single-song work is
aligned on `TwinkleTwinkleRousseau`, `Pirates_1`, `Stan_1`, and `Petrunko_3`;
the first two songs required demo/MIDI alignment and IK/QP numerical fallback
fixes, and their full 2000-iteration PPO baselines are now complete. Exact
metrics and output paths are indexed in:

- `docs/BASELINE_RESULTS.md`
- `docs/BASELINE_RESULTS_zh.md`
- `docs/BASELINE_REPORT_SECTION.tex`
- `docs/BASELINE_REPORT_MATERIALS_zh.md`
- `docs/REPORT_ASSETS_MANIFEST_zh.md`
- `docs/SINGLE_SONG_FOUR_BASELINE.md`

### Quick Start

These steps are the recommended handoff path for course teammates.

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/zhoubiansining/pianomime.git
   cd pianomime
   ```

2. Install system dependencies:

   ```bash
   bash scripts/install_deps.sh
   ```

   On Linux this installs `ffmpeg`, `fluidsynth`, `portaudio`, and related
   packages. The repository already contains the bundled Shadow Hand assets and
   the default soundfont, so a fresh clone does not require any separate
   `third_party` checkout.

3. Create the Python environment:

   ```bash
   bash scripts/setup_python_env.sh
   ```

   This script creates the virtualenv from `configs/baseline.toml`, installs
   CUDA PyTorch, and then installs `requirements.txt`.

4. Download the official dataset and released checkpoints:

   ```bash
   bash scripts/setup_artifacts.sh
   ```

   The repository already includes `dataset_hl.zarr` and `dataset_ll.zarr`, so
   the multi-song baseline does not need any extra shared-server copy step.

5. Run a smoke test:

   ```bash
   bash scripts/check_4090_feasibility.sh
   ```

6. Run one baseline command:

   ```bash
   bash scripts/test_trained_actions.sh Stan_1
   # or
   bash scripts/run_multisong_task.sh Alone_1 0
   # or
   bash scripts/run_ppo.sh Petrunko_3
   ```

### Teammate Reading Order

- Chinese handoff overview: `COURSE_BASELINE_zh.md`
- English handoff overview: `COURSE_BASELINE.md`
- Usage details: `docs/USAGE.md`, `docs/USAGE_zh.md`
- Config details: `docs/CONFIGURATION.md`, `docs/CONFIGURATION_zh.md`
- tmux automation: `docs/EXPERIMENT_AUTOMATION.md`, `docs/EXPERIMENT_AUTOMATION_zh.md`
- Current caveats: `docs/problems.md`, `docs/problems_zh.md`

### Scope Notes

- The course-maintained workflow is centered on the scripts above and the
  checkpoints released by the original project.
- Some research and training files under `multi_task/`, `goal_auto_encoder/`,
  and `tutorial/` are kept for reference but are not the primary course handoff
  entrypoints.
- The dataset-preparation notebook may need extra packages such as MediaPipe
  beyond the baseline environment.

## Original Project Context

**Cheng Qian**<sup>1</sup>, **Julen Urain**<sup>2</sup>, **Kevin Zakka**<sup>3</sup>, **Jan Peters**<sup>2</sup>

<sup>1</sup>TU Munich,
<sup>2</sup>TU Darmstadt,
<sup>3</sup>UC Berkeley

TLDR:
We train a generalist policy for controlling dexterous robot hands to play any
songs using human pianist demonstration videos from the internet. We use
residual reinforcement learning to learn song-specific policies from
demonstrations, and a two-stage diffusion policy to generalize to new songs.

[![Video](https://i.ytimg.com/vi/LW0AiBIcnL0/hqdefault.jpg)](https://youtu.be/LW0AiBIcnL0)

## Dataset Preparation Tutorial

We provide a notebook for preparing data from videos and MIDI files:

[Data Preparation Tutorial](tutorial/data_preprocessing.ipynb)

Inside the notebook, you can learn how to:

- Estimate the homography matrix from video coordinates to piano coordinates
- Extract fingering labels and fingertip trajectories from videos
- Format the processed data for training

We also provide a Google Colab version:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rv1XGPA0a4x3a_M6yXc7uiwKnmmIu95o?usp=sharing)

## Citation

Please use the following citation:

```bibtex
@misc{qian2024pianomimelearninggeneralistdexterous,
      title={PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations},
      author={Cheng Qian and Julen Urain and Kevin Zakka and Jan Peters},
      year={2024},
      eprint={2407.18178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.18178},
}
```

## Acknowledgements

The simulation environment is based on
[RoboPianist](https://github.com/google-research/robopianist).

The diffusion policy is adapted from
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

The inverse-kinematics controller is adapted from
[Pink](https://github.com/stephane-caron/pink).

The human demonstration videos are downloaded from YouTube channel
[PianoX](https://www.youtube.com/channel/UCsR6ZEA0AbBhrF-NCeET6vQ).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
