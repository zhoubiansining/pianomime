#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

resolve_conda_bin() {
    if [[ -n "${CONDA_BIN:-}" ]]; then
        printf '%s\n' "${CONDA_BIN}"
    elif command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [[ -x "/workspace/lwk/anaconda3/bin/conda" ]]; then
        printf '%s\n' "/workspace/lwk/anaconda3/bin/conda"
    elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
        printf '%s\n' "${HOME}/anaconda3/bin/conda"
    elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
        printf '%s\n' "${HOME}/miniconda3/bin/conda"
    else
        echo "conda was not found. Set CONDA_BIN=/path/to/conda or initialize conda in PATH." >&2
        exit 1
    fi
}

CONDA_BIN_RESOLVED="$(resolve_conda_bin)"

cd "${PROJECT_DIR}"

env \
    WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/}" \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}" \
    "${CONDA_BIN_RESOLVED}" run --no-capture-output -n "${CONDA_ENV:-piano}" \
    python pianomime/single_task/train_grpo.py \
        --root-dir "${ROOT_DIR:-${PROJECT_DIR}/robopianist_rl/video/}" \
        --warmstart-steps "${WARMSTART_STEPS:-5000}" \
        --max-steps "${MAX_STEPS:-1000000}" \
        --discount "${DISCOUNT:-0.99}" \
        --trim-silence \
        --gravity-compensation \
        --control-timestep "${CONTROL_TIMESTEP:-0.05}" \
        --n-steps-lookahead "${N_STEPS_LOOKAHEAD:-0}" \
        --disable-fingering-reward \
        --disable-hand-collisions \
        --disable-forearm-reward \
        --tqdm-bar \
        --eval-episodes "${EVAL_EPISODES:-1}" \
        --eval-every-iters "${EVAL_EVERY_ITERS:-10}" \
        --camera-id "${CAMERA_ID:-piano/back}" \
        --midi-start-from "${MIDI_START_FROM:-0}" \
        --residual-action \
        --frame-stack "${FRAME_STACK:-4}" \
        --num-envs "${NUM_ENVS:-32}" \
        --group-size "${GROUP_SIZE:-8}" \
        --trajectory-horizon "${TRAJECTORY_HORIZON:-0}" \
        --n-epochs "${N_EPOCHS:-4}" \
        --clip-range "${CLIP_RANGE:-0.2}" \
        --ent-coef "${ENT_COEF:-0.0}" \
        --initial-lr "${INITIAL_LR:-3e-4}" \
        --lr-decay-rate "${LR_DECAY_RATE:-0.999}" \
        --n-steps "${N_STEPS:-512}" \
        --mimic-task "${MIMIC_TASK:-Petrunko_3}" \
        --environment-name "${ENVIRONMENT_NAME:-Petrunko_3}" \
        --project "${WANDB_PROJECT:-robopianist-Petrunko_3}" \
        --entity "${ENTITY:-}" \
        --mode "${WANDB_MODE:-online}" \
        --tags "${TAGS:-}" \
        --notes "${NOTES:-}" \
        --use-note-trajectory \
        --total-iters "${TOTAL_ITERS:-2000}" \
        --residual-factor "${RESIDUAL_FACTOR:-0.03}" \
        --deepmimic \
        --seed "${SEED:-42}" \
        --name "${NAME:-grpo_r8_wo_rsi}"
