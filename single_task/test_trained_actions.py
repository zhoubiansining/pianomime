import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
from robopianist.suite.tasks import piano_with_shadow_hands_res
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist.wrappers.deep_mimic import DeepMimicWrapper
from robopianist.wrappers.residual import ResidualWrapper
from robopianist.wrappers.dm2gym import Dm2GymWrapper
from dm_env_wrappers import SinglePrecisionWrapper
from dm_env_wrappers import DmControlWrapper
from robopianist.wrappers.evaluation import MidiEvaluationWrapper
from mujoco_utils import composer_utils
import tqdm
import pickle
from pianomime_config import DEFAULT_CONFIG_PATH, load_config, section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("song_name")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()

def play_video(filename: str):
    return filename

args = parse_args()
config = load_config(args.config)
replay_config = section(config, "single_song", "replay")
task_name = args.song_name

# start_from = start_from_dict.START_FROM[task_name]
with (PROJECT_ROOT / "dataset" / "notes" / f"{task_name}.pkl").open("rb") as f:
    note_traj = pickle.load(f)


task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
    note_trajectory=note_traj,
    change_color_on_activation=True,
    trim_silence=True,
    control_timestep=float(replay_config.get("control_timestep", 0.05)),
    disable_hand_collisions=bool(replay_config.get("disable_hand_collisions", True)),
    disable_forearm_reward=bool(replay_config.get("disable_forearm_reward", True)),
    disable_fingering_reward=bool(replay_config.get("disable_fingering_reward", False)),
    midi_start_from=int(replay_config.get("midi_start_from", 0)),
    n_steps_lookahead=int(replay_config.get("n_steps_lookahead", 10)),
    gravity_compensation=bool(replay_config.get("gravity_compensation", True)),
    residual_factor=float(replay_config.get("residual_factor", 0.03)),
    shift=int(replay_config.get("shift", 0)),
)

# Load hand action trajectory
left_hand_action_list = np.load(PROJECT_ROOT / "dataset" / "high_level_trajectories" / f"{task_name}_left_hand_action_list.npy")
right_hand_action_list = np.load(PROJECT_ROOT / "dataset" / "high_level_trajectories" / f"{task_name}_right_hand_action_list.npy")

# Load trained actions
actions = np.load(PROJECT_ROOT / "dataset" / "low_level_policies" / task_name / f"actions_{task_name}.npy")

env = composer_utils.Environment(
    recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
)

env = PianoSoundVideoWrapper(
    env,
    record_every=int(replay_config.get("record_every", 1)),
    camera_id=replay_config.get("camera_id", "piano/back"),
    record_dir=replay_config.get("record_dir", "."),
)
env = DeepMimicWrapper(env,
                      demonstrations_lh=left_hand_action_list,
                      demonstrations_rh=right_hand_action_list,
                      remove_goal_observation=False,
                      mimic_z_axis=bool(replay_config.get("mimic_z_axis", False)),)
env = ResidualWrapper(env, 
                      demonstrations_lh=left_hand_action_list,
                      demonstrations_rh=right_hand_action_list,
                      demo_ctrl_timestep=float(replay_config.get("demo_ctrl_timestep", 0.05)),)
env = MidiEvaluationWrapper(
    environment=env, deque_size=1
)
env = CanonicalSpecWrapper(env, clip=bool(replay_config.get("clip", True)))

env = SinglePrecisionWrapper(env)
env = DmControlWrapper(env)

env = Dm2GymWrapper(env)
step = 0
err_poses = list()

demos = []
env = env.env
timestep = env.reset()
reward = 0

timesteps = tqdm.tqdm(range(actions.shape[0]))
for step in timesteps:
    action = actions[step]
    timestep = env.step(action)
    reward += timestep.reward
    timesteps.set_description(f"Reward: {reward:.2f}")
    if timestep.last():
        break

print(env.get_musical_metrics())

play_video(env.latest_filename)
