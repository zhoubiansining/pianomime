from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm
import torch

import logging_callback
import lr_scheduler

import os

import dm_env_wrappers as wrappers

import robopianist.wrappers as robopianist_wrappers
import wrappers as pianomime_wrappers
from robopianist.suite.tasks import piano_with_shadow_hands, piano_with_shadow_hands_multitask
import piano_with_shadow_hands_res
from robopianist import music
from mujoco_utils import composer_utils
import gymnasium as gym
import pickle
from stable_baselines3.common.utils import set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_note_trajectory(task_name):
    with (PROJECT_ROOT / "dataset" / "notes" / f"{task_name}.pkl").open("rb") as f:
        return pickle.load(f)


def _align_demonstrations_to_task(demonstrations_lh, demonstrations_rh, task, task_name):
    target_len = len(task._notes)
    current_len = demonstrations_lh.shape[0]
    if current_len == target_len:
        return demonstrations_lh, demonstrations_rh
    if current_len > target_len:
        print(f"Truncating {task_name} demonstrations from {current_len} to {target_len} frames.")
        return demonstrations_lh[:target_len], demonstrations_rh[:target_len]
    pad = target_len - current_len
    print(f"Padding {task_name} demonstrations from {current_len} to {target_len} frames.")
    demonstrations_lh = np.concatenate(
        [demonstrations_lh, np.repeat(demonstrations_lh[-1:], pad, axis=0)],
        axis=0,
    )
    demonstrations_rh = np.concatenate(
        [demonstrations_rh, np.repeat(demonstrations_rh[-1:], pad, axis=0)],
        axis=0,
    )
    return demonstrations_lh, demonstrations_rh

def get_env_no_residual(args, record_dir: Optional[Path] = None):
    left_hand_action_list = np.load(
        PROJECT_ROOT / "dataset" / "high_level_trajectories" / f"{args.mimic_task}_left_hand_action_list.npy"
    )
    right_hand_action_list = np.load(
        PROJECT_ROOT / "dataset" / "high_level_trajectories" / f"{args.mimic_task}_right_hand_action_list.npy"
    )
    if args.use_note_trajectory:
        note_traj = _load_note_trajectory(args.mimic_task)
        task = piano_with_shadow_hands.PianoWithShadowHands(
                note_trajectory=note_traj,
                change_color_on_activation=True,
                trim_silence=args.trim_silence,
                control_timestep=args.control_timestep,
                disable_hand_collisions=args.disable_hand_collisions,
                disable_forearm_reward=args.disable_forearm_reward,
                disable_fingering_reward=args.disable_fingering_reward,
                midi_start_from=args.midi_start_from,
                n_steps_lookahead=args.n_steps_lookahead,
                gravity_compensation=args.gravity_compensation,
                reduced_action_space=args.reduced_action_space,
            )
    else:
        task = piano_with_shadow_hands.PianoWithShadowHands(
                midi=music.load(args.mimic_task),
                change_color_on_activation=True,
                trim_silence=args.trim_silence,
                control_timestep=args.control_timestep,
                disable_hand_collisions=args.disable_hand_collisions,
                disable_forearm_reward=args.disable_forearm_reward,
                disable_fingering_reward=args.disable_fingering_reward,
                midi_start_from=args.midi_start_from,
                n_steps_lookahead=args.n_steps_lookahead,
                gravity_compensation=args.gravity_compensation,
                reduced_action_space=args.reduced_action_space,
            )
    left_hand_action_list, right_hand_action_list = _align_demonstrations_to_task(
        left_hand_action_list, right_hand_action_list, task, args.mimic_task
    )
    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )
    if args.deepmimic:
        print("deepmimic")
        env = robopianist_wrappers.DeepMimicWrapper(
            environment=env,
            demonstrations_lh=left_hand_action_list,
            demonstrations_rh=right_hand_action_list,
            demo_ctrl_timestep=args.control_timestep,
            remove_goal_observation=False,
            mimic_z_axis=args.mimic_z_axis,
            n_steps_lookahead=args.n_steps_lookahead,
        )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    env = robopianist_wrappers.Dm2GymWrapper(env)
    return env


def get_env(args, record_dir: Optional[Path] = None):
    left_hand_action_list = np.load(
        PROJECT_ROOT / "dataset" / "high_level_trajectories" / f"{args.mimic_task}_left_hand_action_list.npy"
    )
    right_hand_action_list = np.load(
        PROJECT_ROOT / "dataset" / "high_level_trajectories" / f"{args.mimic_task}_right_hand_action_list.npy"
    )
    length = left_hand_action_list.shape[0]
    trim = False if length >=600 or length < 500 else True
    print(trim)
    if args.use_note_trajectory:
        note_traj = _load_note_trajectory(args.mimic_task)
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
                note_trajectory=note_traj,
                change_color_on_activation=True,
                wrong_press_termination=args.wrong_press_termination,
                trim_silence=trim if args.trim_silence else False,
                control_timestep=args.control_timestep,
                disable_hand_collisions=args.disable_hand_collisions,
                disable_forearm_reward=args.disable_forearm_reward,
                disable_fingering_reward=args.disable_fingering_reward,
                midi_start_from=args.midi_start_from,
                n_steps_lookahead=args.n_steps_lookahead,
                gravity_compensation=args.gravity_compensation,
                reduced_action_space=args.reduced_action_space,
                residual_factor=args.residual_factor,
                curriculum=args.curriculum,
            )
    else:   
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
            midi=music.load(args.mimic_task),
            change_color_on_activation=True,
            wrong_press_termination=args.wrong_press_termination,
            trim_silence=trim if args.trim_silence else False,
            control_timestep=args.control_timestep,
            disable_hand_collisions=args.disable_hand_collisions,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_fingering_reward=args.disable_fingering_reward,
            midi_start_from=args.midi_start_from,
            n_steps_lookahead=args.n_steps_lookahead,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            residual_factor=args.residual_factor,
        )
    left_hand_action_list, right_hand_action_list = _align_demonstrations_to_task(
        left_hand_action_list, right_hand_action_list, task, args.mimic_task
    )

    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )
    print(left_hand_action_list.shape[0])
    if args.deepmimic:
        print("deepmimic")
        env = pianomime_wrappers.DeepMimicWrapper(
            environment=env,
            demonstrations_lh=left_hand_action_list,
            demonstrations_rh=right_hand_action_list,
            demo_ctrl_timestep=args.control_timestep,
            remove_goal_observation=False,
            mimic_z_axis=args.mimic_z_axis,
            n_steps_lookahead=args.n_steps_lookahead,
        )
    if args.residual_action:
        print("residual action")
        env = pianomime_wrappers.ResidualWrapper(
            environment=env,
            demonstrations_lh=left_hand_action_list,
            demonstrations_rh=right_hand_action_list,
            demo_ctrl_timestep=args.control_timestep,
            rsi=args.rsi,
        )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    env = robopianist_wrappers.Dm2GymWrapper(env)
    return env

def make_envs(make_env_fn, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = make_env_fn()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

def get_env_multitask(args, task_names, record_dir: Optional[Path] = None):
    if args.use_note_trajectory:
        note_trajs = []
        demo_rhs = []
        demo_lhs = []

        for task in task_names:
            with open('handtracking/notes/{}.pkl'.format(task), 'rb') as f:
                note_traj = pickle.load(f)
            note_trajs.append(note_traj)

        task = piano_with_shadow_hands_multitask.PianoWithShadowHandsMultiTask(
            note_trajectories=note_trajs,
            task_names=task_names,
            change_color_on_activation=True,
            wrong_press_termination=args.wrong_press_termination,
            trim_silence=True,
            control_timestep=0.05, #
            disable_hand_collisions=True,
            disable_forearm_reward=True,
            disable_fingering_reward=False,
            midi_start_from=0,
            n_steps_lookahead=args.n_steps_lookahead,
            gravity_compensation=True,
            reduced_action_space=False,
            residual_factor=args.residual_factor,
            curriculum=args.curriculum,
            fingering_lookahead=True,
        )

    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )
    # print(env.observation_spec())
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    env = robopianist_wrappers.Dm2GymWrapper(env)
    return env
