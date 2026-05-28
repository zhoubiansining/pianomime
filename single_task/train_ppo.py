import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
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
from copy import copy
from dataclasses import dataclass, replace

import logging_callback
import lr_scheduler

import os
from mujoco_utils import composer_utils
import gymnasium as gym
from utils import get_env, make_envs

from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import pickle
import shutil
import json
import json


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    deepmimic: bool = False
    mimic_task: str = "TwinkleTwinkleRousseau"
    midi_start_from: int = 0    
    residual_action: bool = False
    num_envs: int = 16
    pretrained: Optional[Path] = None
    initial_lr: float = 3e-4
    lr_decay_rate: float = 0.99
    residual_factor: float = 0.02
    n_steps: int = 512
    use_note_trajectory: bool = False
    mimic_z_axis: bool = False
    disable_hand_collisions: bool = True
    rsi: bool = False
    curriculum: bool = False
    total_iters: int = 1000
    # ============ Reward Shaping Options (Optional) ============
    # Velocity Smoothness: penalize rapid joint velocity changes
    reward_velocity_smoothness: bool = False
    reward_velocity_smoothness_coef: float = 0.01
    # Action Smoothness (Jerk Penalty): penalize large action changes
    reward_action_smoothness: bool = False
    reward_action_smoothness_coef: float = 0.01
    # Pre-positioning Reward: reward finger approach before note onset
    reward_prepositioning: bool = False
    reward_prepositioning_coef: float = 0.5
    reward_prepositioning_lookahead: int = 5
    # Finger Collision Penalty: penalize hand-hand/finger-finger collision
    reward_finger_collision: bool = False
    reward_finger_collision_coef: float = 0.5
    # Timing Reward: reward precise note onset timing
    reward_timing: bool = False
    reward_timing_coef: float = 0.5
    # Finger-to-Key Distance Reward: reward fingers close to assigned keys
    reward_finger_key_distance: bool = False
    reward_finger_key_distance_coef: float = 0.5

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"PPO-{args.environment_name}-{args.seed}-{time.time()}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "args.json", "w") as f:
        json.dump(asdict(args), f, indent=2, default=str)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # wandb.login()

    # wandb.init(
    #     project=args.project,
    #     config=asdict(args),
    #     name=run_name,
    #     sync_tensorboard=True,
    # )
    eval_args = copy(args)
    eval_args = replace(eval_args, rsi=False)
    eval_env = get_env(eval_args, record_dir=experiment_dir / "eval")
    eval_history_path = experiment_dir / "eval_history.jsonl"
    best_video_path = experiment_dir / "eval_best.mp4"
    best_f1 = -np.inf

    def make_env():
        env = get_env(args)
        return Monitor(env)
    # Parallel environments
    vec_env = SubprocVecEnv([make_envs(make_env, i) for i in range(args.num_envs)], start_method="fork")

    lr_scheduler_instance = lr_scheduler.LR_Scheduler(initial_lr=args.initial_lr,
                                                      decay_rate=args.lr_decay_rate,)

    policy_kwargs = dict(activation_fn=torch.nn.GELU,
                     net_arch=dict(pi=[1024, 256], vf=[1024, 256]))
    model = PPO("MlpPolicy", 
                vec_env, 
                n_epochs=10,
                n_steps=args.n_steps,
                batch_size=1024,
                learning_rate=lr_scheduler_instance.lr_schedule,
                policy_kwargs=policy_kwargs, 
                verbose=2,
                tensorboard_log="./robopianist_rl/tensorboard/{}".format(run_name),
                )
    if args.pretrained is not None:
        # Reload learning rate scheduler
        custom_objects = { 'learning_rate': lr_scheduler_instance.lr_schedule}
        model = PPO.load(args.pretrained, env=vec_env, custom_objects=custom_objects)
    # last_extending_curriculum_step = 0
    try:
        for i in range(args.total_iters):
            # Training
            model.learn(total_timesteps=args.n_steps*args.num_envs,
                        progress_bar=True,
                        reset_num_timesteps=False,
                        callback= None)
            # Evaluation (only every eval_interval iters, plus always on the last iter)
            if i % args.eval_interval == 0 or i == args.total_iters - 1:
                obs, _ = eval_env.reset()
                eval_reward = 0.0
                while True:
                    action, _state = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = eval_env.step(action)
                    eval_reward += float(reward)
                    if done == True:
                        break
                stats = eval_env.env.get_statistics()
                metrics = eval_env.env.get_musical_metrics()
                f1 = metrics["f1"]

                # Persist per-iteration metrics for plotting training curves later.
                log_entry = {
                    "iter": i,
                    "global_step": int((i + 1) * args.n_steps * args.num_envs),
                    "eval_reward": eval_reward,
                    "f1": float(f1),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "sustain_f1": float(metrics.get("sustain_f1", 0.0)) if metrics.get("sustain_f1") is not None else None,
                    "episode_length": stats.get("episode_length") if stats else None,
                }
                with open(eval_history_path, "a") as f_log:
                    f_log.write(json.dumps(log_entry) + "\n")

                video_file = eval_env.env.latest_filename
                if f1 > best_f1:
                    print("best_f1:{}->{}".format(best_f1, f1))
                    best_f1 = f1
                    model.save("./robopianist_rl/ckpts/{}_best".format(run_name))
                    # Preserve the best video (overwrite previous best)
                    if video_file is not None and Path(video_file).exists():
                        try:
                            shutil.copyfile(str(video_file), str(best_video_path))
                        except Exception as e:
                            print(f"[WARN] failed to copy best video: {e}")

                # Cleanup current iteration's video to avoid disk bloat
                if video_file is not None and Path(video_file).exists():
                    try:
                        Path(video_file).unlink()
                    except FileNotFoundError:
                        pass
    except KeyboardInterrupt:
        pass

    # model.save("./robopianist_rl/ckpts/{}".format(run_name))

    # Evaluate the trained model
    model = PPO.load("./robopianist_rl/ckpts/{}_best".format(run_name), env=vec_env)

    obs, _ = eval_env.reset()
    actions = []
    rewards = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _, info = eval_env.step(action)
        rewards += reward
        if done:
            break
    metrics = eval_env.env.get_musical_metrics()
    statistics = eval_env.env.get_statistics()

    # Preserve final eval video (separate from best video).
    final_video_path = experiment_dir / "eval_final.mp4"
    final_video_src = eval_env.env.latest_filename
    if final_video_src is not None and Path(final_video_src).exists():
        try:
            shutil.copyfile(str(final_video_src), str(final_video_path))
        except Exception as e:
            print(f"[WARN] failed to copy final video: {e}")

    result = {
        "run_name": run_name,
        "mimic_task": args.mimic_task,
        "total_reward": float(rewards),
        "best_f1": float(best_f1),
        "video": str(final_video_src) if final_video_src is not None else None,
        "best_video": str(best_video_path) if best_video_path.exists() else None,
        "final_video": str(final_video_path) if final_video_path.exists() else None,
        "eval_history": str(eval_history_path),
        "metrics": metrics,
        "statistics": statistics,
        "args": asdict(args),
    }
    with open(experiment_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Total reward: {rewards}")
    print(eval_env.env.latest_filename)
    print(metrics)
    actions = np.array(actions)
    os.makedirs(f"./trained_songs/{args.mimic_task}", exist_ok=True)
    np.save("./trained_songs/{}/actions_{}".format(args.mimic_task, args.mimic_task), actions)

    del model # remove to demonstrate saving and loading

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
