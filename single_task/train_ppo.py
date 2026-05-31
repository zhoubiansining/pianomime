import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import wandb
from datetime import datetime
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
from residual_regularized_ppo import ResidualRegularizedPPO


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_every_iters: int = 10
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
    mode: str = "online"
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
    residual_action_regularization: bool = False
    residual_l2_coef: float = 0.0
    residual_smooth_coef: float = 0.0
    residual_regularization_action_dim: Optional[int] = None
    residual_regularization_exclude_sustain: bool = True
    n_steps: int = 512
    use_note_trajectory: bool = False
    mimic_z_axis: bool = False
    disable_hand_collisions: bool = True
    rsi: bool = False
    curriculum: bool = False
    total_iters: int = 1000

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def main(args: Args) -> None:
    if args.residual_l2_coef < 0.0 or args.residual_smooth_coef < 0.0:
        raise ValueError("Residual regularization coefficients must be non-negative.")
    if args.residual_action_regularization and not args.residual_action:
        raise ValueError("--residual-action-regularization requires --residual-action.")
    if (
        not args.residual_action_regularization
        and (args.residual_l2_coef > 0.0 or args.residual_smooth_coef > 0.0)
    ):
        print("[ppo] residual regularization coefficients are set but the feature is disabled; ignoring them.")

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_name_parts = ["PPO", args.environment_name, str(args.seed)]
    if args.name:
        run_name_parts.append(args.name)
    run_name_parts.append(timestamp)
    run_name = "-".join(run_name_parts)

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "online":
        wandb_login_kwargs = {
            "host": os.environ.get("WANDB_HOST", "https://wandb.glm.ai")
        }
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb_login_kwargs["key"] = wandb_api_key
        wandb.login(**wandb_login_kwargs)

    wandb.init(
        project=args.project,
        entity=args.entity or None,
        config=asdict(args),
        name=run_name,
        notes=args.notes or None,
        tags=args.tags.split(",") if args.tags else None,
        sync_tensorboard=False,
        mode=args.mode,
        settings=wandb.Settings(x_disable_stats=True),
    )
    eval_args = copy(args)
    eval_args = replace(eval_args, rsi=False)
    eval_env = get_env(eval_args, record_dir=None, enable_midi_evaluation=True)
    def make_env():
        env = get_env(args)
        return Monitor(env)
    # Parallel environments
    vec_env = SubprocVecEnv([make_envs(make_env, i, seed=args.seed) for i in range(args.num_envs)], start_method="fork")

    lr_scheduler_instance = lr_scheduler.LR_Scheduler(initial_lr=args.initial_lr,
                                                      decay_rate=args.lr_decay_rate,)

    policy_kwargs = dict(activation_fn=torch.nn.GELU,
                     net_arch=dict(pi=[1024, 256], vf=[1024, 256]))
    ppo_cls = ResidualRegularizedPPO if args.residual_action_regularization else PPO
    model_kwargs = {}
    if args.residual_action_regularization:
        model_kwargs = {
            "residual_action_regularization": True,
            "residual_l2_coef": args.residual_l2_coef,
            "residual_smooth_coef": args.residual_smooth_coef,
            "residual_regularization_action_dim": args.residual_regularization_action_dim,
            "residual_regularization_exclude_sustain": args.residual_regularization_exclude_sustain,
        }
        print(
            "[ppo] residual action regularization enabled | "
            f"l2_coef={args.residual_l2_coef} | "
            f"smooth_coef={args.residual_smooth_coef} | "
            f"action_dim={args.residual_regularization_action_dim or 'auto'}"
        )
    model = ppo_cls("MlpPolicy",
                    vec_env,
                    n_epochs=10,
                    n_steps=args.n_steps,
                    batch_size=1024,
                    learning_rate=lr_scheduler_instance.lr_schedule,
                    policy_kwargs=policy_kwargs,
                    verbose=2,
                    tensorboard_log="./robopianist_rl/tensorboard/{}".format(run_name),
                    **model_kwargs,
                    )
    if args.pretrained is not None:
        # Reload learning rate scheduler
        custom_objects = { 'learning_rate': lr_scheduler_instance.lr_schedule}
        model = ppo_cls.load(args.pretrained, env=vec_env, custom_objects=custom_objects, **model_kwargs)
    best_f1 = -np.inf
    best_model_path = Path("./robopianist_rl/ckpts") / f"{run_name}_best"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    # last_extending_curriculum_step = 0
    last_iter = 0
    try:
        for i in range(args.total_iters):
            last_iter = i + 1
            # Training
            model.learn(total_timesteps=args.n_steps*args.num_envs, 
                        progress_bar=True,
                        reset_num_timesteps=False,
                        callback= None)
            should_eval = (
                args.eval_every_iters > 0
                and (i + 1) % args.eval_every_iters == 0
            )
            if not should_eval:
                print(
                    "[ppo] iter {}/{} | timesteps={} | eval=skipped | "
                    "next_eval_iter={}".format(
                        i + 1,
                        args.total_iters,
                        model.num_timesteps,
                        min(
                            args.total_iters,
                            ((i + 1) // args.eval_every_iters + 1)
                            * args.eval_every_iters,
                        ) if args.eval_every_iters > 0 else "disabled",
                    )
                )
                continue
            # Evaluation
            obs, _ = eval_env.reset()
            eval_reward = 0.0
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
                eval_reward += reward
                if done == True:
                    break
            log_dict = prefix_dict("eval", eval_env.env.get_statistics())
            metrics = eval_env.env.get_musical_metrics()
            music_dict = prefix_dict("eval", metrics)
            f1 = metrics["f1"]
            improved = f1 > best_f1
            if f1 > best_f1:
                print("best_f1:{}->{}".format(best_f1, f1))
                best_f1 = f1
                model.save(str(best_model_path))
            wandb_log = (
                log_dict
                | music_dict
                | {
                    "train/iter": last_iter,
                    "train/timesteps": model.num_timesteps,
                    "eval/reward": eval_reward,
                    "eval/best_f1": best_f1,
                    "eval/improved": improved,
                }
            )
            if args.deepmimic:
                wandb_log |= prefix_dict("eval", eval_env.env.get_deepmimic_rews())
            wandb.log(wandb_log, step=last_iter)
            print(
                "[ppo] iter {}/{} | timesteps={} | eval_reward={:.2f} | "
                "precision={:.4f} | recall={:.4f} | f1={:.4f} | "
                "best_f1={:.4f} | improved={}".format(
                    i + 1,
                    args.total_iters,
                    model.num_timesteps,
                    eval_reward,
                    metrics["precision"],
                    metrics["recall"],
                    f1,
                    best_f1,
                    improved,
                )
            )
    except KeyboardInterrupt:
        pass

    # model.save("./robopianist_rl/ckpts/{}".format(run_name))

    # Evaluate the trained model and record the final video.
    if best_f1 > -np.inf:
        model = ppo_cls.load(str(best_model_path), env=vec_env)
    else:
        print("[ppo] no eval checkpoint was saved; using the current model for final rollout")
    final_eval_env = get_env(eval_args, record_dir=experiment_dir / "eval")

    obs, _ = final_eval_env.reset()
    actions = []
    rewards = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _, info = final_eval_env.step(action)
        rewards += reward
        if done:
            break
    print(f"Total reward: {rewards}")
    print(final_eval_env.env.latest_filename)
    final_metrics = final_eval_env.env.get_musical_metrics()
    print(final_metrics)
    final_log = {
        "train/iter": last_iter,
        "train/timesteps": model.num_timesteps,
        "final/reward": rewards,
        **prefix_dict("final", final_metrics),
    }
    if args.mode != "disabled":
        final_log["final/video"] = wandb.Video(
            str(final_eval_env.env.latest_filename),
            fps=round(1.0 / args.control_timestep),
            format="mp4",
        )
    wandb.log(final_log, step=last_iter)
    actions = np.array(actions)
    output_dir = Path("./trained_songs") / args.mimic_task
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"actions_{args.mimic_task}", actions)

    del model # remove to demonstrate saving and loading
    wandb.finish()

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
