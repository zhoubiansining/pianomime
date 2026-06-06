import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict, replace
import wandb
from datetime import datetime
import random
import numpy as np
import math
from tqdm import tqdm
import torch
from copy import copy

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
from stable_baselines3.common.utils import get_schedule_fn

import pickle
import shutil
from ppo_policies import LayerNormActorCriticPolicy
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
    batch_size: int = 1024
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
    key_press_positive_weight: float = 0.5
    key_press_negative_weight: float = 0.5
    key_press_negative_mode: str = "any"
    key_press_reward_scale: float = 2.0
    recall_activation_bonus_coef: float = 0.0
    missed_note_penalty_coef: float = 0.0
    residual_action_regularization: bool = False
    residual_l2_coef: float = 0.0
    residual_smooth_coef: float = 0.0
    residual_l2_coef_final: Optional[float] = None
    residual_smooth_coef_final: Optional[float] = None
    residual_regularization_action_dim: Optional[int] = None
    residual_regularization_exclude_sustain: bool = True
    n_steps: int = 512
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_final: Optional[float] = None
    clip_range_vf: Optional[float] = None
    clip_range_vf_final: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    ent_coef_final: Optional[float] = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    use_sde: bool = False
    sde_sample_freq: int = -1
    dual_clip_coef: Optional[float] = None
    advantage_clip: Optional[float] = None
    lr_schedule_type: str = "legacy"
    final_lr: Optional[float] = None
    lr_warmup_iters: int = 0
    lr_warmup_start_factor: float = 0.1
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    policy_ema_decay: Optional[float] = None
    eval_with_ema: bool = False
    policy_pi_arch: str = "1024,256"
    policy_vf_arch: str = "1024,256"
    activation_fn: str = "gelu"
    policy_layer_norm: bool = False
    layer_norm_eps: float = 1e-5
    ortho_init: bool = True
    log_std_init: float = 0.0
    checkpoint_metric: str = "f1"
    checkpoint_f_beta: float = 2.0
    use_note_trajectory: bool = False
    mimic_z_axis: bool = False
    disable_hand_collisions: bool = True
    rsi: bool = False
    curriculum: bool = False
    total_iters: int = 1000

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def parse_arch(arch: str) -> list[int]:
    layers = []
    for token in arch.split(","):
        token = token.strip()
        if not token:
            continue
        width = int(token)
        if width <= 0:
            raise ValueError(f"Policy architecture widths must be positive. Got {width}.")
        layers.append(width)
    if not layers:
        raise ValueError("Policy architecture must contain at least one layer width.")
    return layers


def get_activation_fn(name: str) -> type[torch.nn.Module]:
    activation_fns = {
        "gelu": torch.nn.GELU,
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "silu": torch.nn.SiLU,
    }
    key = name.lower()
    if key not in activation_fns:
        raise ValueError(f"Unsupported activation_fn={name!r}. Choose one of {sorted(activation_fns)}.")
    return activation_fns[key]


def linear_anneal(start: float, final: Optional[float], iteration: int, total_iters: int) -> float:
    if final is None:
        return start
    if total_iters <= 1:
        return final
    fraction = min(max(iteration / float(total_iters - 1), 0.0), 1.0)
    return start + (final - start) * fraction


def compute_current_lr(args: Args, iteration: int) -> float:
    schedule_type = args.lr_schedule_type.lower()
    if schedule_type == "legacy":
        raise ValueError("compute_current_lr should not be called for the legacy scheduler.")

    final_lr = args.final_lr if args.final_lr is not None else args.initial_lr
    if schedule_type == "constant":
        return args.initial_lr

    if args.lr_warmup_iters > 0 and iteration < args.lr_warmup_iters:
        start_lr = args.initial_lr * args.lr_warmup_start_factor
        warmup_fraction = (iteration + 1) / float(args.lr_warmup_iters)
        return start_lr + (args.initial_lr - start_lr) * warmup_fraction

    decay_iters = max(1, args.total_iters - max(args.lr_warmup_iters, 0) - 1)
    progress = min(max((iteration - args.lr_warmup_iters) / float(decay_iters), 0.0), 1.0)
    if schedule_type == "linear":
        return args.initial_lr + (final_lr - args.initial_lr) * progress
    if schedule_type == "cosine":
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (args.initial_lr - final_lr) * cosine_weight

    raise ValueError(f"Unsupported lr_schedule_type={args.lr_schedule_type!r}.")


def compute_checkpoint_score(metrics: dict, args: Args) -> float:
    metric = args.checkpoint_metric.lower()
    if metric == "f1":
        return float(metrics["f1"])
    if metric == "recall":
        return float(metrics["recall"])
    if metric == "fbeta":
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        beta_sq = args.checkpoint_f_beta ** 2
        denominator = beta_sq * precision + recall
        if denominator <= 0.0:
            return 0.0
        return (1.0 + beta_sq) * precision * recall / denominator
    raise ValueError(f"Unsupported checkpoint_metric={args.checkpoint_metric!r}.")


def validate_args(args: Args) -> None:
    supported_lr_schedules = {"legacy", "constant", "linear", "cosine"}
    supported_checkpoint_metrics = {"f1", "recall", "fbeta"}
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")
    if args.n_steps <= 0 or args.num_envs <= 0:
        raise ValueError("n_steps and num_envs must be positive.")
    if args.total_iters <= 0:
        raise ValueError("total_iters must be positive.")
    if args.clip_range <= 0.0:
        raise ValueError("clip_range must be positive.")
    if args.clip_range_final is not None and args.clip_range_final <= 0.0:
        raise ValueError("clip_range_final must be positive when set.")
    if args.clip_range_vf is not None and args.clip_range_vf <= 0.0:
        raise ValueError("clip_range_vf must be positive when set.")
    if args.clip_range_vf_final is not None and args.clip_range_vf_final <= 0.0:
        raise ValueError("clip_range_vf_final must be positive when set.")
    if args.clip_range_vf_final is not None and args.clip_range_vf is None:
        raise ValueError("clip_range_vf_final requires clip_range_vf to be set.")
    if args.ent_coef < 0.0:
        raise ValueError("ent_coef must be non-negative.")
    if args.ent_coef_final is not None and args.ent_coef_final < 0.0:
        raise ValueError("ent_coef_final must be non-negative when set.")
    if args.vf_coef < 0.0:
        raise ValueError("vf_coef must be non-negative.")
    if args.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive.")
    if not 0.0 < args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in (0, 1].")
    if args.target_kl is not None and args.target_kl <= 0.0:
        raise ValueError("target_kl must be positive when set.")
    if args.dual_clip_coef is not None and args.dual_clip_coef <= 1.0:
        raise ValueError("dual_clip_coef must be greater than 1.0 when set.")
    if args.advantage_clip is not None and args.advantage_clip <= 0.0:
        raise ValueError("advantage_clip must be positive when set.")
    if args.lr_schedule_type.lower() not in supported_lr_schedules:
        raise ValueError(f"lr_schedule_type must be one of {sorted(supported_lr_schedules)}.")
    if args.initial_lr <= 0.0:
        raise ValueError("initial_lr must be positive.")
    if args.final_lr is not None and args.final_lr <= 0.0:
        raise ValueError("final_lr must be positive when set.")
    if args.lr_warmup_iters < 0:
        raise ValueError("lr_warmup_iters must be non-negative.")
    if not 0.0 <= args.lr_warmup_start_factor <= 1.0:
        raise ValueError("lr_warmup_start_factor must be in [0, 1].")
    if args.key_press_positive_weight < 0.0 or args.key_press_negative_weight < 0.0:
        raise ValueError("Key press reward weights must be non-negative.")
    if args.key_press_negative_mode not in {"any", "fraction"}:
        raise ValueError("key_press_negative_mode must be one of ['any', 'fraction'].")
    if args.key_press_reward_scale < 0.0:
        raise ValueError("key_press_reward_scale must be non-negative.")
    if args.recall_activation_bonus_coef < 0.0 or args.missed_note_penalty_coef < 0.0:
        raise ValueError("Recall bonus and missed-note penalty coefficients must be non-negative.")
    if args.checkpoint_metric.lower() not in supported_checkpoint_metrics:
        raise ValueError(f"checkpoint_metric must be one of {sorted(supported_checkpoint_metrics)}.")
    if args.checkpoint_f_beta <= 0.0:
        raise ValueError("checkpoint_f_beta must be positive.")
    if args.log_std_min is not None and args.log_std_max is not None and args.log_std_min > args.log_std_max:
        raise ValueError("log_std_min must be <= log_std_max.")
    if args.policy_ema_decay is not None and not 0.0 < args.policy_ema_decay < 1.0:
        raise ValueError("policy_ema_decay must be in (0, 1) when set.")
    if args.eval_with_ema and args.policy_ema_decay is None:
        raise ValueError("--eval-with-ema requires --policy-ema-decay.")
    if args.layer_norm_eps <= 0.0:
        raise ValueError("layer_norm_eps must be positive.")
    if args.residual_l2_coef < 0.0 or args.residual_smooth_coef < 0.0:
        raise ValueError("Residual regularization coefficients must be non-negative.")
    if args.residual_l2_coef_final is not None and args.residual_l2_coef_final < 0.0:
        raise ValueError("residual_l2_coef_final must be non-negative when set.")
    if args.residual_smooth_coef_final is not None and args.residual_smooth_coef_final < 0.0:
        raise ValueError("residual_smooth_coef_final must be non-negative when set.")
    if args.residual_action_regularization and not args.residual_action:
        raise ValueError("--residual-action-regularization requires --residual-action.")
    if (
        not args.residual_action_regularization
        and (args.residual_l2_coef > 0.0 or args.residual_smooth_coef > 0.0)
    ):
        print("[ppo] residual regularization coefficients are set but the feature is disabled; ignoring them.")


def main(args: Args) -> None:
    validate_args(args)

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
    if args.lr_schedule_type.lower() == "legacy":
        learning_rate_schedule = lr_scheduler_instance.lr_schedule
    else:
        learning_rate_schedule = lambda _: args.initial_lr

    policy_kwargs = dict(
        activation_fn=get_activation_fn(args.activation_fn),
        net_arch=dict(pi=parse_arch(args.policy_pi_arch), vf=parse_arch(args.policy_vf_arch)),
        ortho_init=args.ortho_init,
        log_std_init=args.log_std_init,
    )
    policy_class = "MlpPolicy"
    if args.policy_layer_norm:
        policy_class = LayerNormActorCriticPolicy
        policy_kwargs["layer_norm_eps"] = args.layer_norm_eps
    ppo_cls = ResidualRegularizedPPO
    model_kwargs = {
        "residual_action_regularization": args.residual_action_regularization,
        "residual_l2_coef": args.residual_l2_coef,
        "residual_smooth_coef": args.residual_smooth_coef,
        "residual_regularization_action_dim": args.residual_regularization_action_dim,
        "residual_regularization_exclude_sustain": args.residual_regularization_exclude_sustain,
        "dual_clip_coef": args.dual_clip_coef,
        "advantage_clip": args.advantage_clip,
        "log_std_min": args.log_std_min,
        "log_std_max": args.log_std_max,
        "policy_ema_decay": args.policy_ema_decay,
    }
    if args.residual_action_regularization:
        print(
            "[ppo] residual action regularization enabled | "
            f"l2_coef={args.residual_l2_coef} | "
            f"smooth_coef={args.residual_smooth_coef} | "
            f"action_dim={args.residual_regularization_action_dim or 'auto'}"
        )
    if args.dual_clip_coef is not None:
        print(f"[ppo] dual-clip enabled | dual_clip_coef={args.dual_clip_coef}")
    if args.advantage_clip is not None:
        print(f"[ppo] advantage clipping enabled | advantage_clip={args.advantage_clip}")
    if args.policy_layer_norm:
        print(f"[ppo] layer norm policy enabled | layer_norm_eps={args.layer_norm_eps}")
    if args.lr_schedule_type.lower() != "legacy":
        print(
            "[ppo] global lr schedule enabled | "
            f"type={args.lr_schedule_type} | initial_lr={args.initial_lr} | "
            f"final_lr={args.final_lr if args.final_lr is not None else args.initial_lr} | "
            f"warmup_iters={args.lr_warmup_iters}"
        )
    if args.log_std_min is not None or args.log_std_max is not None:
        print(f"[ppo] log_std clamp enabled | min={args.log_std_min} | max={args.log_std_max}")
    if args.policy_ema_decay is not None:
        print(f"[ppo] policy EMA enabled | decay={args.policy_ema_decay} | eval_with_ema={args.eval_with_ema}")
    if args.checkpoint_metric.lower() != "f1":
        print(
            "[ppo] recall-oriented checkpoint selection enabled | "
            f"metric={args.checkpoint_metric} | beta={args.checkpoint_f_beta}"
        )
    model = ppo_cls(policy_class,
                    vec_env,
                    n_epochs=args.n_epochs,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    gamma=args.discount,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    clip_range_vf=args.clip_range_vf,
                    normalize_advantage=args.normalize_advantage,
                    ent_coef=args.ent_coef,
                    vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm,
                    use_sde=args.use_sde,
                    sde_sample_freq=args.sde_sample_freq,
                    target_kl=args.target_kl,
                    learning_rate=learning_rate_schedule,
                    policy_kwargs=policy_kwargs,
                    verbose=2,
                    seed=args.seed,
                    tensorboard_log="./robopianist_rl/tensorboard/{}".format(run_name),
                    **model_kwargs,
                    )
    if args.pretrained is not None:
        # Reload learning rate scheduler
        custom_objects = {
            "learning_rate": learning_rate_schedule,
            "clip_range": args.clip_range,
            "clip_range_vf": args.clip_range_vf,
            "ent_coef": args.ent_coef,
        }
        model = ppo_cls.load(args.pretrained, env=vec_env, custom_objects=custom_objects, **model_kwargs)
    best_f1 = -np.inf
    best_checkpoint_score = -np.inf
    best_model_path = Path("./robopianist_rl/ckpts") / f"{run_name}_best"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    # last_extending_curriculum_step = 0
    last_iter = 0
    try:
        for i in range(args.total_iters):
            last_iter = i + 1
            current_clip_range = linear_anneal(args.clip_range, args.clip_range_final, i, args.total_iters)
            current_clip_range_vf = (
                linear_anneal(args.clip_range_vf, args.clip_range_vf_final, i, args.total_iters)
                if args.clip_range_vf is not None
                else None
            )
            current_ent_coef = linear_anneal(args.ent_coef, args.ent_coef_final, i, args.total_iters)
            current_residual_l2_coef = linear_anneal(
                args.residual_l2_coef,
                args.residual_l2_coef_final,
                i,
                args.total_iters,
            )
            current_residual_smooth_coef = linear_anneal(
                args.residual_smooth_coef,
                args.residual_smooth_coef_final,
                i,
                args.total_iters,
            )
            model.clip_range = get_schedule_fn(current_clip_range)
            model.clip_range_vf = get_schedule_fn(current_clip_range_vf) if current_clip_range_vf is not None else None
            model.ent_coef = current_ent_coef
            model.residual_l2_coef = current_residual_l2_coef
            model.residual_smooth_coef = current_residual_smooth_coef
            if args.lr_schedule_type.lower() != "legacy":
                current_learning_rate = compute_current_lr(args, i)
                model.lr_schedule = get_schedule_fn(current_learning_rate)
            else:
                current_learning_rate = model.policy.optimizer.param_groups[0]["lr"]

            # Training
            model.learn(total_timesteps=args.n_steps*args.num_envs, 
                        progress_bar=args.tqdm_bar,
                        reset_num_timesteps=False,
                        callback= None)
            current_learning_rate = model.policy.optimizer.param_groups[0]["lr"]
            should_eval = (
                args.eval_every_iters > 0
                and (i + 1) % args.eval_every_iters == 0
            )
            if not should_eval:
                print(
                    "[ppo] iter {}/{} | timesteps={} | eval=skipped | "
                    "next_eval_iter={} | lr={:.8f} | clip_range={:.4f} | ent_coef={:.6f}".format(
                        i + 1,
                        args.total_iters,
                        model.num_timesteps,
                        min(
                            args.total_iters,
                            ((i + 1) // args.eval_every_iters + 1)
                            * args.eval_every_iters,
                        ) if args.eval_every_iters > 0 else "disabled",
                        current_learning_rate,
                        current_clip_range,
                        current_ent_coef,
                    )
                )
                continue
            # Evaluation
            ema_backup = model.swap_to_ema_policy() if args.eval_with_ema else None
            try:
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
                checkpoint_score = compute_checkpoint_score(metrics, args)
                if f1 > best_f1:
                    print("best_f1:{}->{}".format(best_f1, f1))
                    best_f1 = f1
                improved = checkpoint_score > best_checkpoint_score
                if improved:
                    print(
                        "best_checkpoint_score:{}->{} ({})".format(
                            best_checkpoint_score,
                            checkpoint_score,
                            args.checkpoint_metric,
                        )
                    )
                    best_checkpoint_score = checkpoint_score
                    model.save(str(best_model_path))
            finally:
                model.restore_policy_parameters(ema_backup)
            wandb_log = (
                log_dict
                | music_dict
                | {
                    "train/iter": last_iter,
                    "train/timesteps": model.num_timesteps,
                    "train/current_learning_rate": current_learning_rate,
                    "train/current_clip_range": current_clip_range,
                    "train/current_ent_coef": current_ent_coef,
                    "train/current_residual_l2_coef": current_residual_l2_coef,
                    "train/current_residual_smooth_coef": current_residual_smooth_coef,
                    "eval/ema_policy": args.eval_with_ema,
                    "eval/reward": eval_reward,
                    "eval/checkpoint_score": checkpoint_score,
                    "eval/best_checkpoint_score": best_checkpoint_score,
                    "eval/best_f1": best_f1,
                    "eval/improved": improved,
                }
            )
            if current_clip_range_vf is not None:
                wandb_log["train/current_clip_range_vf"] = current_clip_range_vf
            if args.deepmimic:
                wandb_log |= prefix_dict("eval", eval_env.env.get_deepmimic_rews())
            wandb.log(wandb_log, step=last_iter)
            print(
                "[ppo] iter {}/{} | timesteps={} | eval_reward={:.2f} | "
                "precision={:.4f} | recall={:.4f} | f1={:.4f} | "
                "best_f1={:.4f} | checkpoint_metric={} | checkpoint_score={:.4f} | "
                "best_checkpoint_score={:.4f} | improved={} | lr={:.8f} | "
                "clip_range={:.4f} | ent_coef={:.6f}".format(
                    i + 1,
                    args.total_iters,
                    model.num_timesteps,
                    eval_reward,
                    metrics["precision"],
                    metrics["recall"],
                    f1,
                    best_f1,
                    args.checkpoint_metric,
                    checkpoint_score,
                    best_checkpoint_score,
                    improved,
                    current_learning_rate,
                    current_clip_range,
                    current_ent_coef,
                )
            )
    except KeyboardInterrupt:
        pass

    # model.save("./robopianist_rl/ckpts/{}".format(run_name))

    # Evaluate the trained model and record the final video.
    if best_checkpoint_score > -np.inf:
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
