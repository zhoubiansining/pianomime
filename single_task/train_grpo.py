import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
for path in (script_dir, repo_dir):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import tyro
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device, obs_as_tensor, set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

import lr_scheduler
from utils import get_env, make_envs

import os
import random


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
    disable_hand_collisions: bool = True
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
    rsi: bool = False
    curriculum: bool = False
    total_iters: int = 1000

    # GRPO-specific options.
    group_size: int = 4
    trajectory_horizon: int = 0
    n_epochs: int = 4
    clip_range: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    advantage_eps: float = 1e-8
    trajectory_batch_size: int = 0
    seed_stride: int = 100_000
    device: str = "auto"


@dataclass
class GRPORollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    traj_ids: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    completed: np.ndarray
    rollout_steps: int
    env_steps: int


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def _set_random_state_recursive(obj, seed: int, visited: Optional[set[int]] = None) -> None:
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    set_seed = getattr(obj, "set_seed", None)
    if callable(set_seed):
        try:
            set_seed(seed)
        except TypeError:
            pass

    if hasattr(obj, "_random_state"):
        try:
            obj._random_state = np.random.RandomState(seed)
        except Exception:
            pass

    for attr in ("env", "_env", "venv", "environment", "_environment"):
        child = getattr(obj, attr, None)
        if child is not None:
            _set_random_state_recursive(child, seed, visited)


class RecursiveSeedWrapper(gym.Wrapper):
    """Propagates Gym reset seeds into nested dm_control/RoboPianist wrappers."""

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            _set_random_state_recursive(self.env, int(seed))
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            return self.env.reset(seed=seed)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def make_group_reset_seeds(args: Args, iteration: int) -> list[int]:
    num_groups = args.num_envs // args.group_size
    seeds = []
    max_seed = 2**32 - 1
    for group_idx in range(num_groups):
        group_seed = (args.seed + iteration * args.seed_stride + group_idx) % max_seed
        seeds.extend([group_seed] * args.group_size)
    return seeds


def reset_vec_env_with_seeds(vec_env: SubprocVecEnv, seeds: list[int]) -> np.ndarray:
    if len(seeds) != vec_env.num_envs:
        raise ValueError(f"Expected {vec_env.num_envs} seeds, got {len(seeds)}.")
    vec_env._seeds = seeds
    return vec_env.reset()


def make_policy(vec_env: SubprocVecEnv, args: Args, device: torch.device) -> ActorCriticPolicy:
    lr_schedule = lambda _: args.initial_lr
    policy_kwargs = dict(
        activation_fn=torch.nn.GELU,
        net_arch=dict(pi=[1024, 256], vf=[1024, 256]),
    )
    policy = ActorCriticPolicy(
        vec_env.observation_space,
        vec_env.action_space,
        lr_schedule,
        use_sde=False,
        **policy_kwargs,
    )
    return policy.to(device)


def save_checkpoint(
    path: Path,
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    args: Args,
    run_name: str,
    best_f1: float,
    total_env_steps: int,
    iteration: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": asdict(args),
            "run_name": run_name,
            "best_f1": best_f1,
            "total_env_steps": total_env_steps,
            "iteration": iteration,
        },
        path,
    )


def load_checkpoint(path: Path, policy: ActorCriticPolicy, optimizer: Optional[torch.optim.Optimizer], device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def maybe_load_pretrained(
    pretrained: Optional[Path],
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    vec_env: SubprocVecEnv,
    device: torch.device,
) -> None:
    if pretrained is None:
        return

    candidate_paths = [Path(pretrained)]
    if Path(pretrained).suffix != ".zip":
        candidate_paths.append(Path(f"{pretrained}.zip"))

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            load_checkpoint(path, policy, optimizer, device)
            print(f"[grpo] loaded GRPO checkpoint from {path}")
            return
        except Exception:
            pass

        try:
            ppo_model = PPO.load(path, env=vec_env, device=device)
            policy.load_state_dict(ppo_model.policy.state_dict(), strict=False)
            print(f"[grpo] initialized policy from PPO checkpoint {path}")
            return
        except Exception:
            pass

    raise FileNotFoundError(f"Could not load pretrained checkpoint: {pretrained}")


def compute_group_advantages(returns: np.ndarray, group_size: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    grouped_returns = returns.reshape(-1, group_size)
    group_means = grouped_returns.mean(axis=1, keepdims=True)
    group_stds = grouped_returns.std(axis=1, keepdims=True)
    safe_stds = np.where(group_stds > eps, group_stds, 1.0)
    grouped_advantages = (grouped_returns - group_means) / safe_stds
    grouped_advantages = np.where(group_stds > eps, grouped_advantages, 0.0)
    return grouped_advantages.reshape(-1).astype(np.float32), group_stds.reshape(-1)


def collect_grpo_rollout(
    vec_env: SubprocVecEnv,
    policy: ActorCriticPolicy,
    args: Args,
    device: torch.device,
    iteration: int,
) -> GRPORollout:
    seeds = make_group_reset_seeds(args, iteration)
    obs = reset_vec_env_with_seeds(vec_env, seeds)

    n_envs = vec_env.num_envs
    active = np.ones(n_envs, dtype=bool)
    completed = np.zeros(n_envs, dtype=bool)
    returns = np.zeros(n_envs, dtype=np.float32)
    discount_factors = np.ones(n_envs, dtype=np.float32)
    lengths = np.zeros(n_envs, dtype=np.int32)

    observations = []
    actions = []
    old_log_probs = []
    traj_ids = []

    horizon = args.trajectory_horizon
    low = vec_env.action_space.low
    high = vec_env.action_space.high
    inactive_action = np.clip(np.zeros(vec_env.action_space.shape, dtype=np.float32), low, high)

    policy.set_training_mode(False)
    rollout_steps = 0
    while active.any() and (horizon <= 0 or rollout_steps < horizon):
        was_active = active.copy()
        active_ids = np.flatnonzero(was_active)

        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, device)
            sampled_actions, _, log_probs = policy(obs_tensor)
        sampled_actions_np = sampled_actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy().reshape(-1)

        env_actions = np.clip(sampled_actions_np, low, high)
        env_actions[~was_active] = inactive_action

        new_obs, rewards, dones, _ = vec_env.step(env_actions)

        observations.append(np.array(obs[active_ids]).copy())
        actions.append(np.array(sampled_actions_np[active_ids]).copy())
        old_log_probs.append(np.array(log_probs_np[active_ids]).copy())
        traj_ids.append(active_ids.astype(np.int32))

        returns[active_ids] += discount_factors[active_ids] * rewards[active_ids].astype(np.float32)
        discount_factors[active_ids] *= args.discount
        lengths[active_ids] += 1

        done_active = was_active & dones
        completed[done_active] = True
        active[done_active] = False

        obs = new_obs
        rollout_steps += 1

    flat_observations = np.concatenate(observations, axis=0)
    flat_actions = np.concatenate(actions, axis=0)
    flat_old_log_probs = np.concatenate(old_log_probs, axis=0).astype(np.float32)
    flat_traj_ids = np.concatenate(traj_ids, axis=0).astype(np.int32)

    trajectory_advantages, _ = compute_group_advantages(
        returns,
        args.group_size,
        args.advantage_eps,
    )
    flat_advantages = trajectory_advantages[flat_traj_ids]

    return GRPORollout(
        observations=flat_observations,
        actions=flat_actions,
        old_log_probs=flat_old_log_probs,
        advantages=flat_advantages,
        traj_ids=flat_traj_ids,
        returns=returns,
        lengths=lengths,
        completed=completed,
        rollout_steps=rollout_steps,
        env_steps=rollout_steps * n_envs,
    )


def trajectory_mean(values: torch.Tensor, traj_ids: torch.Tensor, selected_traj_ids: np.ndarray) -> torch.Tensor:
    means = []
    for traj_id in selected_traj_ids:
        mask = traj_ids == int(traj_id)
        if torch.any(mask):
            means.append(values[mask].mean())
    if not means:
        return values.mean()
    return torch.stack(means).mean()


def train_grpo(policy: ActorCriticPolicy, rollout: GRPORollout, args: Args, device: torch.device) -> dict:
    policy.set_training_mode(True)

    n_trajectories = args.num_envs
    trajectory_ids = np.arange(n_trajectories)
    trajectory_batch_size = args.trajectory_batch_size or n_trajectories
    trajectory_batch_size = min(trajectory_batch_size, n_trajectories)

    pg_losses = []
    entropy_losses = []
    clip_fractions = []
    approx_kl_divs = []

    continue_training = True
    for _ in range(args.n_epochs):
        np.random.shuffle(trajectory_ids)
        for start_idx in range(0, n_trajectories, trajectory_batch_size):
            selected_traj_ids = trajectory_ids[start_idx : start_idx + trajectory_batch_size]
            mask = np.isin(rollout.traj_ids, selected_traj_ids)
            if not np.any(mask):
                continue

            obs_tensor = torch.as_tensor(rollout.observations[mask], device=device)
            actions_tensor = torch.as_tensor(rollout.actions[mask], device=device)
            old_log_probs_tensor = torch.as_tensor(rollout.old_log_probs[mask], device=device)
            advantages_tensor = torch.as_tensor(rollout.advantages[mask], device=device)
            traj_ids_tensor = torch.as_tensor(rollout.traj_ids[mask], device=device)

            _, log_probs, entropy = policy.evaluate_actions(obs_tensor, actions_tensor)
            log_probs = log_probs.reshape(-1)
            log_ratio = log_probs - old_log_probs_tensor
            ratio = torch.exp(log_ratio)

            policy_loss_1 = advantages_tensor * ratio
            policy_loss_2 = advantages_tensor * torch.clamp(
                ratio,
                1.0 - args.clip_range,
                1.0 + args.clip_range,
            )
            per_step_policy_loss = -torch.min(policy_loss_1, policy_loss_2)
            policy_loss = trajectory_mean(per_step_policy_loss, traj_ids_tensor, selected_traj_ids)

            if entropy is None:
                per_step_entropy_loss = log_probs
            else:
                per_step_entropy_loss = -entropy.reshape(-1)
            entropy_loss = trajectory_mean(per_step_entropy_loss, traj_ids_tensor, selected_traj_ids)

            loss = policy_loss + args.ent_coef * entropy_loss

            with torch.no_grad():
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).cpu().item()
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > args.clip_range).float()).cpu().item()

            if args.target_kl is not None and approx_kl_div > 1.5 * args.target_kl:
                continue_training = False
                break

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            policy.optimizer.step()

            pg_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())
            clip_fractions.append(clip_fraction)
            approx_kl_divs.append(approx_kl_div)

        if not continue_training:
            break

    return {
        "policy_gradient_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
        "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0,
        "early_stop": not continue_training,
    }


def evaluate_policy(policy: ActorCriticPolicy, env) -> tuple[float, dict]:
    obs, _ = env.reset()
    eval_reward = 0.0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        eval_reward += reward
        if done:
            break
    return eval_reward, env.env.get_musical_metrics()


def main(args: Args) -> None:
    if args.group_size < 2:
        raise ValueError("GRPO requires --group-size >= 2.")
    if args.num_envs % args.group_size != 0:
        raise ValueError("--num-envs must be divisible by --group-size.")

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_name_parts = ["GRPO", args.environment_name, str(args.seed), f"G{args.group_size}"]
    if args.name:
        run_name_parts.append(args.name)
    run_name_parts.append(timestamp)
    run_name = "-".join(run_name_parts)

    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)
    device = get_device(args.device)

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

    eval_args = replace(args, rsi=False)
    eval_env = get_env(eval_args, record_dir=None, enable_midi_evaluation=True)

    def make_env():
        env = get_env(args)
        env = Monitor(env)
        return RecursiveSeedWrapper(env)

    vec_env = SubprocVecEnv(
        [make_envs(make_env, i, seed=args.seed) for i in range(args.num_envs)],
        start_method="fork",
    )

    policy = make_policy(vec_env, args, device)
    lr_scheduler_instance = lr_scheduler.LR_Scheduler(
        initial_lr=args.initial_lr,
        decay_rate=args.lr_decay_rate,
    )
    maybe_load_pretrained(args.pretrained, policy, policy.optimizer, vec_env, device)

    best_f1 = -np.inf
    best_model_path = Path("./robopianist_rl/ckpts") / f"{run_name}_best.pt"
    total_env_steps = 0
    last_iter = 0

    try:
        for i in range(args.total_iters):
            last_iter = i + 1
            learning_rate = lr_scheduler_instance.lr_schedule(0.0)
            set_optimizer_lr(policy.optimizer, learning_rate)

            rollout = collect_grpo_rollout(vec_env, policy, args, device, iteration=i)
            total_env_steps += rollout.env_steps
            train_stats = train_grpo(policy, rollout, args, device)

            trajectory_advantages, group_stds = compute_group_advantages(
                rollout.returns,
                args.group_size,
                args.advantage_eps,
            )
            rollout_log = {
                "train/iter": last_iter,
                "train/env_steps": total_env_steps,
                "train/rollout_env_steps": rollout.env_steps,
                "train/transitions": int(rollout.lengths.sum()),
                "train/rollout_steps": rollout.rollout_steps,
                "train/return_mean": float(rollout.returns.mean()),
                "train/return_std": float(rollout.returns.std()),
                "train/return_min": float(rollout.returns.min()),
                "train/return_max": float(rollout.returns.max()),
                "train/length_mean": float(rollout.lengths.mean()),
                "train/completion_rate": float(rollout.completed.mean()),
                "train/advantage_mean": float(trajectory_advantages.mean()),
                "train/advantage_std": float(trajectory_advantages.std()),
                "train/group_return_std_mean": float(group_stds.mean()),
                "train/zero_std_group_fraction": float((group_stds <= args.advantage_eps).mean()),
                "train/learning_rate": learning_rate,
                **prefix_dict("train", train_stats),
            }

            should_eval = (
                args.eval_every_iters > 0
                and (i + 1) % args.eval_every_iters == 0
            )
            if not should_eval:
                wandb.log(rollout_log, step=last_iter)
                print(
                    "[grpo] iter {}/{} | env_steps={} | return_mean={:.2f} | "
                    "return_std={:.2f} | len_mean={:.1f} | eval=skipped".format(
                        i + 1,
                        args.total_iters,
                        total_env_steps,
                        rollout.returns.mean(),
                        rollout.returns.std(),
                        rollout.lengths.mean(),
                    )
                )
                continue

            eval_reward, metrics = evaluate_policy(policy, eval_env)
            f1 = metrics["f1"]
            improved = f1 > best_f1
            if improved:
                print(f"best_f1:{best_f1}->{f1}")
                best_f1 = f1
                save_checkpoint(
                    best_model_path,
                    policy,
                    policy.optimizer,
                    args,
                    run_name,
                    best_f1,
                    total_env_steps,
                    last_iter,
                )

            wandb_log = (
                rollout_log
                | prefix_dict("eval", eval_env.env.get_statistics())
                | prefix_dict("eval", metrics)
                | {
                    "eval/reward": eval_reward,
                    "eval/best_f1": best_f1,
                    "eval/improved": improved,
                }
            )
            if args.deepmimic:
                wandb_log |= prefix_dict("eval", eval_env.env.get_deepmimic_rews())
            wandb.log(wandb_log, step=last_iter)

            print(
                "[grpo] iter {}/{} | env_steps={} | return_mean={:.2f} | "
                "eval_reward={:.2f} | precision={:.4f} | recall={:.4f} | "
                "f1={:.4f} | best_f1={:.4f} | improved={}".format(
                    i + 1,
                    args.total_iters,
                    total_env_steps,
                    rollout.returns.mean(),
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

    if best_f1 > -np.inf:
        load_checkpoint(best_model_path, policy, policy.optimizer, device)
    else:
        print("[grpo] no eval checkpoint was saved; using the current policy for final rollout")

    final_eval_env = get_env(eval_args, record_dir=experiment_dir / "eval")
    obs, _ = final_eval_env.reset()
    actions = []
    rewards = 0.0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _, _ = final_eval_env.step(action)
        rewards += reward
        if done:
            break

    print(f"Total reward: {rewards}")
    print(final_eval_env.env.latest_filename)
    final_metrics = final_eval_env.env.get_musical_metrics()
    print(final_metrics)

    final_log = {
        "train/iter": last_iter,
        "train/env_steps": total_env_steps,
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
    np.save(output_dir / f"actions_{args.mimic_task}_grpo", actions)

    vec_env.close()
    final_eval_env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
