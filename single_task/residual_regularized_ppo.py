from typing import Generator, NamedTuple, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import explained_variance, get_schedule_fn


class ResidualRegularizedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    prev_observations: th.Tensor
    prev_valid: th.Tensor


class ResidualRegularizedRolloutBuffer(RolloutBuffer):
    """Rollout buffer that also exposes same-env previous observations."""

    prev_observations: np.ndarray
    prev_valid: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.prev_observations = np.zeros_like(self.observations)
        self.prev_valid = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        episode_start = np.asarray(episode_start, dtype=bool).reshape(self.n_envs)
        if self.pos > 0:
            self.prev_observations[self.pos] = self.observations[self.pos - 1]
            self.prev_valid[self.pos] = (~episode_start).astype(np.float32)
        else:
            self.prev_observations[self.pos] = np.asarray(obs).copy()
            self.prev_valid[self.pos] = 0.0
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[ResidualRegularizedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "prev_observations",
                "prev_valid",
            ]
            for tensor in tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[object] = None,
    ) -> ResidualRegularizedRolloutBufferSamples:
        del env
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.prev_observations[batch_inds],
            self.prev_valid[batch_inds].flatten(),
        )
        return ResidualRegularizedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class ResidualRegularizedPPO(PPO):
    """PPO with action-magnitude and adjacent-step smoothness penalties.

    The penalties are applied to the current policy's deterministic action
    (the distribution mode) so they have gradients. For PianoMime residual
    control, this corresponds to the residual correction before the sustain
    action unless a custom action dimension is provided.
    """

    def __init__(
        self,
        *args,
        residual_action_regularization: bool = False,
        residual_l2_coef: float = 0.0,
        residual_smooth_coef: float = 0.0,
        residual_regularization_action_dim: Optional[int] = None,
        residual_regularization_exclude_sustain: bool = True,
        dual_clip_coef: Optional[float] = None,
        advantage_clip: Optional[float] = None,
        log_std_min: Optional[float] = None,
        log_std_max: Optional[float] = None,
        policy_ema_decay: Optional[float] = None,
        **kwargs,
    ):
        if residual_l2_coef < 0.0 or residual_smooth_coef < 0.0:
            raise ValueError("Residual regularization coefficients must be non-negative.")
        if dual_clip_coef is not None and dual_clip_coef <= 1.0:
            raise ValueError("dual_clip_coef must be greater than 1.0 when enabled.")
        if advantage_clip is not None and advantage_clip <= 0.0:
            raise ValueError("advantage_clip must be positive when enabled.")
        if log_std_min is not None and log_std_max is not None and log_std_min > log_std_max:
            raise ValueError("log_std_min must be <= log_std_max.")
        if policy_ema_decay is not None and not 0.0 < policy_ema_decay < 1.0:
            raise ValueError("policy_ema_decay must be in (0, 1) when enabled.")
        self.residual_action_regularization = residual_action_regularization
        self.residual_l2_coef = residual_l2_coef
        self.residual_smooth_coef = residual_smooth_coef
        self.residual_regularization_action_dim = residual_regularization_action_dim
        self.residual_regularization_exclude_sustain = residual_regularization_exclude_sustain
        self.dual_clip_coef = dual_clip_coef
        self.advantage_clip = advantage_clip
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.policy_ema_decay = policy_ema_decay
        self._ema_policy_params: Optional[dict[str, th.Tensor]] = None
        self._regularized_action_dim: Optional[int] = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if isinstance(self.observation_space, spaces.Dict):
            if self._uses_residual_regularization:
                raise NotImplementedError("Residual action regularization currently expects a flat Box observation space.")
            buffer_cls = DictRolloutBuffer
        else:
            buffer_cls = ResidualRegularizedRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self._maybe_init_policy_ema()
        self._clamp_log_std()
        self._regularized_action_dim = self._resolve_regularized_action_dim()
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    @property
    def _uses_residual_regularization(self) -> bool:
        return bool(
            self.residual_action_regularization
            and (self.residual_l2_coef > 0.0 or self.residual_smooth_coef > 0.0)
        )

    def _resolve_regularized_action_dim(self) -> Optional[int]:
        if not self._uses_residual_regularization:
            return None
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("Residual action regularization requires a continuous Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if self.residual_regularization_action_dim is None:
            regularized_action_dim = action_dim - int(self.residual_regularization_exclude_sustain)
        else:
            regularized_action_dim = self.residual_regularization_action_dim
        if regularized_action_dim <= 0 or regularized_action_dim > action_dim:
            raise ValueError(
                "residual_regularization_action_dim must be in [1, action_dim]. "
                f"Got {regularized_action_dim} for action_dim={action_dim}."
            )
        return int(regularized_action_dim)

    def _policy_mode_actions(self, observations: th.Tensor | TensorDict) -> th.Tensor:
        return self.policy.get_distribution(observations).mode()

    def _regularized_actions(self, actions: th.Tensor) -> th.Tensor:
        if self._regularized_action_dim is None:
            raise RuntimeError("Residual action regularization is not configured.")
        return actions[:, : self._regularized_action_dim]

    def _clamp_log_std(self) -> None:
        if self.log_std_min is None and self.log_std_max is None:
            return
        if not hasattr(self.policy, "log_std"):
            return
        with th.no_grad():
            min_value = -float("inf") if self.log_std_min is None else self.log_std_min
            max_value = float("inf") if self.log_std_max is None else self.log_std_max
            self.policy.log_std.clamp_(min=min_value, max=max_value)

    def _maybe_init_policy_ema(self) -> None:
        if self.policy_ema_decay is None:
            return
        self._ema_policy_params = {
            name: param.detach().clone()
            for name, param in self.policy.named_parameters()
            if param.requires_grad
        }

    def _update_policy_ema(self) -> None:
        if self.policy_ema_decay is None:
            return
        if self._ema_policy_params is None:
            self._maybe_init_policy_ema()
            return
        with th.no_grad():
            for name, param in self.policy.named_parameters():
                if name not in self._ema_policy_params:
                    continue
                self._ema_policy_params[name].mul_(self.policy_ema_decay).add_(
                    param.detach(),
                    alpha=1.0 - self.policy_ema_decay,
                )

    def swap_to_ema_policy(self) -> Optional[dict[str, th.Tensor]]:
        if self._ema_policy_params is None:
            return None
        backup = {}
        with th.no_grad():
            for name, param in self.policy.named_parameters():
                if name not in self._ema_policy_params:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(self._ema_policy_params[name])
        return backup

    def restore_policy_parameters(self, backup: Optional[dict[str, th.Tensor]]) -> None:
        if backup is None:
            return
        with th.no_grad():
            for name, param in self.policy.named_parameters():
                if name in backup:
                    param.copy_(backup[name])

    def _compute_residual_regularization(
        self,
        observations: th.Tensor | TensorDict,
        prev_observations: th.Tensor | TensorDict,
        prev_valid: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, float]:
        zero = th.zeros((), device=self.device)
        residual_l2_loss = zero
        residual_smooth_loss = zero

        mode_actions = self._regularized_actions(self._policy_mode_actions(observations))
        if self.residual_l2_coef > 0.0:
            residual_l2_loss = mode_actions.pow(2).mean()

        smooth_valid_fraction = 0.0
        if self.residual_smooth_coef > 0.0:
            prev_mode_actions = self._regularized_actions(self._policy_mode_actions(prev_observations))
            per_sample_smooth_loss = (mode_actions - prev_mode_actions).pow(2).mean(dim=1)
            valid = prev_valid.float().reshape(-1)
            valid_sum = valid.sum()
            smooth_valid_fraction = float(valid.mean().detach().cpu().item()) if valid.numel() else 0.0
            if valid_sum > 0:
                residual_smooth_loss = (per_sample_smooth_loss * valid).sum() / valid_sum

        residual_regularization_loss = (
            self.residual_l2_coef * residual_l2_loss
            + self.residual_smooth_coef * residual_smooth_loss
        )
        return residual_regularization_loss, residual_l2_loss, residual_smooth_loss, smooth_valid_fraction

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        clipped_advantage_fractions = []
        residual_l2_losses = []
        residual_smooth_losses = []
        residual_regularization_losses = []
        smooth_valid_fractions = []
        ent_coef = float(self.ent_coef(self._current_progress_remaining)) if callable(self.ent_coef) else float(self.ent_coef)

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if self.advantage_clip is not None:
                    clipped_advantage_fractions.append(
                        th.mean((th.abs(advantages) > self.advantage_clip).float()).item()
                    )
                    advantages = th.clamp(advantages, -self.advantage_clip, self.advantage_clip)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                surrogate_loss = th.min(policy_loss_1, policy_loss_2)
                if self.dual_clip_coef is not None:
                    dual_clip_loss = self.dual_clip_coef * advantages
                    surrogate_loss = th.where(
                        advantages < 0,
                        th.max(surrogate_loss, dual_clip_loss),
                        surrogate_loss,
                    )
                policy_loss = -surrogate_loss.mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss

                if self._uses_residual_regularization:
                    if not hasattr(rollout_data, "prev_observations"):
                        raise RuntimeError("Residual regularization requires ResidualRegularizedRolloutBuffer samples.")
                    (
                        residual_regularization_loss,
                        residual_l2_loss,
                        residual_smooth_loss,
                        smooth_valid_fraction,
                    ) = self._compute_residual_regularization(
                        rollout_data.observations,
                        rollout_data.prev_observations,
                        rollout_data.prev_valid,
                    )
                    loss = loss + residual_regularization_loss
                    residual_l2_losses.append(residual_l2_loss.item())
                    residual_smooth_losses.append(residual_smooth_loss.item())
                    residual_regularization_losses.append(residual_regularization_loss.item())
                    smooth_valid_fractions.append(smooth_valid_fraction)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self._clamp_log_std()
                self._update_policy_ema()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/ent_coef", ent_coef)
        if self.dual_clip_coef is not None:
            self.logger.record("train/dual_clip_coef", self.dual_clip_coef)
        if self.advantage_clip is not None:
            self.logger.record("train/advantage_clip", self.advantage_clip)
            self.logger.record("train/advantage_clip_fraction", np.mean(clipped_advantage_fractions))
        if self._uses_residual_regularization:
            self.logger.record("train/residual_l2_loss", np.mean(residual_l2_losses))
            self.logger.record("train/residual_smooth_loss", np.mean(residual_smooth_losses))
            self.logger.record("train/residual_regularization_loss", np.mean(residual_regularization_losses))
            self.logger.record("train/residual_smooth_valid_fraction", np.mean(smooth_valid_fractions))
            self.logger.record("train/residual_l2_coef", self.residual_l2_coef)
            self.logger.record("train/residual_smooth_coef", self.residual_smooth_coef)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            if self.log_std_min is not None:
                self.logger.record("train/log_std_min_limit", self.log_std_min)
            if self.log_std_max is not None:
                self.logger.record("train/log_std_max_limit", self.log_std_max)
        if self.policy_ema_decay is not None:
            self.logger.record("train/policy_ema_decay", self.policy_ema_decay)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
