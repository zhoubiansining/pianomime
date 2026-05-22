"""A wrapper for residual learning framework."""
from controller.ik_controller import move_finger_to_key, move_fingers_to_keys, move_fingers_to_pos_qp
import collections
from typing import Any, Dict, Optional

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper
import math
from dm_control import mjcf
from dm_control.utils.rewards import tolerance
from dm_control.mujoco.wrapper import mjbindings
mjlib = mjbindings.mjlib

_FINGERTIP_CLOSE_ENOUGH = 0.01

class ResidualWrapper(EnvironmentWrapper):
    """Change step function."""
    def __init__(
        self,
        environment: dm_env.Environment,
        demonstrations_lh: np.ndarray,
        demonstrations_rh: np.ndarray,
        demo_ctrl_timestep: float = 0.05,
        rsi: bool = False,
        enable_ik: bool = True,
        external_demo: bool = False,
    ) -> None:
        super().__init__(environment)
        self._demonstrations_lh = demonstrations_lh
        self._demonstrations_rh = demonstrations_rh
        useful_columns = [0, 1, 2]
        self._demonstrations_lh = self._demonstrations_lh[:, useful_columns, :]
        self._demonstrations_rh = self._demonstrations_rh[:, useful_columns, :]
        self._step_scale = self._environment.task.control_timestep / demo_ctrl_timestep
        self._reference_frame_idx = -int(round(self._environment.task._initial_buffer_time/
                                        self._environment.task.control_timestep))
        self._rsi = rsi
        self._enable_ik = enable_ik
        assert self._demonstrations_lh.shape[0] == self._demonstrations_rh.shape[0]
        self._demonstrations_length = self._demonstrations_lh.shape[0]
        # Update the observation spec.
        self._wrapped_observation_spec = self._environment.observation_spec()
        self._observation_spec = collections.OrderedDict()
        self._observation_spec.update(self._wrapped_observation_spec)
        # Add the prior action observation.
        prior_action = np.zeros(self._environment.action_spec().shape[0]-1, dtype=np.float64)
        prior_action_spec = specs.Array(
            shape=prior_action.shape, dtype=prior_action.dtype, name='prior_action'
        )
        self._observation_spec['prior_action'] = prior_action_spec
        self._prior_action = None
        self._lh_target = None
        self._rh_target = None
        self._mimic_reward = 0
        self._external_demo = external_demo
        self.current_demo_lh = None
        self.current_demo_rh = None
        self._episode_start_idx = 0
        self._rng = np.random.default_rng()

    @property
    def episode_start_idx(self) -> int:
        return self._episode_start_idx

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def observation_spec(self):
        return self._observation_spec
    
    def _add_prior_action_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        prior_qpos = self._get_prior_action()
        self._prior_action = self.qpos2ctrl(prior_qpos)
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"prior_action": self._prior_action}
            )
        )
    
    def _add_demo_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        if self._external_demo:
            if self.current_demo_lh is not None and self.current_demo_rh is not None:
                demo_lh = self.current_demo_lh[0:self.task._n_steps_lookahead+1]
                demo_rh = self.current_demo_rh[0:self.task._n_steps_lookahead+1]
                self.current_demo_lh = None
                self.current_demo_rh = None
            else:
                raise ValueError("External demo is enabled but no demo is provided.")
        else:
            demo_lh = self._get_demo_window(self._demonstrations_lh)
            demo_rh = self._get_demo_window(self._demonstrations_rh)
        demo_lh = np.transpose(demo_lh, (0, 2, 1)).flatten()
        demo_rh = np.transpose(demo_rh, (0, 2, 1)).flatten()
        demo = np.concatenate((demo_lh, demo_rh)).flatten()
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"demo": demo}
            )
        ) 

    def set_current_demo(self, demonstrations_lh, demonstrations_rh):
        self.current_demo_lh = demonstrations_lh
        self.current_demo_rh = demonstrations_rh

    def _initial_reference_frame(self) -> int:
        return -int(round(self._environment.task._initial_buffer_time /
                          self._environment.task.control_timestep))

    def _reference_to_task_idx(self, reference_frame_idx: int) -> int:
        return int(max(0, round(reference_frame_idx / self._step_scale)))

    def _sample_rsi_reference_frame(self) -> int:
        max_demo_idx = self._demonstrations_length - 1
        max_task_idx = max(0, len(self.task._notes) - 1)
        max_reference_idx = min(max_demo_idx, int(max_task_idx * self._step_scale))
        return int(self._rng.integers(0, max_reference_idx + 1))

    def _set_episode_start(self, reference_frame_idx: int) -> None:
        self._reference_frame_idx = reference_frame_idx
        self._episode_start_idx = self._reference_to_task_idx(reference_frame_idx)
        self.task._t_idx = self._episode_start_idx
        self.task._episode_start_idx = self._episode_start_idx
        if hasattr(self._environment, "_reference_frame_idx"):
            self._environment._reference_frame_idx = reference_frame_idx
        if hasattr(self._environment, "_episode_start_idx"):
            self._environment._episode_start_idx = self._episode_start_idx

    def _get_demo_window(self, demonstrations: np.ndarray) -> np.ndarray:
        frames = []
        for idx in range(self._reference_frame_idx,
                         self._reference_frame_idx + self.task._n_steps_lookahead + 1):
            clamped_idx = min(max(idx, 0), self._demonstrations_length - 1)
            frames.append(demonstrations[clamped_idx])
        return np.stack(frames, axis=0)

    def _configure_hands_from_qpos(self, hand_qpos: np.ndarray) -> None:
        right_qpos, left_qpos = np.split(hand_qpos, 2)
        self.physics.bind(self.task.right_hand.joints).qpos = right_qpos
        self.physics.bind(self.task.left_hand.joints).qpos = left_qpos

        prior_action = self.qpos2ctrl(hand_qpos)
        right_action, left_action = np.split(prior_action, 2)
        self.physics.bind(self.task.right_hand.actuators).ctrl = right_action
        self.physics.bind(self.task.left_hand.actuators).ctrl = left_action
        self.physics.forward()

    def _refresh_task_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        observation = collections.OrderedDict(timestep.observation)
        if "goal" in observation:
            self.task._update_goal_state()
            observation["goal"] = self.task._goal_state.ravel()
        if "fingering" in observation:
            self.task._update_fingering_state()
            observation["fingering"] = self.task._fingering_state.ravel()
        hand_observables = [
            ("rh_shadow_hand/joints_pos", self.task.right_hand.observables.joints_pos),
            ("lh_shadow_hand/joints_pos", self.task.left_hand.observables.joints_pos),
            ("rh_shadow_hand/joints_vel", self.task.right_hand.observables.joints_vel),
            ("lh_shadow_hand/joints_vel", self.task.left_hand.observables.joints_vel),
        ]
        for key, observable in hand_observables:
            if key in observation:
                observation[key] = observable(self.physics).copy()
        timestep = timestep._replace(observation=observation)
        if hasattr(self._environment, "_add_demo_observation"):
            timestep = self._environment._add_demo_observation(timestep)
        return timestep

    def _get_prior_action(self) -> np.ndarray:
        if self._external_demo:
            if self.current_demo_lh is not None and self.current_demo_rh is not None:
                qvel_left, lh_dof_indices, self._lh_target = move_fingers_to_pos_qp(self,
                                            self.current_demo_lh[0],
                                            finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                            hand_side='left',
                                            targeting_wrist=True,
                                            )
                qvel_right, rh_dof_indices, self._rh_target = move_fingers_to_pos_qp(self,
                                            self.current_demo_rh[0],
                                            finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                            hand_side='right',
                                            targeting_wrist=True,
                                            )
            else:
                raise ValueError("External demo is enabled but no demo is provided.")
        else:
            qvel_left, lh_dof_indices, self._lh_target = move_fingers_to_pos_qp(self,
                                        self._demonstrations_lh[max(0, self._reference_frame_idx)],
                                        finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                        hand_side='left',
                                        targeting_wrist=True,
                                        )
            qvel_right, rh_dof_indices, self._rh_target = move_fingers_to_pos_qp(self,
                                        self._demonstrations_rh[max(0, self._reference_frame_idx)],
                                        finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                        hand_side='right',
                                        targeting_wrist=True,
                                        )
        v_full = np.zeros(self.physics.model.nv, dtype=self.physics.data.qpos.dtype)
        v_full[lh_dof_indices] = qvel_left
        v_full[rh_dof_indices] = qvel_right
        pos = self.physics.data.qpos.copy()
        mjlib.mj_integratePos(self.physics.model.ptr, pos, v_full, 0.05)
        return pos[88:]

    def qpos2ctrl(self, qpos):
        # Tendon is estimated by the sum of the two joint angles
        action = np.zeros(46, dtype=np.float64)
        action[0:2] = qpos[3:5]
        action[7:9] = qpos[5:7]
        action[9] = qpos[7] + qpos[8]
        action[10:12] = qpos[9:11]
        action[12] = qpos[11] + qpos[12]
        action[13:15] = qpos[13:15]
        action[15] = qpos[15] + qpos[16]
        action[16:19] = qpos[17:20]
        action[19] = qpos[20] + qpos[21]
        action[2:7] = qpos[22:27]
        action[20:23] = qpos[0:3]
        action[23:25] = qpos[30:32]
        action[30:32] = qpos[32:34]
        action[32] = qpos[34] + qpos[35]
        action[33:35] = qpos[36:38]
        action[35] = qpos[38] + qpos[39]
        action[36:38] = qpos[40:42]
        action[38] = qpos[42] + qpos[43]
        action[39:42] = qpos[44:47]
        action[42] = qpos[47] + qpos[48]
        action[25:30] = qpos[49:54]
        action[43:46] = qpos[27:30]
        return action

    def step(self, action) -> dm_env.TimeStep:
        if self._enable_ik:
            action_hand = action[:-1] + self._prior_action # Apply residual
        else:
            action_hand = action[:-1]
        self.non_residual_action = action_hand
        action_sustain = action[-1]
        # Merge action_sustain into action_hand 
        action = np.append(action_hand, action_sustain)
        timestep = self._environment.step(action)
        self._reference_frame_idx = int(min(self._reference_frame_idx+self._step_scale, self._demonstrations_length-1))
        return self._add_demo_observation(self._add_prior_action_observation(timestep))
    
    def get_non_residual_action(self):
        return self.non_residual_action

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        self._mimic_reward = 0
        if self._rsi:
            self._set_episode_start(self._sample_rsi_reference_frame())
            reference_joint_pos = self._get_prior_action()
            self._configure_hands_from_qpos(reference_joint_pos)
        else:
            self._set_episode_start(self._initial_reference_frame())
        timestep = self._refresh_task_observation(timestep)
        return self._add_demo_observation(self._add_prior_action_observation(timestep))
