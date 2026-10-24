from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  """Terminate when the 3D angle between gravity directions exceeds *threshold*.

  Projects the world gravity vector (0, 0, -1) into both the reference anchor
  frame and the robot's current anchor frame, then computes the angle between
  them via acos(dot).  Unlike bad_anchor_ori, which only compares the
  z component, this detects misalignment in any roll/pitch combination.

  The *dot* product is clamped to [-1, 1] for numerical safety.

  Args:
    threshold: Maximum allowed angle in **radians** before termination.
  """
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  dot = torch.sum(
    motion_projected_gravity_b * robot_projected_gravity_b, dim=1
  )
  dot = torch.clamp(dot, -1.0, 1.0)
  angle_error = torch.acos(dot)
  return angle_error > threshold


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)
