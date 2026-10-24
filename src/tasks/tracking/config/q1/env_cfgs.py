"""Q1 flat tracking environment configurations."""

from src.assets.robots import (
  Q1_ACTION_SCALE,
  get_q1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.tracking import mdp
from src.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from dataclasses import replace, field

from src.tasks.tracking.terrains.terrain_generator_cfg import Q1_BEYOND_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def q1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  copy_motion_end: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Q1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.njmax = 500
  cfg.scene.entities = {"robot": get_q1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = Q1_ACTION_SCALE
  # joint_pos_action.scale = 0.25

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.copy_motion_end = copy_motion_end
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  # events
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-5]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.15
  cfg.viewer.body_name = "torso_link"
  
  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  # scene
  # cfg.scene.terrain = TerrainEntityCfg(
  #       terrain_type="generator",
  #       terrain_generator=replace(ROUGH_TERRAINS_CFG),
  #       max_init_terrain_level=5,
  #     )

  return cfg

