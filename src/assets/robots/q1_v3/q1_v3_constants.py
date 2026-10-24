"""Q1 v3 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

Q1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "q1_v3" / "xml" / "q1_v3_mc_without_scene.xml"
)
assert Q1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, Q1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(Q1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

ARMATURE_ROB_S = 0.002

EFFORT_ROB_1 = 36
EFFORT_ROB_2 = 21

Q1_ACTUATOR_47 = DelayedActuatorCfg(
  base_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=[
      ".*_elbow_joint",
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_shoulder_yaw_joint",
    ],
    stiffness=20.0,
    damping=0.8,
    effort_limit=EFFORT_ROB_2,
    armature=ARMATURE_ROB_S,
  ),
  delay_min_lag=0,
  delay_max_lag=10,
)
Q1_ACTUATOR_59 = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=["waist_yaw_joint", ".*_hip_pitch_joint", ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_knee_joint"],
    stiffness=30.0,
    damping=1.5,
    effort_limit=EFFORT_ROB_1,
    armature=ARMATURE_ROB_S,
  ),
  delay_min_lag=0,
  delay_max_lag=10,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
Q1_ACTUATOR_WAIST = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=["waist_roll_joint"],
    stiffness=80.0,
    damping=3.5,
    effort_limit=EFFORT_ROB_1,
    armature=ARMATURE_ROB_S ,
  ),
  delay_min_lag=0,
  delay_max_lag=10,
)

Q1_ACTUATOR_ANKLE = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    stiffness=20.0,
    damping=0.8,
    effort_limit=EFFORT_ROB_2 ,
    armature=ARMATURE_ROB_S ,
  ),
  delay_min_lag=0,
  delay_max_lag=10,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.42),
  joint_pos={
    'left_hip_pitch_joint': -0.09,
    'left_hip_roll_joint': -0.02,
    'left_hip_yaw_joint': -0.02,
    'left_knee_joint': 0.22,
    'left_ankle_pitch_joint': -0.13,
    'left_ankle_roll_joint': 0.0,

    'right_hip_pitch_joint': -0.09,
    'right_hip_roll_joint': 0.02,
    'right_hip_yaw_joint': 0.02,
    'right_knee_joint': 0.22,
    'right_ankle_pitch_joint': -0.13,
    'right_ankle_roll_joint': 0.0,

    'waist_roll_joint': 0.0,
    'waist_yaw_joint': 0.0,

    'left_shoulder_pitch_joint': 0.0,
    'left_shoulder_roll_joint': 0.18,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 1.4,

    'right_shoulder_pitch_joint': 0.0,
    'right_shoulder_roll_joint': -0.18,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 1.4,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

Q1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    Q1_ACTUATOR_47,
    Q1_ACTUATOR_59,
    Q1_ACTUATOR_WAIST,
    Q1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_q1_robot_cfg() -> EntityCfg:
  """Get a fresh Q1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=Q1_ARTICULATION,
  )


Q1_ACTION_SCALE: dict[str, float] = {}
for a in Q1_ARTICULATION.actuators:
  names = a.target_names_expr
  for n in names:
    Q1_ACTION_SCALE[n] = 0.25 


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_q1_robot_cfg())

  viewer.launch(robot.spec.compile())
